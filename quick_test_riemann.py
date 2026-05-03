#!/usr/bin/env python
"""Quick adversarial decode test for Riemannian checkpoint."""
import sys, time, math
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn.functional as F, numpy as np
from safetensors.torch import load_file
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from frame_utils import camera_size, segnet_model_input_size
from train_distill import posenet_preprocess_diff, margin_loss
from test_riemannian import replace_with_lowrank, IDEAL_COLORS

device = torch.device('cuda')

print('Loading Riemannian SegNet...')
model = SegNet().to(device)
model.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
replace_with_lowrank(model, rank_ratio=0.3)
model.load_state_dict(torch.load('compressed_models/segnet_riemannian.pt',
                                  map_location=device, weights_only=True))
model.eval()
for p in model.parameters(): p.requires_grad_(False)

print('Loading teachers...')
t_seg = SegNet().eval().to(device)
t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
t_pose = PoseNet().eval().to(device)
t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
for p in t_seg.parameters(): p.requires_grad_(False)
for p in t_pose.parameters(): p.requires_grad_(False)

seg_logits = torch.load('distill_data/seg_logits.pt', weights_only=True)
ta = seg_logits.argmax(1).to(device)
pt = torch.load('distill_data/pose_outputs.pt', weights_only=True).to(device)
del seg_logits

colors = IDEAL_COLORS.to(device)
mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
Wc, Hc = camera_size
results = {'ts': [], 'tp': []}

for st in [0, 200, 400]:
    end = min(st+4, 600); tgt_s = ta[st:end]; tgt_p = pt[st:end]; B = tgt_s.shape[0]
    init = colors[tgt_s].permute(0,3,1,2).clone()
    f1 = init.requires_grad_(True)
    f0 = init.detach().mean(dim=(-2,-1),keepdim=True).expand_as(init).clone().requires_grad_(True)
    opt = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
    lr_s = [0.06+0.57*(1+math.cos(math.pi*i/149)) for i in range(150)]
    print(f'Batch {st}: ', end='', flush=True)
    for it in range(150):
        for pg in opt.param_groups: pg['lr'] = lr_s[it]
        opt.zero_grad(set_to_none=True)
        p = it/149
        seg_l = margin_loss(model(f1), tgt_s, 0.1 if p<0.5 else 5.0)
        if p >= 0.3:
            both = torch.stack([f0, f1], dim=1)
            pn_in = posenet_preprocess_diff(both)
            pose_l = F.smooth_l1_loss(t_pose(pn_in)['pose'][:,:6], tgt_p)
            total = 120*seg_l + 0.2*pose_l
        else: total = 120*seg_l
        total.backward()
        opt.step()
        with torch.no_grad(): f0.data.clamp_(0,255); f1.data.clamp_(0,255)
        if (it+1)%50==0: print(f'{it+1}', end=' ', flush=True)
    print(flush=True)
    with torch.no_grad():
        f1u = F.interpolate(f1.data,(Hc,Wc),mode='bicubic',align_corners=False).clamp(0,255).round().byte().float()
        f0u = F.interpolate(f0.data,(Hc,Wc),mode='bicubic',align_corners=False).clamp(0,255).round().byte().float()
        ts_in = F.interpolate(f1u,(mH,mW),mode='bilinear')
        results['ts'].extend((t_seg(ts_in).argmax(1)!=tgt_s).float().mean((1,2)).cpu().tolist())
        tp_pair = F.interpolate(torch.stack([f0u,f1u],1).reshape(-1,3,Hc,Wc),(mH,mW),mode='bilinear').reshape(B,2,3,mH,mW)
        tpo = t_pose(posenet_preprocess_diff(tp_pair))['pose'][:,:6]
        results['tp'].extend((tpo-tgt_p).pow(2).mean(1).cpu().tolist())
    print(f'  seg={np.mean(results["ts"][-B:]):.6f} pose={np.mean(results["tp"][-B:]):.6f}')
    del f0,f1,opt; torch.cuda.empty_cache()

s=100*np.mean(results['ts']); p=math.sqrt(10*np.mean(results['tp']))
print(f'\n100*seg={s:.4f} sqrt(10*p)={p:.4f} distortion={s+p:.4f}')
print(f'Non-factored SVD (proven): 1.31')
print(f'Factored SVD (broken): 26.96')
print(f'Leader: 1.95')
