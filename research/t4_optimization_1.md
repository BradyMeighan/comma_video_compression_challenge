Let me read your inflate.py first — that's where the optimization loop lives.Now let me read the context.md section I missed:Okay Brady, I've read through everything. Here's my analysis of the highest-impact optimizations, ordered by expected bang-for-buck on T4.

---

**1. Kill the `.item()` CPU-GPU syncs in the hot loop (easiest, ~5-10%)**

Lines 288-290 call `.item()` every single iteration, which forces a CUDA synchronize. On T4, each sync costs 0.1-0.5ms. Over 100 iterations that's 20-100ms wasted per batch just waiting for the GPU to finish so you can read a single float.

Fix: only pull values to CPU every N iterations, and use GPU-side tensors for the early stopping check:

```python
# Instead of:
seg_loss_val = seg_l.item()
if run_pose:
    pose_loss_val = pose_l.item()

# Do: check every 5 iterations
if it % 5 == 0 or it >= num_iters - AVG_LAST_K:
    seg_loss_val = seg_l.item()
    if run_pose:
        pose_loss_val = pose_l.item()
```

Or even better for the early stopping check, stay on GPU entirely:

```python
# GPU-side early stopping (no sync):
if it >= EARLY_STOP_MIN and not early_stop_triggered and run_pose:
    converged = (seg_l < EARLY_STOP_SEG_THRESH) & (pose_l < EARLY_STOP_POSE_THRESH)
    if converged:  # bool tensor, but the `if` does one sync — still better than 2 .item() calls/iter
        converged_count += 1
    ...
```

**2. Fused AdamW (trivial change, ~3-5%)**

```python
optimizer = torch.optim.AdamW(
    [frame_0, frame_1], lr=lr, weight_decay=0.025, betas=(0.9, 0.88),
    fused=True  # <-- single CUDA kernel for the entire update
)
```

This is available in PyTorch 2.0+. Eliminates multiple kernel launches per optimizer step. Since you only have 2 parameters, the launch overhead is proportionally large vs. actual compute.

**3. Don't optimize frame_0 during SegNet-only phase (~5-8% for early iters)**

During iterations 0 through `early_cutoff`, frame_0 receives zero gradient (SegNet only uses frame_1). But AdamW still runs weight decay and momentum updates on it, wasting kernel launches and bandwidth.

```python
# Use two param groups:
seg_only_optimizer = torch.optim.AdamW([frame_1], lr=lr, ..., fused=True)
# Then at early_cutoff, switch to full optimizer:
full_optimizer = torch.optim.AdamW([frame_0, frame_1], lr=lr, ..., fused=True)
```

Or simpler: just swap the optimizer's param_groups at the transition point, seeding frame_0's state from scratch (it won't have stale momentum since it never got gradients).

**4. CUDA Graphs (~15-25% — the big one, but needs some work)**

Your optimization loop has identical tensor shapes every iteration. CUDA graphs capture the entire forward-backward-step sequence as a single GPU-side replay, eliminating all CPU orchestration and kernel launch overhead.

You need two graphs: one for SegNet-only iterations (early phase) and one for SegNet+PoseNet (main phase).

```python
# Warmup phase (required before capture):
for _ in range(3):
    # run one iteration normally to warm up cudnn, allocator, etc.
    ...

# Capture SegNet-only graph:
seg_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(seg_graph):
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda'):
        seg_logits = segnet(frame_1)
        seg_l = margin_loss(seg_logits, target_seg, seg_margin)
        total = alpha * seg_l
    scaler.scale(total).backward()
    scaler.step(optimizer)
    scaler.update()
    frame_1.data.clamp_(0, 255)

# In the loop, just replay:
for it in range(num_iters):
    if it < early_cutoff:
        seg_graph.replay()
    else:
        full_graph.replay()
```

The catch: you can't do conditional logic, `.item()`, or dynamic shapes inside the captured graph. But your loop is already fixed-shape, and if you move the `.item()` calls outside the graph (check every N iterations with a manual sync), this is very feasible. The loss scaling from dynamic phases (a_eff, b_eff) can be handled by writing new values into the pre-allocated tensors before replaying.

On T4 specifically, where kernel launch latency is ~10μs and you launch hundreds of kernels per iteration, this can eliminate a huge chunk of overhead.

**5. `torch.compile` with `reduce-overhead` on a step function (alternative to #4)**

If CUDA graphs feel too manual, `torch.compile(mode="reduce-overhead")` essentially does CUDA graphs under the hood. The key insight you may have missed: **compile once, amortize across all 38 batches**. If compilation takes 90 seconds but saves 15% per iteration across 38 batches × 90 iters × 0.4s = ~200 seconds saved, you net ~110 seconds — that's ~7 extra iterations per batch.

```python
@torch.compile(mode="reduce-overhead")
def seg_step(frame_1, target_seg, seg_margin, a_eff):
    seg_logits = segnet(frame_1)
    seg_l = margin_loss(seg_logits, target_seg, seg_margin)
    total = a_eff * seg_l
    return total, seg_l

# The compile happens on first call, then every subsequent call is fast
```

The important thing is to not compile the entire `optimize_batch` function — just the inner compute. And make sure `segnet`/`posenet` are captured as constants in the closure so they don't trigger recompilation.

**6. Accumulate scheduler.step() (minor, ~1-2%)**

You're calling `scheduler.step()` inside a `torch.no_grad()` block every iteration. The cosine annealing schedule is pure math — precompute all LR values into a tensor and just index into it:

```python
lr_schedule = [lr * 0.05 + 0.5 * (lr - lr * 0.05) * (1 + math.cos(math.pi * i / num_iters))
               for i in range(num_iters)]
# In loop:
for pg in optimizer.param_groups:
    pg['lr'] = lr_schedule[it]
```

Eliminates scheduler object overhead entirely.

**7. Algorithmic: SGD+Nesterov instead of AdamW**

AdamW stores 2 state tensors per parameter (first + second moment). SGD+momentum stores 1. For your 2 small parameters this isn't about memory — it's about the per-step compute. SGD's update rule is ~2x fewer FLOPs than AdamW's. Given that optimizer overhead is ~20% of your iteration time, this could shave 5-10%.

The risk is convergence speed. But for adversarial pixel optimization from a good initialization (ideal_colors), SGD with aggressive momentum often works well. Worth a quick A/B test:

```python
optimizer = torch.optim.SGD(
    [frame_0, frame_1], lr=lr * 5, momentum=0.9, nesterov=True
)
```

(You'll need to crank the LR since SGD typically needs higher LR than AdamW.)

**8. Reduce margin_loss allocation**

Your `margin_loss` clones the entire logits tensor every iteration:

```python
competitor = logits.clone()  # allocates (B, 5, 384, 512) every call
competitor.scatter_(1, target.unsqueeze(1), float('-inf'))
```

Replace with:

```python
def margin_loss(logits, target, margin=3.0):
    target_logits = logits.gather(1, target.unsqueeze(1))  # (B, 1, H, W)
    # Mask target class without cloning full tensor:
    mask = torch.arange(logits.shape[1], device=logits.device).view(1, -1, 1, 1) == target.unsqueeze(1).unsqueeze(1)
    max_other = logits.masked_fill(mask, float('-inf')).max(dim=1, keepdim=True).values
    return F.relu(max_other - target_logits + margin).mean()
```

This avoids a full-tensor clone per iteration. On T4 that's (16 × 5 × 384 × 512) × 4 bytes = ~60MB allocation/deallocation per iteration.

---

**Priority order for implementation:**

1. Remove `.item()` syncs — 5 minutes of work, immediate payoff
2. `fused=True` on AdamW — one-word change
3. Fix margin_loss clone — 10 minutes
4. Precompute LR schedule — 5 minutes
5. Skip frame_0 in early phase — 15 minutes
6. CUDA graphs OR torch.compile reduce-overhead — 1-2 hours, but the biggest single win

Stacking #1-5 should get you ~15-20% faster per iteration. Adding #6 on top could push to 30-40%. That would get you from 85-95 iterations to ~110-130 on T4, which should meaningfully close the gap toward 3090 scores.

Want me to write a patched version of the optimization loop with all of these applied?