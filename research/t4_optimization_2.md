# **Advanced GPU Acceleration and Optimization Strategies for Adversarial Video Compression on NVIDIA T4**

## **Architectural Context and Objective Function Topology**

The optimization of pixel values through deep neural networks using frozen weight matrices represents a highly specialized paradigm within computational machine learning. In the context of the comma.ai video compression challenge, the objective is to compress a 60-second dashcam video sequence while minimizing the mathematical distortion of the reconstructed frames.1 This distortion is not evaluated through traditional pixel-space metrics such as Peak Signal-to-Noise Ratio (PSNR) or Structural Similarity Index Measure (SSIM). Instead, the challenge relies on the perceptual and temporal evaluations of two distinct, frozen neural networks: a segmentation model (SegNet) and an ego-motion estimation model (PoseNet).1

The optimization loop operates dynamically at decode time, executing gradient descent directly on the decoded image frames to invert the networks. By iteratively refining the pixel values of the reconstructed frames, the system generates adversarial configurations that force the evaluation networks to output the correct target distributions, effectively hiding extreme compression artifacts from the neural evaluators.1

The primary barrier to achieving a superior leaderboard score is the stringent evaluation environment. Submissions are strictly constrained to an automated continuous integration (CI) pipeline running on an NVIDIA T4 GPU with 16GB of VRAM, subject to a hard 30-minute total execution timeout.1 Within this pipeline, the system currently processes 600 frame pairs across 38 batches of 16 pairs each.1 While a local NVIDIA RTX 3090 GPU easily achieves 150 iterations per batch, the T4 GPU manages only 85 to 95 iterations within the allotted time, resulting in a suboptimal score of 0.39 to 0.40.1 Because the relationship between the number of iterations and the final distortion score is functionally linear within this specific operational band, maximizing the iterations per second (it/s) or fundamentally altering the convergence efficiency of the optimization step is imperative.1

The competition evaluates submissions utilizing an asymmetric, heavily weighted scoring formula. The final score is computed as the sum of three distinct components: the segmentation distortion multiplied by 100, the compression rate multiplied by 25, and the square root of the PoseNet distortion multiplied by 10\.1

| Evaluation Component | Mathematical Definition | Scoring Weight | Landscape Topology |
| :---- | :---- | :---- | :---- |
| **SegNet Distortion** | Average fraction of pixels where argmax(pred\_original)\!= argmax(pred\_reconstructed). Evaluated on a 5-class U-Net using a tu-efficientnet\_b2 encoder. | 100x | Step-function topology. A 1% failure rate incurs a massive 1.0 point penalty. Gradients rely entirely on maintaining a positive margin between the target class logit and all other class logits. |
| **Compression Rate** | The byte size of the submitted archive.zip divided by the original file size (37,545,489 bytes). | 25x | Constant scalar penalty. Reducing file size by 3% yields only a 0.75 point improvement, making it a secondary priority compared to segmentation accuracy. |
| **PoseNet Distortion** | Mean Squared Error (MSE) of the first 6 dimensions of a 12-dimension pose vector. Evaluated on a fastvit\_t12 backbone across two consecutive frames. | sqrt(10 \* MSE) | Smooth L1 topology with diminishing returns. Halving the raw MSE from 0.38 to 0.19 yields less than a 0.6 point improvement due to the square root function suppressing the derivative at higher values. |

Understanding the mathematical interaction between this objective function and the underlying PyTorch automatic differentiation engine is critical. Every fractional millisecond saved in the iteration loop compounds across 600 frame pairs, translating directly into deeper convergence and lower distortion.

## **Hardware Architecture: Navigating the NVIDIA T4 Constraints**

To extract maximum computational throughput, the software pipeline must be explicitly tailored to the physical characteristics of the underlying hardware. The NVIDIA T4 is powered by the Turing architecture (specifically the TU104 die), featuring 2,560 CUDA cores, 320 Turing-generation Tensor Cores, and 16GB of GDDR6 memory.2

The transition from a local RTX 3090 workstation to the cloud-based T4 CI environment exposes a severe bottleneck transitioning from a compute-bound workload to a memory-bandwidth-bound workload. The RTX 3090 possesses 936 GB/s of memory bandwidth, whereas the T4 operates at a comparatively constrained 320 GB/s.1 In backpropagation scenarios involving high-resolution floating-point images (384x512 pixels) and memory-intensive element-wise operations during optimizer steps, the PyTorch execution engine spends the majority of its time waiting for memory to be read from and written to the global GDDR6 pool, leaving the actual CUDA compute cores drastically underutilized.

### **Tensor Core Activation and Memory Alignment**

To maximize the utility of the T4's 320 Tensor Cores, the dimensional properties of the tensors flowing through the neural networks must perfectly align with the hardware's expected memory transactions. The PyTorch pipeline currently utilizes Automatic Mixed Precision (AMP) with a GradScaler, which is the correct foundational step.1 Mixed precision leverages the Tensor Cores to perform matrix-multiply-accumulate (MMA) operations in FP16 while maintaining an FP32 master copy of the weights for numerical stability.4

However, Tensor Cores are only fully activated when specific conditions are met. Matrix dimensions—including batch size, input channels, output channels, and intermediate hidden dimensions—must be divisible by 8 for FP16 data.5 The current architectural implementation utilizes a batch size of 16, which inherently satisfies this divisibility requirement for the leading dimension.1 The channels\_last memory format is also already applied to both the EfficientNet-B2 and FastViT-T12 models.1 The channels\_last format physically reorganizes the memory layout of the tensors from NCHW to NHWC, allowing the Turing Tensor Cores to read continuous blocks of memory during grouped convolution operations, preventing cache fragmentation.6

While the dense neural network backbones are operating at near-optimal theoretical limits for the T4, the surrounding preprocessing, loss calculations, and optimizer steps are consuming 60% of the total iteration time.1 These operations are highly un-optimized, memory-bound tasks that circumvent the Tensor Cores entirely, relying instead on the slower, standard CUDA cores.

## **Differentiable Preprocessing Re-Architecture**

The profiling breakdown reveals that PoseNet preprocessing, consisting primarily of an RGB to YUV420 color space conversion, combined with the forward and backward passes of the PoseNet architecture, consumes approximately 40% of the computational budget per iteration.1 This is exceptionally high for a lightweight fastvit\_t12 model. The inefficiency is deeply rooted in the current mathematical implementation of the differentiable color space conversion.

### **Vectorized BT.601 Convolution Mapping**

The BT.601 standard dictates the precise linear transformation required to convert RGB pixel values into the YUV format.7 The current implementation (rgb\_to\_yuv6\_diff) executes this transformation utilizing raw PyTorch tensor arithmetic, performing multiple isolated multiplications, additions, and clamping operations on the extracted Red, Green, and Blue channels.1

Because PyTorch eagerly executes each of these operations sequentially, it launches a new CUDA kernel for every individual mathematical step. The GPU must read the RGB tensor from global memory, apply the 0.299 multiplication for the Red channel, write it back to memory, read it again for the Green channel addition, and so forth.7 This memory thrashing destroys pipeline throughput.

The linear nature of the RGB to YUV transformation allows it to be perfectly mapped onto a standard 2D convolution operation.9 Deep learning frameworks process convolutions with extreme efficiency, utilizing highly optimized, vertically fused cuDNN kernels.6 The mathematical formulation for the Luma (Y) and Chroma (U, V) components can be implemented as a 1x1 convolution with a fixed, frozen weight matrix and bias vector.

| YUV Component | RGB Multiplication Coefficients (Kernel Weights) | Bias Term |
| :---- | :---- | :---- |
| **Y (Luma)** | R: 0.299, G: 0.587, B: 0.114 | 0.0 |
| **U (Chroma)** | R: \-0.147, G: \-0.289, B: 0.436 | 128.0 |
| **V (Chroma)** | R: 0.615, G: \-0.515, B: \-0.100 | 128.0 |

By instantiating a torch.nn.Conv2d layer with in\_channels=3, out\_channels=3, kernel\_size=1, and setting requires\_grad=False on its parameters, the entire color transformation is executed as a single, fused memory transaction.9 This bypasses the launch overhead of dozens of element-wise kernels and keeps the gradients fully intact for the backward pass.

### **Spatial Downsampling via Average Pooling**

The 4:2:0 YUV specification requires chroma subsampling, a process where the U and V planes are spatially reduced to half the width and half the height of the Y plane.12 The existing implementation achieves this by slicing the tensors with a stride of two and manually adding the overlapping 2x2 blocks together (e.g., channel\[..., 0::2, 0::2\] \+ channel\[..., 1::2, 0::2\]...).1

Strided slicing in PyTorch creates non-contiguous memory views. When the autograd engine attempts to backpropagate through these non-contiguous views, it is forced to invoke slow fallback kernels to re-align the memory space, causing massive latency spikes during the loss.backward() phase.1

A mathematically equivalent and hardware-optimized approach is to execute the chroma subsampling using torch.nn.functional.avg\_pool2d.14 By applying F.avg\_pool2d(chroma\_tensor, kernel\_size=2, stride=2), the U and V channels are perfectly downsampled using highly optimized C++ backend algorithms.14 The gradient of an average pooling operation is natively supported by PyTorch and executes with virtually zero overhead, completely eliminating the need for manual tensor recombination and memory realignment.

## **Eradicating Framework Overhead and Synchronization**

The timing breakdown explicitly states that the "Optimizer step \+ overhead" consumes 20% of the iteration time.1 For a straightforward parameter update using a standard optimizer on a small set of image frames, this percentage indicates a severe systemic inefficiency. The root cause is invariably tied to CPU-GPU synchronization stalls and un-fused element-wise operations.

### **The CPU-GPU Synchronization Trap**

Modern GPUs operate asynchronously. When a PyTorch script is executed, the Python interpreter running on the CPU dispatches operations into a CUDA queue.16 The CPU immediately moves to the next line of code while the GPU processes the queue in the background. This decoupling allows the CPU to prepare the next batch of data while the GPU is busy computing.17

However, adversarial optimization loops inherently require the CPU to monitor the progress of the generated frames. If the script relies on methods like loss.item(), tensor.cpu(), or tensor.numpy() to extract scalar values for logging or early stopping mechanisms, it creates a hard synchronization barrier.6 When .item() is invoked to evaluate a condition (such as checking if the segmentation threshold has been crossed), the CPU halts all operation dispatching. It sits entirely idle, waiting for the GPU to finish every pending calculation in the queue just to return a single float value across the PCIe bus.16 This completely breaks the asynchronous pipeline, causing the T4's CUDA cores to starve for instructions.

| Synchronization Trigger | Mechanism of Latency | Asynchronous Alternative |
| :---- | :---- | :---- |
| loss.item() | Halts the Python interpreter until the GPU calculates the loss, forces a PCIe transfer. | Store losses in a GPU list using loss.detach().clone(). Transfer concurrently using non\_blocking=True at the end of the batch. |
| if seg\_l \< THRESH: | CPU evaluates a condition based on a GPU tensor state, causing a pipeline stall. | Move branching logic to the GPU using torch.where, or evaluate the condition only once every 10 iterations. |
| tensor.clamp\_(0, 255\) | In-place modifications can disrupt the autograd graph and force memory reallocation checks. | Use out-of-place clamping or integrate the clamp directly into a fused optimizer step. |

To salvage the 20% overhead budget, the optimization loop must be strictly decoupled. Early stopping mechanisms should not evaluate the loss condition on every single iteration. By configuring the loop to check the convergence threshold only every 10 iterations (e.g., if iter % 10 \== 0), the synchronization penalty is immediately reduced by an order of magnitude.18

For telemetry and logging, losses should be appended to a buffer using batch\_losses.append(loss.detach()). At the conclusion of the loop, the entire buffer can be transferred to the CPU memory space concurrently without interrupting the active backward pass.19

## **Advanced Graph Compilation Strategies**

The user has correctly identified that invoking torch.compile directly on the optimization pipeline "destroys the net score" because the minutes required for compilation eat directly into the 30-minute CI iteration budget.1

PyTorch's compilation engine, TorchInductor, relies on a Just-In-Time (JIT) methodology. Upon the first invocation of a compiled function, the AOTAutograd library traces the Python code, unrolls the forward and backward passes, and passes the resulting graph to the Triton compiler.20 The Triton compiler then performs extensive auto-tuning, testing various hardware-specific kernel configurations to find the absolute fastest execution path for the T4 GPU.20 While this results in exceptional runtime speed, the profiling phase can easily exceed 15 minutes for complex vision models like FastViT and EfficientNet.23

However, abandoning compilation entirely leaves massive performance gains on the table. The solution lies in applying strict, localized compilation directives.

### **Regional Compilation**

Rather than decorating the entire optimize\_batch function or the massive DistortionNet wrapper with @torch.compile, compilation must be restricted strictly to the static, compute-heavy neural network backbones.24 "Regional compilation" allows developers to selectively compile the repeated blocks—such as the inner layers of the FastViT-T12 model—while leaving the dynamic Python optimization loop, the if/else early stopping branches, and the data loading logic in standard eager mode.21

By surgically applying the compiler only to the nn.Module instances of SegNet and PoseNet, the size of the captured FX Graph is drastically reduced.24 This limits the surface area the compiler must analyze, bringing the cold start time down from minutes to mere seconds.24 Furthermore, PyTorch 2.5 introduces the configuration flag torch.\_dynamo.config.inline\_inbuilt\_nn\_modules=True, which aggressively prevents the compiler from triggering cache recompilations when minor dynamic inputs change, ensuring the compilation cost is paid exactly once at the start of the CI run.24

### **The reduce-overhead Directive**

When torch.compile is invoked without parameters, it defaults to a balanced mode. The user can explicitly dictate the compiler's behavior to circumvent the lengthy auto-tuning phase. On an NVIDIA T4, the primary bottleneck in a batch size 16 workload is often Python dispatch overhead and memory bandwidth, not raw dense math compute.22

By invoking the compiler with @torch.compile(mode="reduce-overhead"), the engine explicitly shifts its priority toward utilizing CUDA Graphs.22 This mode bypasses the exhaustive max-autotune profiling phase, keeping the cold start overhead minimal.27 Instead, it focuses on eliminating the Python interpreter from the execution path by baking the sequence of GPU operations into a static, replayable graph. This provides the execution speed of a compiled model without the catastrophic CI time penalties.

## **Manual CUDA Graph Capture for Static Workloads**

If the constraints of the CI environment render even regional torch.compile too slow, manual CUDA Graph capture remains the ultimate, lowest-level optimization for static workloads. CUDA Graphs fundamentally eliminate Python launch overhead by capturing a sequence of GPU kernels and their memory dependencies during a "warmup" phase, encapsulating them into a single, executable block.29

In the context of the comma.ai challenge, the computational topology is entirely static. The weights of the models are permanently frozen, the input frames are of a fixed 384x512 resolution, and the objective is to optimize the pixel values via gradient descent.1 Because no dimensions change, the workload is a perfect candidate for manual graph recording.

Implementing manual CUDA graphs requires strict adherence to persistent memory addressing.30 The memory locations of the input tensors (frame\_0 and frame\_1), the target labels, and the gradient buffers must remain constant throughout the entire optimization loop.31

1. **Static Memory Allocation:** Pre-allocate static tensors for the inputs and targets on the device. Instead of creating new tensors every iteration, update the contents of the static tensors using the .copy\_() method.32  
2. **Graph Recording:** Execute a single, standard forward and backward pass to warm up the caching allocator. Then, instantiate a torch.cuda.CUDAGraph() object and execute the complete optimization sequence—including the forward passes, loss calculations, the backward() call, and the optimizer.step()—within the recording context.32  
3. **Graph Replay:** During the actual batch processing loop, replace the entire optimization block with a single call to graph.replay().

When graph.replay() is invoked, the T4 GPU executes the entire backpropagation pipeline directly from its hardware command queue, with absolute zero intervention from the CPU.33 The PyTorch framework overhead drops to exactly zero milliseconds, guaranteeing that the pipeline runs as fast as the physical silicon allows.

## **Autograd Memory Mechanics and Batch Accumulation**

The user notes that attempting to increase the batch size to 32 results in an immediate Out-Of-Memory (OOM) error on the 16GB T4.1 Scaling the batch size is highly desirable, as processing more frames simultaneously smooths the gradient trajectory, potentially allowing the optimizer to converge to the target score in fewer total steps.34 Resolving the OOM constraint requires understanding the memory mechanics of the PyTorch automatic differentiation engine.

### **The Fallacy of Frozen Weights**

A pervasive misconception in PyTorch optimization is that setting requires\_grad=False on model parameters eliminates the memory overhead associated with the forward pass.36 This assumption is correct only during standard inference. However, adversarial generation and network inversion present a unique case: the input pixels themselves require gradients (input.requires\_grad \= True).36

To apply the chain rule of calculus and route the derivative of the loss function back to the input pixels, the autograd engine must traverse backward through the entire computational graph.38 Consequently, PyTorch is forced to store the intermediate feature maps and activation tensors of *every single layer* between the loss function and the input space, entirely regardless of whether the layer's weights are frozen.36

When pushing a batch size of 32 through the deep architectures of both an EfficientNet-B2 and a FastViT-T12 model, the sheer volume of high-resolution activation maps generated prior to spatial pooling operations quickly exhausts the 16GB capacity of the T4's GDDR6 memory.36

### **Virtual Scaling via Gradient Accumulation**

Because expanding the physical batch size natively is biologically constrained by the hardware's VRAM, the functionally equivalent mathematical approach is Gradient Accumulation.42 If a batch size of 32 provides the necessary gradient stability for faster convergence, it can be simulated perfectly by executing two sequential forward and backward passes at a batch size of 16 before invoking the optimizer step.42

By default, PyTorch accumulates gradients in the .grad attribute of a tensor rather than overwriting them.44

Python

\# Simulated Batch Size 32 via Accumulation  
optimizer.zero\_grad(set\_to\_none=True)

\# First micro-batch  
loss\_1 \= evaluate\_models(frame\_batch\_part\_1)  
loss\_1.backward()

\# Second micro-batch  
loss\_2 \= evaluate\_models(frame\_batch\_part\_2)  
loss\_2.backward()

\# Execute optimizer step based on accumulated gradients  
optimizer.step()

While gradient accumulation does not directly increase the iterations processed per second (as it still requires the same total number of forward and backward passes), it drastically improves the quality of the optimizer's trajectory.34 This stabilization often enables the use of higher learning rates or allows the optimizer to achieve the target distortion score in significantly fewer total steps, successfully banking time within the 30-minute CI budget.

Furthermore, memory management during the accumulation phase is critical. The script must employ optimizer.zero\_grad(set\_to\_none=True).46 Standard zeroing merely fills the gradient tensors with zeros, leaving the massive memory structures intact. Setting the gradients to None actively destroys the tensors, returning the memory to the caching allocator and significantly reducing the peak memory footprint during the subsequent forward pass.46

## **Transcending AdamW: Algorithmic Efficiency**

The choice of optimization algorithm fundamentally dictates how efficiently the gradients navigate the high-dimensional loss landscape. The current implementation relies on standard AdamW.1 While AdamW is the undisputed industry standard for training the internal weight matrices of neural networks, optimizing pixel values for adversarial image reconstruction presents a profoundly different mathematical topography.48

### **Fused AdamW for Bandwidth Compression**

If the architectural decision is made to retain the AdamW optimizer, its standard implementation must be modified. The default PyTorch optimizer executes parameter updates by iterating over tensors in a Python for loop, calculating the momentum, variance, and weight decay step-by-step.50 This approach launches a separate, small CUDA kernel for every individual parameter operation. In a memory-bandwidth-constrained environment like the T4, invoking dozens of tiny memory-bound kernels creates a massive bottleneck.50

PyTorch offers a highly optimized alternative via the fused=True argument during initialization (torch.optim.AdamW(..., fused=True)).46 Fused optimizers combine the element-wise gradient calculations, the tracking of the running averages (momentum and RMSprop), and the final parameter updates into a single, horizontally integrated CUDA kernel.46 The GPU reads the pixel data, the gradients, and the momentum states from memory exactly once, performs all math in the high-speed registers, and writes the results back to global memory.50

When optimizing large, dense tensors like raw image pixels, enabling fused AdamW can yield up to a 30% reduction in optimizer overhead, directly shrinking the 20% pipeline bottleneck currently observed.46

### **Second-Order Methods: The L-BFGS Advantage**

The problem of inverting a neural network to deduce an input image that minimizes a specific loss function—commonly observed in Neural Style Transfer, Deep Dream, and adversarial generation algorithms—is a deterministic optimization problem over a continuous, fixed landscape.48 Stochastic gradient descent methods, including AdamW, are explicitly designed to navigate noisy, shifting landscapes caused by stochastic mini-batch sampling.55 When the objective is deterministic (i.e., the batch of frames and the model weights remain absolutely constant during the iteration loop), first-order stochastic methods are mathematically sub-optimal.54

For deterministic image reconstruction tasks, second-order optimization algorithms, particularly the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method, comprehensively outperform AdamW in terms of convergence quality per iteration.56

While AdamW estimates the trajectory based solely on the immediate gradient (first-order derivative) and its historical moving average, L-BFGS approximates the inverse Hessian matrix of the entire loss landscape.58 This provides the algorithm with critical curvature information, allowing it to calculate not only the direction of steepest descent but also the mathematically optimal step size via built-in line searches.59

| Optimizer Characteristic | AdamW (First-Order) | L-BFGS (Second-Order Quasi-Newton) |
| :---- | :---- | :---- |
| **Landscape Navigation** | Uses momentum and variance to adapt learning rates per parameter. Ideal for noisy, stochastic mini-batches. | Approximates the inverse Hessian to understand landscape curvature. Ideal for smooth, deterministic loss surfaces. |
| **Memory Complexity** | Low. Requires storing two historical states (momentum, variance) per parameter. | High. Memory scales as ![][image1]. Requires storing multiple massive gradient vectors. |
| **Implementation Constraint** | Standard loss.backward() followed by optimizer.step(). | Requires wrapping the forward and backward passes in a closure function to allow multiple evaluations for line searches. |
| **Convergence Efficiency** | Requires hundreds of iterations to incrementally step down the gradient slope. | Achieves massive jumps toward the global minimum per iteration, requiring drastically fewer total steps. |

In practical adversarial image optimization, L-BFGS can achieve in 15 to 20 iterations a level of distortion reduction that AdamW might require 200 iterations to reach.60 By migrating to L-BFGS, the pipeline philosophy shifts from attempting to execute *more* iterations per second to requiring drastically *fewer* iterations to achieve the baseline 0.39 score, essentially banking massive amounts of time within the 30-minute CI budget.

However, implementing L-BFGS on a 16GB T4 requires strict memory management. L-BFGS relies on storing a history of previous gradients to construct its Hessian approximation.54 The default history\_size parameter in PyTorch is often set to 100, which will trigger an immediate OOM error when optimizing high-resolution floating-point image tensors.59 The history\_size must be aggressively tuned down (typically between 5 and 15\) to balance the VRAM constraints with the quality of the curvature approximation. Additionally, the forward and backward passes must be encapsulated within a closure function, as L-BFGS will invoke the pipeline multiple times per single step to perform its line search validation.50

### **Parameter Exclusivity and Optimizer Initialization**

A frequent source of memory bloat and computational overhead occurs during the initialization of the optimizer. It is a common anti-pattern to pass the entire model.parameters() generator to the optimizer and rely on the requires\_grad=False flag to prevent updates to the frozen weights.47

While the PyTorch autograd engine will correctly ignore the frozen weights, passing them into the optimizer's initialization forces the framework to actively track their states, allocating unnecessary parameter groups and bloating the optimizer's internal dictionaries.64 The optimizer should strictly be initialized using a filtered generator: filter(lambda p: p.requires\_grad, parameters).

In the context of this adversarial pipeline, the optimizer should exclusively be fed the specific frame\_0 and frame\_1 pixel tensors. The SegNet and PoseNet parameter trees must be completely detached from the optimizer's memory scope, ensuring that the optimizer's step function focuses solely on the modifiable input space.64

## **Strategic Exploitation of the Scoring Asymmetry**

Hardware and software optimizations must operate in lockstep with a strategic understanding of the competition's evaluation metric. The objective function is wildly asymmetric, prioritizing segmentation accuracy over temporal dynamics:

score \= 100 \* segnet\_distortion \+ 25 \* rate \+ sqrt(10 \* posenet\_distortion).1

The SegNet distortion component carries a 100x multiplier and calculates mathematical error based strictly on argmax class disagreements.1 This creates a non-smooth, step-function loss topology. Minor fluctuations in the logits that cause the dominant class ranking to flip induce massive penalties (a 1.0 point penalty for a 1% pixel failure rate). Conversely, massive changes to the underlying logit values that do *not* alter the top-1 ranking incur absolutely zero penalty.1

The current pipeline utilizes a margin loss to ensure the target class logit exceeds competing classes by a defined numerical margin.1 To optimize iteration efficiency, this margin should decay dynamically over time. Forcing the optimizer to constantly push logits further apart once the correct class is safely established wastes valuable iterations that could be spent minimizing the PoseNet smooth L1 loss.

Furthermore, the PoseNet MSE is heavily suppressed by the square root function: sqrt(10 \* posenet\_distortion).1 Because the derivative of ![][image2] decreases exponentially as ![][image3] increases, large numerical errors in the PoseNet output are penalized far less severely than equivalent errors in the linear SegNet term.

The optimization loop should practically ignore the PoseNet architecture until the SegNet outputs are highly stable. While the current implementation skips PoseNet for the first 30% of iterations, this static schedule is suboptimal.1 The PoseNet forward and backward passes—which consume 40% of the total iteration time—should only trigger dynamically when a specific batch's SegNet margin loss falls below a strict numerical safety threshold. By dynamically routing the computational budget away from the heavily suppressed PoseNet loss and focusing entirely on securing the catastrophic SegNet penalties, the pipeline can achieve lower objective scores within the constrained 85-iteration budget of the T4 GPU.

#### **Works cited**

1. context.md  
2. NVIDIA TURING GPU ARCHITECTURE, accessed April 6, 2026, [https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)  
3. NVIDIA T4 Tensor Core GPU for AI Inference | NVIDIA Data Center, accessed April 6, 2026, [https://www.nvidia.com/en-us/data-center/tesla-t4/](https://www.nvidia.com/en-us/data-center/tesla-t4/)  
4. Train With Mixed Precision \- NVIDIA Docs, accessed April 6, 2026, [https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)  
5. Tips for Optimizing GPU Performance Using Tensor Cores | NVIDIA Technical Blog, accessed April 6, 2026, [https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/)  
6. Performance Tuning Guide — PyTorch Tutorials 2.11.0+cu130 documentation, accessed April 6, 2026, [https://docs.pytorch.org/tutorials/recipes/recipes/tuning\_guide.html](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)  
7. Recommended 8-Bit YUV Formats for Video Rendering \- Win32 apps | Microsoft Learn, accessed April 6, 2026, [https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-8-bit-yuv-formats-for-video-rendering](https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-8-bit-yuv-formats-for-video-rendering)  
8. YUV to RGB Conversion \- FOURCC.org, accessed April 6, 2026, [https://fourcc.org/fccyvrgb.php](https://fourcc.org/fccyvrgb.php)  
9. Conv2d — PyTorch 2.11 documentation, accessed April 6, 2026, [https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)  
10. Demystifying the Convolutions in PyTorch \[0.4in\]Avinash Kak Purdue University \[0.1in\], accessed April 6, 2026, [https://engineering.purdue.edu/DeepLearn/pdf-kak/DemystifyConvo.pdf](https://engineering.purdue.edu/DeepLearn/pdf-kak/DemystifyConvo.pdf)  
11. RGB to Lab conversion \- vision \- PyTorch Forums, accessed April 6, 2026, [https://discuss.pytorch.org/t/rgb-to-lab-conversion/8019](https://discuss.pytorch.org/t/rgb-to-lab-conversion/8019)  
12. conversion from rgb to yuv 4:2:0 \- Stack Overflow, accessed April 6, 2026, [https://stackoverflow.com/questions/3203245/conversion-from-rgb-to-yuv-420](https://stackoverflow.com/questions/3203245/conversion-from-rgb-to-yuv-420)  
13. Neural Network Inference Optimization/Acceleration | by Subrata Goswami \- Medium, accessed April 6, 2026, [https://whatdhack.medium.com/neural-network-inference-optimization-8651b95e44ee](https://whatdhack.medium.com/neural-network-inference-optimization-8651b95e44ee)  
14. How to Apply a 2D Average Pooling in PyTorch? | by Hey Amit | Data Scientist's Diary, accessed April 6, 2026, [https://medium.com/data-scientists-diary/how-to-apply-a-2d-average-pooling-in-pytorch-8178ea451a43](https://medium.com/data-scientists-diary/how-to-apply-a-2d-average-pooling-in-pytorch-8178ea451a43)  
15. Custom pooling/conv layer \- PyTorch Forums, accessed April 6, 2026, [https://discuss.pytorch.org/t/custom-pooling-conv-layer/18916](https://discuss.pytorch.org/t/custom-pooling-conv-layer/18916)  
16. Writing Sync-Free Code — CUDA Graph Best Practice for PyTorch, accessed April 6, 2026, [https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/sync-free-code.html](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/sync-free-code.html)  
17. A guide on good usage of non\_blocking and pin\_memory() in PyTorch, accessed April 6, 2026, [https://docs.pytorch.org/tutorials/intermediate/pinmem\_nonblock.html](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)  
18. 7 Hidden PyTorch Memory Optimization Patterns for Production Deployment That Most Developers Miss | by Nithin Bharadwaj | TechKoala Insights \- Medium, accessed April 6, 2026, [https://medium.com/techkoala-insights/7-hidden-pytorch-memory-optimization-patterns-for-production-deployment-that-most-developers-miss-a652aa28fffd](https://medium.com/techkoala-insights/7-hidden-pytorch-memory-optimization-patterns-for-production-deployment-that-most-developers-miss-a652aa28fffd)  
19. Monitoring loss without synchronisation points \- PyTorch Forums, accessed April 6, 2026, [https://discuss.pytorch.org/t/monitoring-loss-without-synchronisation-points/152702](https://discuss.pytorch.org/t/monitoring-loss-without-synchronisation-points/152702)  
20. Maximizing AI/ML Model Performance with PyTorch Compilation \- Towards Data Science, accessed April 6, 2026, [https://towardsdatascience.com/maximizing-ai-ml-model-performance-with-pytorch-compilation/](https://towardsdatascience.com/maximizing-ai-ml-model-performance-with-pytorch-compilation/)  
21. Maximizing AI/ML Model Performance with PyTorch Compilation | by Chaim Rand \- Medium, accessed April 6, 2026, [https://chaimrand.medium.com/maximizing-ai-ml-model-performance-with-pytorch-compilation-7cdf840202e6](https://chaimrand.medium.com/maximizing-ai-ml-model-performance-with-pytorch-compilation-7cdf840202e6)  
22. torch.compile — PyTorch 2.11 documentation, accessed April 6, 2026, [https://docs.pytorch.org/docs/stable/generated/torch.compile.html](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)  
23. Really slow compilation times for torch.compile causing distributed training errors · Issue \#108971 · pytorch/pytorch \- GitHub, accessed April 6, 2026, [https://github.com/pytorch/pytorch/issues/108971](https://github.com/pytorch/pytorch/issues/108971)  
24. Reducing torch.compile cold start compilation time with regional compilation, accessed April 6, 2026, [https://docs.pytorch.org/tutorials/recipes/regional\_compilation.html](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)  
25. torch.compile and Diffusers: A Hands-On Guide to Peak Performance \- PyTorch, accessed April 6, 2026, [https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)  
26. Reducing AoT cold start compilation time with regional compilation \- PyTorch documentation, accessed April 6, 2026, [https://docs.pytorch.org/tutorials/recipes/regional\_aot.html](https://docs.pytorch.org/tutorials/recipes/regional_aot.html)  
27. 8 TorchCompile Pitfalls (and How to Dodge Them) | by Modexa \- Medium, accessed April 6, 2026, [https://medium.com/@Modexa/8-torchcompile-pitfalls-and-how-to-dodge-them-3364cd7352ce](https://medium.com/@Modexa/8-torchcompile-pitfalls-and-how-to-dodge-them-3364cd7352ce)  
28. How should I use torch.compile properly? \- Stack Overflow, accessed April 6, 2026, [https://stackoverflow.com/questions/75886125/how-should-i-use-torch-compile-properly](https://stackoverflow.com/questions/75886125/how-should-i-use-torch-compile-properly)  
29. CUDAGraph Trees — PyTorch 2.11 documentation, accessed April 6, 2026, [https://docs.pytorch.org/docs/stable/user\_guide/torch\_compiler/torch.compiler\_cudagraph\_trees.html](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)  
30. How CUDA Graph Works in torch.compile \- GPU Notes, accessed April 6, 2026, [https://fkong.tech/posts/2025-12-23-cuda-graph-in-torch-compile/](https://fkong.tech/posts/2025-12-23-cuda-graph-in-torch-compile/)  
31. Handling Dynamic Patterns — CUDA Graph Best Practice for PyTorch, accessed April 6, 2026, [https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/handling-dynamic-patterns.html](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/handling-dynamic-patterns.html)  
32. PyTorch CUDA Graph Capture \- Lei Mao's Log Book, accessed April 6, 2026, [https://leimao.github.io/blog/PyTorch-CUDA-Graph-Capture/](https://leimao.github.io/blog/PyTorch-CUDA-Graph-Capture/)  
33. Accelerating PyTorch with CUDA Graphs, accessed April 6, 2026, [https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)  
34. Breaking Down Backpropagation in PyTorch: A Complete Guide \- Medium, accessed April 6, 2026, [https://medium.com/@noel.benji/breaking-down-backpropagation-in-pytorch-3762ea107d3a](https://medium.com/@noel.benji/breaking-down-backpropagation-in-pytorch-3762ea107d3a)  
35. EfficientNet from Scratch \- Kaggle, accessed April 6, 2026, [https://www.kaggle.com/code/vikramsandu/efficientnet-from-scratch](https://www.kaggle.com/code/vikramsandu/efficientnet-from-scratch)  
36. Lowering PyTorch's Memory Consumption for Selective Differentiation \- arXiv, accessed April 6, 2026, [https://arxiv.org/html/2404.12406v1](https://arxiv.org/html/2404.12406v1)  
37. Requires\_grad= False does not save memory \- PyTorch Forums, accessed April 6, 2026, [https://discuss.pytorch.org/t/requires-grad-false-does-not-save-memory/21936](https://discuss.pytorch.org/t/requires-grad-false-does-not-save-memory/21936)  
38. Optimize PyTorch training with the autograd engine \- Red Hat Developer, accessed April 6, 2026, [https://developers.redhat.com/articles/2026/03/03/optimize-pytorch-training-autograd-engine](https://developers.redhat.com/articles/2026/03/03/optimize-pytorch-training-autograd-engine)  
39. Grad through frozen weights \- PyTorch Forums, accessed April 6, 2026, [https://discuss.pytorch.org/t/grad-through-frozen-weights/84672](https://discuss.pytorch.org/t/grad-through-frozen-weights/84672)  
40. GPU Memory Usage During the Pre-training \- PyTorch Forums, accessed April 6, 2026, [https://discuss.pytorch.org/t/gpu-memory-usage-during-the-pre-training/21586](https://discuss.pytorch.org/t/gpu-memory-usage-during-the-pre-training/21586)  
41. PyTorch Activation Checkpointing: Complete Guide | by Hey Amit \- Medium, accessed April 6, 2026, [https://medium.com/@heyamit10/pytorch-activation-checkpointing-complete-guide-58d4f3b15a3d](https://medium.com/@heyamit10/pytorch-activation-checkpointing-complete-guide-58d4f3b15a3d)  
42. Training Neural Nets on Larger Batches: Practical Tips for 1-GPU, Multi-GPU & Distributed setups | by Thomas Wolf | HuggingFace | Medium, accessed April 6, 2026, [https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)  
43. How can you train your model on large batches when your GPU can't hold more than a few samples? \- PyTorch Forums, accessed April 6, 2026, [https://discuss.pytorch.org/t/how-can-you-train-your-model-on-large-batches-when-your-gpu-can-t-hold-more-than-a-few-samples/80581](https://discuss.pytorch.org/t/how-can-you-train-your-model-on-large-batches-when-your-gpu-can-t-hold-more-than-a-few-samples/80581)  
44. Accumulating gradients for a larger batch size with PyTorch \- Stack Overflow, accessed April 6, 2026, [https://stackoverflow.com/questions/70119232/accumulating-gradients-for-a-larger-batch-size-with-pytorch](https://stackoverflow.com/questions/70119232/accumulating-gradients-for-a-larger-batch-size-with-pytorch)  
45. How to increase batch size with limited GPU memory \- data \- PyTorch Forums, accessed April 6, 2026, [https://discuss.pytorch.org/t/how-to-increase-batch-size-with-limited-gpu-memory/179436](https://discuss.pytorch.org/t/how-to-increase-batch-size-with-limited-gpu-memory/179436)  
46. AdamW — PyTorch 2.11 documentation, accessed April 6, 2026, [https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)  
47. Best practice for freezing layers? \- autograd \- PyTorch Forums, accessed April 6, 2026, [https://discuss.pytorch.org/t/best-practice-for-freezing-layers/58156](https://discuss.pytorch.org/t/best-practice-for-freezing-layers/58156)  
48. Picking an optimizer for Style Transfer – Surprising results comparing BFGS with Adam \[R\], accessed April 6, 2026, [https://www.reddit.com/r/MachineLearning/comments/5yjfm5/picking\_an\_optimizer\_for\_style\_transfer/](https://www.reddit.com/r/MachineLearning/comments/5yjfm5/picking_an_optimizer_for_style_transfer/)  
49. Selecting the best optimizers for deep learning–based medical image segmentation \- PMC, accessed April 6, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10551178/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10551178/)  
50. torch.optim — PyTorch 2.11 documentation, accessed April 6, 2026, [https://docs.pytorch.org/docs/stable/optim.html](https://docs.pytorch.org/docs/stable/optim.html)  
51. Adam vs. AdamW: Understanding Weight Decay and Its Impact on Model Performance | by Ahmed Yassin, accessed April 6, 2026, [https://yassin01.medium.com/adam-vs-adamw-understanding-weight-decay-and-its-impact-on-model-performance-b7414f0af8a1](https://yassin01.medium.com/adam-vs-adamw-understanding-weight-decay-and-its-impact-on-model-performance-b7414f0af8a1)  
52. SGD — PyTorch 2.11 documentation, accessed April 6, 2026, [https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)  
53. Performance Comparison between Torch.Compile and APEX optimizers, accessed April 6, 2026, [https://dev-discuss.pytorch.org/t/performance-comparison-between-torch-compile-and-apex-optimizers/2023](https://dev-discuss.pytorch.org/t/performance-comparison-between-torch-compile-and-apex-optimizers/2023)  
54. A Comparison of Selected Optimization Methods for Neural Networks \- Diva-portal.org, accessed April 6, 2026, [https://www.diva-portal.org/smash/get/diva2:1438308/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:1438308/FULLTEXT01.pdf)  
55. Optimizers in Machine Learning and AI: A Comprehensive Overview | by Ansh Mittal, accessed April 6, 2026, [https://medium.com/@anshm18111996/comprehensive-overview-optimizers-in-machine-learning-and-ai-57a2b0fbcc79](https://medium.com/@anshm18111996/comprehensive-overview-optimizers-in-machine-learning-and-ai-57a2b0fbcc79)  
56. Multi-Stage Optimization for Photorealistic Neural Style Transfer \- CVF Open Access, accessed April 6, 2026, [https://openaccess.thecvf.com/content\_CVPRW\_2019/papers/NTIRE/Yang\_Multi-Stage\_Optimization\_for\_Photorealistic\_Neural\_Style\_Transfer\_CVPRW\_2019\_paper.pdf](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Yang_Multi-Stage_Optimization_for_Photorealistic_Neural_Style_Transfer_CVPRW_2019_paper.pdf)  
57. Linear regression with PyTorch: LBFGS vs Adam | by Soham | Medium, accessed April 6, 2026, [https://medium.com/@dragonbornmonk/linear-regression-with-pytorch-lbfgs-vs-adam-19b62ce83d4](https://medium.com/@dragonbornmonk/linear-regression-with-pytorch-lbfgs-vs-adam-19b62ce83d4)  
58. When training a neural network, why choose Adam over L-BGFS for the optimizer?, accessed April 6, 2026, [https://scicomp.stackexchange.com/questions/34172/when-training-a-neural-network-why-choose-adam-over-l-bgfs-for-the-optimizer](https://scicomp.stackexchange.com/questions/34172/when-training-a-neural-network-why-choose-adam-over-l-bgfs-for-the-optimizer)  
59. The reason of superiority of Limited-memory BFGS over ADAM solver \- Stats StackExchange, accessed April 6, 2026, [https://stats.stackexchange.com/questions/315626/the-reason-of-superiority-of-limited-memory-bfgs-over-adam-solver](https://stats.stackexchange.com/questions/315626/the-reason-of-superiority-of-limited-memory-bfgs-over-adam-solver)  
60. A comparison of the performance of the Adam optimizer, an algorithm for... | Download Scientific Diagram \- ResearchGate, accessed April 6, 2026, [https://www.researchgate.net/figure/A-comparison-of-the-performance-of-the-Adam-optimizer-an-algorithm-for-first-order\_fig3\_360640362](https://www.researchgate.net/figure/A-comparison-of-the-performance-of-the-Adam-optimizer-an-algorithm-for-first-order_fig3_360640362)  
61. The model produced by Adam is close to the true model, while L-BFGS-B's... \- ResearchGate, accessed April 6, 2026, [https://www.researchgate.net/figure/The-model-produced-by-Adam-is-close-to-the-true-model-while-L-BFGS-Bs-output-despite\_fig3\_322652684](https://www.researchgate.net/figure/The-model-produced-by-Adam-is-close-to-the-true-model-while-L-BFGS-Bs-output-despite_fig3_322652684)  
62. comparing two optimizer, adam, lbfgs · Issue \#328 · jcjohnson/neural-style \- GitHub, accessed April 6, 2026, [https://github.com/jcjohnson/neural-style/issues/328](https://github.com/jcjohnson/neural-style/issues/328)  
63. Optimizing Neural Networks with LFBGS in PyTorch \- | Johannes Haupt, accessed April 6, 2026, [https://johaupt.github.io/blog/pytorch\_lbfgs.html](https://johaupt.github.io/blog/pytorch_lbfgs.html)  
64. Guide to Freezing Layers in PyTorch: Best Practices and Practical Examples \- Medium, accessed April 6, 2026, [https://medium.com/we-talk-data/guide-to-freezing-layers-in-pytorch-best-practices-and-practical-examples-8e644e7a9598](https://medium.com/we-talk-data/guide-to-freezing-layers-in-pytorch-best-practices-and-practical-examples-8e644e7a9598)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKoAAAAYCAYAAABqdGb8AAAHwUlEQVR4Xu2ZeaxdUxTGPzGEmFUNNTSlpGjNRRURkYYEEa2QEBFiiPCPmiVyEYmKIqiKIa0/RFFDIvVUhNcQFRJTUDHEEEOQEoIEMaxf91nv7rPvOeeed6/3tHq+5Mu996xz1tnDt9dae1+pQYMGDRqsvtjEuGF6sUGDCvStmTHGGcZZxj2M6+bNHdjXuMC4eWpYA8Bg0d91UkODEcd44+LsszaYqCOMrxoHjKdlfNb4oXFq+9YcdjS+YNwrNRi2Nr5o/DvjD8bJuTukyzKb8wvjlNwdIwPe8a3COwcVBFsGxuYs4yKF/q5tIADdYbzGuH5i6xeHG59UzSDHy+cYP1GnILHdoyCy/RIbE3ibsZVcT3GC8UcFUbTyplVg8l82TkoNIwz69oS6CxXboEL7j8+bKrGF8Wz1md5WAxxg/MX4mXH7xNYv0NCdxstTQwoma77xe+NBic1B+l+p4DBOkUTH97PPKrSM5ylEyxXGbXNWaR/jvcb1kuujgQfUXahguvFS48apoQKM26Pq7nt1Bxo5x3iiRqZEmqagiwmpIcb5xr+yzzJsaXzN+K5COnewCpaoOmIwsfcpRE2ETlQ6NXdHKDEuTq6VAX+p0GMwkOPUva521BVqLzhdoSwaCd//J7i+0EEhJhq/UnGUi+GO4tCPOBHpVX5TCXZRiJbcz8r5zfiMcaPoHsqOw6LfVRhrfMR4YGpQEOmZCvVU3VrKhbqZcXeFDSSpLhY6KZyCn7S/Q3Sdd1DXzzYeq5AZ6C+ghPrcuFxhnBm3dEH784whvuNozb08c6TxYIUNH+/YSeE55gs73EahvfF1PuuOASjrC37xT9bkus9b2oaYcT+5/zjjtcZjst9FIJg9qJKI3VKIcDck11PQ4K+VFyqf/O5Ws9E5UiagkYgUsSJawCJ4WNULJQXRGT9xqdKLSAFCfcV4t0JUP8P4sUJd7n4uMH6pfI3K5D2vUH8zFkwCwsTupyDfZeQ7/rjuoCxg8V+o0J9TjO8Zj8rsLJZ3FN5J+/CxVMEfJzIPGX/P7AMKi4mFxj6Da68bd1U9VPUFv/crvCuef9r3k8Jmmb4RjHjmT4XFBejjm8YrFdpyl/EtFe/yyc742jQ1+AaBtH903tQB7NwXO6Kh36h7JGwp75+0z0B6vUtn5mv49Wks1l5FChAqfWNyHEQ4anLa5iBC/qq2UFmAy5QfWOpwt/v4wjT1IwxEeL3yEYRSgTH1ExT3wenEbsaLFN7pJw+IPF70gPSJMAojUwm69QUwJrFQsbFYfN5mKojU++Rtj+8hGJG952a/Y+Av9j8Ej4jphBThdnXu2BEqK4jPMnh9GqdLbywbqwkaXn2awsWK6HsRKUCoRbU3u9y4b77z9clj8TExTATfiTykPE97VUJFBEUBwt/hGa7KB/BM54seQcxT981tim59AYxJKlSfNyLkB8bn1D5m8uBGXx20kfQ+qM7+4I++eOk0BBdqoYojUBNxjkrKic9K6wg1rk9jtBSET0SYo+5RuQx0nMGibYcmtrpAqIPKD1wdobIoblb+DHiR2hNVJTIWb+of+DuYcBZ5lQ+AMBHXpwrBAIEi1OFmp259AalQXciUc0+p8/iS+/HDOTylQUxKwVQTjCulBLVxDkQQIkmVUBECaYQXzk5sdYRKzcPzKSYqbOJIc0uVj2Z1QdsQOpF0Z4Xz0LLjtSr0KlQHGeJkhWMoxolzZdqWimwr4/6rnggZ6g91LlB/x2IFsbkP2lgGShaiISXV1dnvXlHWF5AKFWDjDxv6MjO7RiTl7JhIio/SnXyC0tTvq7FowBxMPCsl3lg4iJaIjfqmDC11pjdAB/2oyidlOIhF6u3aTr2JtVeh8sl9DtpEfTao4CsVKs/TXuB1+knZbwfzwHzQN1BHqEQ9Nltsnh5Tb4u+W19AkVCL9MF9+Juc2dKNOvdNV+ecI2g2sSyWDhCqcbZAnUJk90kRf4uKjxQ8Isc1SIyxCtFy79SQYZrCRqDs+TIwiOzCWe1pm4crVnxRM6W7zbpC5e/mMUN3BIExlvhlIliE7Ow52UCEV2T3MfmkRI7ZvA8uDjZZbLb8vkG1S4Ey8F6E30qu10W3voBUqOwPVij/HFmD0wOCE88xRwSzSZkd8JcpWdb9OtisLVFnSTCEqcaPFF6IquGA8W0FsaYOHVynI6SxGAwyvuJ6h0jCeVwMxP+4yqN5GfY0XqdOkToYrJuyzypMUVjB3safFXbdy6NrHMmwUCHfuUaavVVhct9QiGQc35AuEZ/vyAFnvSx2xpP/s8dHNs5tGRd8EJGWKQibxQaYbP4t9LbwnXYUgehV5x/CMlT1hfmMx4Rxom1zo9+UgJAxihc4c0Q9ulLBJ8eQ+I9rX+CLOo7qhUBE7PxnKTR6nMoFGoMUhiiJGGsbNlCYCMaOyWSnXATuwV62sIggRClPsb0Agc5TZzqlLEs3MilvVLt93frSK/BNSi/zO0EhOpNlRwSE/JfUXwHfYPggIhGZzs1+l22iED+LoIoIM812ow0yGdG2bDH/K2BnT1gvqmMbjAzIfqRTjrmo/xaqv4j8X4Jg97Tq7yt6BiUCdQisUy406B+MM5tQstlCtevaNQ30o6Vw9Dkq2iFkX2I8JDU0aFCBGQrly6iItEGDBg3WbvwDpyXEY8TyMKoAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAAXCAYAAAD6FjQuAAABjUlEQVR4Xu2UPSiGURTHj3zka1CKhEUiixKLKAwGJWRSSlIGA/kIicEgw7uQhcFCUfKxSAwGk8FXMVikDMpgFGXif5z73N73dOl5cQflV7/e555ze8597z3PJfrHTTmcjsMfsQ/bYX1Ivw3/q00d9MU6LNZBH+TCZR30xSws00EfZMINHTQUwmbzy2TDJphnZ8TJOKxTsQQ4CBdhF7yGEbgKx+ANLLWzQ5JC0u6aRjhFUpRZgfckW70Hn2GlyYWmj2SbNAOwxDxnwEO4BZNgDWyFiSZvJ+3A/uig4lgHHBTBB5LtdlIAz+ELyX676IVDOuigBb7CWp0IaINVcB6+wYbY9AcnME0HSc6pBy6RdOoCvIP5Jp8D52C6GVv4RuBiugn4ZSMqFsAf+C28JOm4C3hEUpgXMgo7g8kaPlguWBEVOyPHygzJJN23Cw9IzvyU5D3bcMLMcVJNUmzNjLvhpM264U7j7Ur9ZPwlvDIuyF/+FcyKTf8uHSTFuENnVM4Lj/CJ5H7zzjDJvfY3eQcUf0TWfipzsAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAAAyklEQVR4Xu3RsQtBURTH8SMpokRSVslgYDAbKMrfIKtZFgMjo1JmyYBRFrNdKYPJ5A+wKJOB73n3vZIno+n96jPczjm3+84T8fLv+FFEDWH4kEUVobc+iWKJDno4YowhplgjqI16wwAla0wkhQtWyOOKHSJajKMv9iQp4IYGAmgiZ9dc0aa7mPf/jD5phj1iHzUr+nETtJDAScyADmp0G1qzUscTI5TxQNeu6UVzZOyzpHHAAhu0cRazsi0qTqMT3URSzI/5dvbiygvC9RzA6VnpHQAAAABJRU5ErkJggg==>