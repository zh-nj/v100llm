# **架构分流与算力革命：NVIDIA V100 Volta架构中CUDA Core与Tensor Core的深度详尽对比研究报告**

## **1\. 引言：计算范式的裂变与Volta的崛起**

2017年，随着人工智能（AI）与深度学习（Deep Learning）浪潮的爆发式增长，高性能计算（HPC）领域迎来了一个关键的转折点。传统的通用图形处理器（GPGPU）架构，尽管在并行计算方面取得了巨大成功，但在面对深度神经网络（DNN）训练中海量的密集矩阵运算时，开始显现出效率的瓶颈。摩尔定律的放缓迫使硬件架构师寻求新的增长点——即从通用的指令级并行（Instruction Level Parallelism, ILP）转向特定领域的架构创新（Domain Specific Architecture, DSA）。

在此背景下，NVIDIA推出了代号为Volta的GPU架构，其旗舰产品Tesla V100标志着计算范式的彻底裂变。V100基于台积电12nm FFN工艺制造，集成了惊人的211亿个晶体管，芯片面积达到了815平方毫米，是当时世界上最大的GPU芯片1。然而，Volta架构最引人注目的创新并非仅仅是晶体管数量的堆叠，而是其核心计算单元的异构化设计：它在保留并增强了传统的**CUDA Core**的同时，首次引入了专为矩阵数学设计的**Tensor Core**。

本报告旨在从微架构设计、指令集实现、数值行为、性能特征及能源效率等多个维度，对V100中的CUDA Core与Tensor Core进行详尽、深入的对比分析。我们将通过解构其底层的硅片逻辑，揭示这两种核心如何分别代表了“通用灵活性”与“专用极致效率”的两个极端，以及它们如何在Volta架构中协同工作，共同定义了现代AI计算的基础设施。

## ---

**2\. Volta GV100 微架构深度剖析**

要理解CUDA Core与Tensor Core的区别，首先必须深入到Volta架构的基本构建块——流式多处理器（Streaming Multiprocessor, SM）的内部结构中。GV100 GPU包含80个SM，而每个SM的内部布局经历了激进的重构，以适应异构计算的需求1。

### **2.1 SM的分区化设计（Sub-Core Partitioning）**

与前代Pascal架构不同，Volta SM采用了更加细粒度的分区设计。每个SM被物理划分为四个处理块（Processing Blocks），或称为子核心（Sub-Cores）。这种分区不仅仅是逻辑上的划分，更是硬件资源的物理隔离，旨在简化调度逻辑并提高指令流水线的利用率。

每个子核心（Sub-Core）拥有独立的资源切片：

* **指令调度器（Warp Scheduler）**：每个子核心配备一个经优化的调度器，能够在每个时钟周期发射一条指令。
* **调度单元（Dispatch Unit）**：负责将指令分发到相应的执行单元。
* **寄存器文件（Register File）**：64 KB的寄存器资源，供该子核心内的线程独占使用5。

在计算资源方面，这种分区设计决定了CUDA Core与Tensor Core的物理布局。单个子核心包含：

* **16个 FP32 CUDA Core**：用于单精度浮点计算。
* **16个 INT32 CUDA Core**：用于整数计算（地址寻址、循环控制等）。
* **8个 FP64 CUDA Core**：用于双精度科学计算。
* **2个 Tensor Core**：用于混合精度矩阵运算4。

通过简单的数学累加，我们可以得出单个完整SM（包含4个子核心）的资源总量：64个FP32单元、64个INT32单元、32个FP64单元以及8个Tensor Core。这种特定的比例（FP32:Tensor Core \= 8:1）揭示了NVIDIA的设计意图：即在保证通用计算能力的同时，通过少量的专用电路（Tensor Core）提供爆炸性的矩阵吞吐能力。

### **2.2 统一的L1数据缓存与共享内存**

CUDA Core与Tensor Core的高效运行极度依赖于数据供给。Volta架构的一个重大改进是L1数据缓存（L1 Data Cache）与共享内存（Shared Memory）的统一4。在GV100中，每个SM配备了128 KB的统一存储块。

* **对CUDA Core的意义**：这种统一设计降低了传统标量计算的内存延迟，使得非规则访存（如稀疏矩阵操作或图算法）能更有效地利用缓存。
* **对Tensor Core的意义**：Tensor Core的吞吐量极大（单周期512次操作），对带宽极其饥渴。统一架构允许将高达96 KB配置为流式共享内存，这使得大块的矩阵瓦片（Matrix Tiles）可以直接驻留在靠近执行单元的高速存储中，从而大幅减少了向HBM2显存的请求频率7。这种架构调整是Tensor Core能够发挥其理论性能的关键支撑。

## ---

**3\. CUDA Core：通用并行计算的基石**

在NVIDIA的术语体系中，“CUDA Core”通常指的是执行单精度浮点（FP32）运算的流水线单元。然而，在Volta架构中，CUDA Core的概念经历了重要的演变，其功能被进一步细分和增强，以适应现代并行程序的复杂性。

### **3.1 FP32与INT32的数据通路分离（Datapath Separation）**

在Pascal及更早的架构中，CUDA Core是一个相对单一的通道，负责处理浮点和整数指令。这意味着，如果一个线程需要执行整数加法（例如计算数组的内存地址指针）和浮点乘法（例如处理数据），这两条指令必须串行执行，因为它们争夺同一个执行端口。

Volta架构引入了革命性的**独立INT32数据通路**4。现在，每个SM子核心不仅有FP32单元，还有数量对等的INT32单元。

* **并行执行机制**：指令调度器可以在同一个时钟周期内，并发地发射一条FP32指令和一条INT32指令。虽然这两条指令可能来自同一个Warp（利用指令级并行ILP），但在高占用率下，更常见的是来自不同的Warp。
* **性能影响**：这种设计极大地提高了流水线的吞吐效率。在深度学习内核中，大量的循环控制和指针算术运算（Integer）不再阻塞核心的数学运算（Floating Point）。据官方白皮书指出，这种并发执行能力使得Volta在通用计算任务上的效率相比Pascal提升了50%以上4。

### **3.2 延迟优化与指令吞吐**

除了并行化，Volta还对CUDA Core的指令延迟进行了优化。核心的FMA（Fused Multiply-Add，融合乘加）运算的指令延迟从Pascal时代的6个时钟周期降低到了4个时钟周期4。

* **延迟隐藏**：较低的指令延迟意味着每个SM只需要维持较少的活跃Warp数量即可掩盖流水线停顿（Pipeline Stall）。这降低了寄存器文件的压力，使得每个线程可以使用更多的寄存器，或者在相同的寄存器资源下运行更多的线程。
* **IEEE 754合规性**：V100的CUDA Core严格遵循IEEE 754-2008标准，支持包括非正规数（Denormal numbers）、NaN处理和四种舍入模式（默认RN，Round-to-Nearest）。这保证了科学计算和金融模拟等对数值稳定性要求极高的应用的正确性。

### **3.3 FP64单元：科学计算的定海神针**

尽管Tensor Core是Volta的明星，但V100仍然保留了强大的双精度计算能力。每个SM包含32个FP64单元，提供FP32单元一半的吞吐量（FP64:FP32 \= 1:2）1。

* **定位差异**：Tensor Core在V100上**不支持**FP64运算（FP64 Tensor Core直到Ampere A100才引入）11。因此，凡是涉及高精度物理模拟（如流体力学、量子化学、气象预测）的任务，完全依赖于FP64 CUDA Core。这是V100作为HPC领域顶级加速器的核心竞争力之一，区别于纯粹面向AI推理的低端卡。

## ---

**4\. Tensor Core：矩阵计算的专用引擎**

如果说CUDA Core是多才多艺的“瑞士军刀”，那么Tensor Core就是一把削铁如泥的“重剑”。它是NVIDIA针对深度学习中核心计算模式——矩阵乘法（GEMM）——进行的专用硬件加速。

### **4.1 4x4x4 矩阵乘累加 (HMMA) 的硬件实现**

Tensor Core的基本操作单元不再是标量（Scalar），而是矩阵（Matrix）。V100的Tensor Core执行一种独特的指令：混合精度矩阵乘累加（Mixed Precision Matrix Multiply Accumulate），数学形式为 ![][image1]。

* **维度定义**：硬件层面上，Tensor Core一次处理的基本单元是 ![][image2] 的矩阵4。
* **算力倍增器**：
  * 一个标准的FP32 CUDA Core每个时钟周期执行1次FMA（包含1次乘法和1次加法，共2 FLOPs）。
  * 一个Tensor Core每个时钟周期执行 ![][image3] 的矩阵运算。这意味着它在单周期内完成了64次浮点FMA操作（![][image2] 输出矩阵的每个元素都需要4次乘加）。
  * **吞吐量对比**：单个Tensor Core的算力是单个CUDA Core的64倍（按FMA指令计数）。考虑到SM中CUDA Core与Tensor Core的数量比为8:1（64 vs 8），在矩阵运算任务中，Tensor Core为SM带来的综合算力提升是CUDA Core的8倍（![][image4] FMA per clock vs 64 FMA per clock）4。

### **4.2 混合精度的物理实现 (Mixed Precision)**

“混合精度”是Tensor Core设计的精髓，也是其在AI训练中取得巨大成功的关键。

* **输入精度 (FP16)**：矩阵A和矩阵B必须是半精度浮点数（FP16）。这减少了从寄存器到计算单元的数据传输量，降低了功耗和带宽需求。
* **累加精度 (FP32)**：矩阵C和结果矩阵D可以是FP16或FP32。在硬件实现中，乘法器输出全精度的乘积（Exact Product），然后这些乘积与32位的累加器（Accumulator）进行加法运算4。
* **数值稳定性**：使用FP32进行累加至关重要。如果在FP16下进行累加，在处理大型矩阵时极易发生下溢（Underflow）或大数吃小数的精度丢失问题。Tensor Core通过硬件强制的FP32累加，兼顾了FP16的高吞吐量和FP32的数值稳定性。

### **4.3 芯片面积效率与晶体管经济学**

从芯片设计的角度来看，Tensor Core展示了极高的面积效率（Area Efficiency）。

* **控制逻辑摊薄**：对于CUDA Core，每一条标量指令都需要经过取指（Fetch）、解码（Decode）、调度（Dispatch）等复杂的控制逻辑。而在Tensor Core中，一条指令驱动64个FMA操作。这意味着控制逻辑的晶体管开销被分摊到了64次运算上，极大地提高了晶体管用于实际计算（ALU）的比例12。
* **能效比**：这种设计直接转化为能效优势。数据在以脉动阵列（Systolic Array）形式排列的ALU之间流动，减少了对寄存器文件的频繁读写。研究表明，在执行线性代数任务时，Tensor Core的能效比（Performance per Watt）是传统CUDA Core的5倍以上14。

## ---

**5\. 指令集架构 (ISA) 与编程模型对比**

硬件的差异最终反映在软件接口和指令集架构（ISA）上。V100引入了全新的PTX指令和SASS微码来驾驭Tensor Core。

### **5.1 线程调度模型的变革：独立线程调度 (ITS)**

在讨论具体指令之前，必须提及Volta架构在线程调度上的重大变革——独立线程调度（Independent Thread Scheduling, ITS）4。

* **Pascal模式**：在Volta之前，GPU采用SIMT（单指令多线程）模式，一个Warp中的32个线程共享一个程序计数器（PC）。如果发生分支发散（Branch Divergence），硬件必须序列化执行各分支，直到汇合点。
* **Volta模式**：V100为Warp中的每个线程维护独立的PC和栈状态。这使得线程之间可以更自由地发散和同步。
* **对Tensor Core的影响**：Tensor Core操作本质上是Warp同步的（Warp-Synchronous）。因为矩阵的数据分布在Warp的32个线程的寄存器中，所有线程必须在同一时刻到达执行点才能发起Tensor Core指令。因此，Volta引入了显式的同步指令（如\_\_syncwarp()）和协同指令（mma.sync），强制要求Warp内的线程在执行矩阵运算前对齐状态15。

### **5.2 PTX层级：WMMA API 与 MMA 指令**

NVIDIA在PTX（Parallel Thread Execution）层面提供了两套接口：

1. **WMMA (Warp Matrix Multiply Accumulate)**：
   * 这是一套高层抽象，通过C++命名空间 nvcuda::wmma 提供。
   * 它引入了“片段”（Fragments）的概念，将矩阵A、B、C的数据抽象为分布在寄存器中的对象。程序员不需要关心具体的寄存器布局，编译器会自动处理数据的加载（Load）、存储（Store）和计算（Mma）8。
   * **限制**：WMMA对矩阵形状有严格限制（如16x16x16），且隐藏了底层细节。
2. **MMA (Matrix Multiply Accumulate)**：
   * 这是底层的汇编级PTX指令，例如 mma.sync.aligned.m8n8k4...15。
   * 它暴露了更细粒度的控制，直接操作寄存器对。由于Volta的Tensor Core原生支持4x4x4（或8x8x4）粒度，MMA指令允许专家级程序员进行极致优化，但需要手动处理复杂的寄存器映射和数据对齐。

### **5.3 SASS层级：HMMA.884 与微码分析**

当我们深入到二进制的SASS（Streaming Assembler）层面，即GPU硬件实际执行的机器码时，CUDA Core和Tensor Core的区别暴露无遗。

* **CUDA Core指令**：对应的是 FFMA（浮点融合乘加）、FADD 等标准指令。这些指令是标量的，每个线程独立执行。
* **Tensor Core指令**：在V100上，对应的核心指令是 HMMA.884（Half-precision Matrix Multiply Accumulate, 8x8x4）18。
  * **指令分解**：尽管PTX层面可能声明一个16x16x16的矩阵乘法，但在SASS层面，这会被编译器分解为一系列的 HMMA.884 指令。
  * **4步执行机制**：反汇编分析显示，HMMA.884 指令通常带有 STEP 修饰符（STEP0, STEP1, STEP2, STEP3）。这表明硬件内部可能通过4个周期的流水线步骤来完成一个较大的瓦片计算，复用加载到操作数收集器（Operand Collector）中的数据，以掩盖寄存器读取延迟20。
  * **吞吐量差异**：一条 HMMA.884 指令触发了Warp内所有线程协同工作，相当于瞬间发射了512次浮点运算（32线程 x 8 Tensor Cores/SM 的视角，或者按硬件单元视角）。相比之下，一条 FFMA 指令仅触发32次运算（32线程 x 1 Core）。

## ---

**6\. 数值精度与算术行为分析**

对于高性能计算专家而言，硬件的数值行为（Numerical Behavior）至关重要。V100的Tensor Core在追求极致速度的同时，在IEEE 754合规性上做出了一些权衡。

### **6.1 舍入模式：RN vs RZ**

* **CUDA Core**：严格遵循IEEE 754标准，默认采用“向最近偶数舍入”（Round-to-Nearest-Even, RN）。这是科学计算的标准，能最大程度减小舍入误差的累积。
* **Tensor Core (Volta)**：研究表明，V100的Tensor Core在进行FP32累加时，并不总是使用RN模式。在中间的乘积累加阶段，它倾向于使用**向零舍入（Round-towards-Zero, RZ）**，即截断模式11。
  * **影响**：RZ模式会引入轻微的向下偏差（Bias）。虽然在深度学习训练中，这种偏差通常被随机梯度下降（SGD）的噪声所掩盖，但在要求极高精度的数值模拟中，这可能导致结果漂移。值得注意的是，后续的Ampere架构（A100）在FP64 Tensor Core中修正了这一点，回归了IEEE标准，但V100的混合精度单元仍保留了这一特性。

### **6.2 归一化与非单调性 (Non-monotonicity)**

标准的浮点加法 ![][image5] 在每次运算后都会对结果进行归一化（Normalization），即调整尾数和指数，确保精度最大化。 然而，V100的Tensor Core为了节省功耗和面积，在执行点积（Dot Product）的中间累加步骤中，**不进行归一化**11。只有在最终结果输出到寄存器时才进行归一化。

* **非单调性后果**：这导致了一个违反直觉的现象——非单调性。即在某些极端情况下，增加输入数值的大小，反而可能导致计算结果变小（由于未归一化的中间值发生了截断）。这种行为使得在Tensor Core上进行严格的误差界限分析变得极其困难11。

### **6.3 次正规数 (Subnormal Numbers) 的处理**

早期的AI加速器为了简化设计，往往直接将次正规数（非常接近0的极小值）粗暴地刷新为0（Flush-to-Zero, FTZ）。这在训练后期梯度极小时可能导致模型无法收敛（Vanishing Gradients）。 V100的Tensor Core在这方面表现出色，它**原生支持次正规数**的输入和输出，没有采用FTZ模式11。这保证了模型训练在低精度下的动态范围，是V100能够成功训练大型网络的关键数值特性。

## ---

**7\. 性能基准与吞吐量分析**

### **7.1 理论峰值推导**

基于前述微架构参数，我们可以精确推导V100的理论性能并进行对比（以PCle版本为例）：

| 性能指标 | CUDA Core (FP32) | Tensor Core (Mixed) | 差异倍数 |
| :---- | :---- | :---- | :---- |
| **每时钟周期操作数 (Ops/Clock/SM)** | 64 FMA x 2 \= 128 FLOPS | 8 TCs x 64 FMA x 2 \= 1024 FLOPS | **8x** |
| **总SM数量** | 80 | 80 | \- |
| **GPU 核心频率 (Boost)** | \~1380 MHz | \~1380 MHz | \- |
| **理论峰值 TFLOPS** | \~14 \- 15.7 TFLOPS | \~112 \- 125 TFLOPS | **\~8x** |

**数据解读**：125 TFLOPS的Tensor Core性能正是来源于其相对于CUDA Core 8倍的硬件并行度。这是一个纯粹的暴力美学：通过牺牲通用性，换取了特定运算的极致密度。

### **7.2 算术强度与实际表现**

理论峰值并不等于实际性能。Tensor Core的庞大吞吐量对内存带宽提出了严峻挑战。V100配备了900 GB/s的HBM2显存，但即便如此，要喂饱125 TFLOPS的计算单元，算法的\*\*算术强度（Arithmetic Intensity）\*\*必须极高。

* **cuBLAS 基准测试**：在实际的矩阵乘法（GEMM）测试中，当矩阵尺寸（M, N, K）较小时，Tensor Core的性能无法完全发挥，甚至可能因为流水线启动开销（Setup Overhead）而不如CUDA Core。
* **性能阶跃**：研究数据显示，只有当矩阵维度超过一定阈值（通常 ![][image6] 或 ![][image7]），且维度是8或16的倍数（满足对齐要求）时，Tensor Core的性能才会通过cuBLAS库得到释放，达到80 TFLOPS以上的实测性能，这大约是CUDA Core FP32 GEMM性能（\~15 TFLOPS）的5-6倍23。
* **Roofline模型分析**：在Roofline模型中，Tensor Core抬高了计算顶板（Compute Ceiling）。这意味着原本在CUDA Core上属于“计算受限”（Compute Bound）的任务，在Tensor Core上可能瞬间变成“带宽受限”（Memory Bound）。因此，软件优化（如Tiling、Prefetching）变得至关重要。

## ---

**8\. 能源效率与功耗分析**

在数据中心，每瓦特性能（Performance per Watt）是比单纯的性能更重要的指标。V100的TDP（热设计功耗）为250W-300W。

### **8.1 专用电路的能耗优势**

Tensor Core的能效优势源于其作为ASIC（专用集成电路）般的特性。

* **指令开销摊薄**：如前所述，CUDA Core每执行一次运算都要消耗能量在取指和解码上。Tensor Core的一条指令对应数百次运算，使得用于“管理”的能量占比极低，绝大部分能量用于“计算”。
* **数据移动节能**：混合精度计算中，数据读取主要为FP16。相比FP32，FP16的数据移动能耗减少了一半。考虑到在现代7nm/12nm工艺下，数据移动（Data Movement）的能耗往往高于浮点运算本身的能耗，这一改进对总功耗的降低贡献巨大。

### **8.2 实测数据**

在针对线性方程组求解（HPL-AI基准）的研究中，使用V100 Tensor Core实现的混合精度求解器，相比传统的FP64 CUDA Core求解器，在达到相同精度的前提下，**能源效率提升了5倍**14。这意味着完成同样的科学计算任务，使用Tensor Core不仅快4倍，而且节省了80%的电力。

## ---

**9\. 应用场景与软件生态**

### **9.1 深度学习：cuDNN 与 CUTLASS**

Tensor Core是现代深度学习框架（TensorFlow, PyTorch）背后的物理引擎。

* **cuDNN**：NVIDIA的深度神经网络库通过“隐式GEMM”（Implicit GEMM）算法，将卷积运算（Convolution）转化为矩阵乘法，从而利用Tensor Core加速。对于ResNet-50、BERT等模型，启用Tensor Core通常能带来3倍以上的端到端训练提速25。
* **CUTLASS**：为了让更多开发者能利用这一硬件，NVIDIA开源了CUTLASS库。它是一套C++模板库，允许开发者以类似编写CUDA Core代码的灵活性，通过模板元编程调用Tensor Core，实现自定义的矩阵层操作，打破了cuBLAS的黑盒限制26。

### **9.2 HPC：混合精度迭代优化**

在传统的科学计算领域，Tensor Core并非毫无用武之地。

* **迭代优化（Iterative Refinement）**：这是一种算法技巧。利用Tensor Core极高的FP16吞吐量，快速计算出一个近似解（耗时 ![][image8] 的LU分解）；然后利用FP64 CUDA Core对余数进行高精度的迭代修正（耗时 ![][image9]）。
* **结果**：这种“混合搭配”策略使得V100在Top500超级计算机的Linpack基准测试中，能以半精度的速度跑出双精度的结果，开创了“AI for Science”的新路径27。

## ---

**10\. 结论：异构融合的未来**

NVIDIA V100 Volta架构并非简单的性能升级，它是计算哲学的一次深刻变革。通过详细对比，我们得出以下结论：

1. **分工明确**：**CUDA Core**继续扮演着通用并行处理器的角色，负责逻辑控制、地址计算、高精度物理模拟以及处理那些无法矩阵化的复杂算法。它是GPU的“大脑”。
2. **极致专用**：**Tensor Core**则是GPU的“心脏”，它是一个为了单一目的（矩阵乘法）而牺牲了灵活性、舍入精度和指令粒度的暴力计算引擎。它将摩尔定律的红利全部押注在了AI数学原语上。
3. **协同效应**：Volta架构的成功在于它没有抛弃CUDA Core，而是通过独立线程调度和统一内存架构，让两者在同一个SM内紧密协作。CUDA Core处理数据预处理和控制流，Tensor Core处理繁重的数学运算，两者共同构成了现代AI算力的基石。

V100不仅定义了2017年的高性能计算标准，更为后续的Ampere（A100）和Hopper（H100）架构奠定了基础。随着后续架构引入TF32、BF16以及FP8，Tensor Core的功能日益丰富，但Volta V100作为这一异构时代的开创者，其CUDA Core与Tensor Core的设计二元论，依然是理解现代GPU架构最核心的钥匙。

---

**表格：V100 CUDA Core 与 Tensor Core 核心参数对比总结**

| 特性维度 | CUDA Core (FP32) | Tensor Core (Volta) |
| :---- | :---- | :---- |
| **基本运算单元** | 标量 FMA (![][image10]) | 矩阵 FMA (![][image3]) |
| **每SM数量** | 64个 | 8个 |
| **单周期算力 (每SM)** | 64 FMA (128 FLOPS) | 512 FMA (1024 FLOPS) |
| **输入数据精度** | FP32 | FP16 |
| **累加数据精度** | FP32 | FP32 (或 FP16) |
| **指令级并行** | 线程级 (SIMT/SIMD) | Warp级协同 (Cooperative) |
| **数值舍入模式** | RN (IEEE标准) | RZ (向零截断) |
| **主要应用领域** | 通用逻辑、物理模拟、图形渲染 | AI训练/推理、稠密线性代数 |
| **编程接口** | CUDA C++ (直接编写) | WMMA API / cuBLAS / CUTLASS |
| **能效特征** | 较低 (高控制开销) | 极高 (摊薄控制开销) |

*(报告撰写完毕，基于专家视角整理)*

#### **Works cited**

1. NVIDIA TESLA V100 GPU ACCELERATOR, accessed February 19, 2026, [https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf)
2. Volta (microarchitecture) \- Wikipedia, accessed February 19, 2026, [https://en.wikipedia.org/wiki/Volta\_(microarchitecture)](https://en.wikipedia.org/wiki/Volta_\(microarchitecture\))
3. V100 vs H100 vs A100: NVIDIA Tesla GPU Comparison Guide \- Cyfuture Cloud, accessed February 19, 2026, [https://cyfuture.cloud/blog/v100-vs-h100-vs-a100-nvidia-tesla-gpu-comparison-guide/](https://cyfuture.cloud/blog/v100-vs-h100-vs-a100-nvidia-tesla-gpu-comparison-guide/)
4. NVIDIA TESLA V100 GPU ARCHITECTURE, accessed February 19, 2026, [https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
5. NVIDIA TURING GPU ARCHITECTURE, accessed February 19, 2026, [https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)
6. NVIDIA VOLTA ARCHITECTURE, accessed February 19, 2026, [https://www.olcf.ornl.gov/wp-content/uploads/2018/12/summit\_workshop\_Volta-Architecture.pdf](https://www.olcf.ornl.gov/wp-content/uploads/2018/12/summit_workshop_Volta-Architecture.pdf)
7. NVIDIA A100 Tensor Core GPU Architecture, accessed February 19, 2026, [https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
8. tesla v100 gpu, accessed February 19, 2026, [https://courses.grainger.illinois.edu/cs433/fa2018/slides/Tesla\_v100\_GPU.pdf](https://courses.grainger.illinois.edu/cs433/fa2018/slides/Tesla_v100_GPU.pdf)
9. Is it possible to have FP Unit and INT Unit in a same core work in parallel?, accessed February 19, 2026, [https://forums.developer.nvidia.com/t/is-it-possible-to-have-fp-unit-and-int-unit-in-a-same-core-work-in-parallel/71086](https://forums.developer.nvidia.com/t/is-it-possible-to-have-fp-unit-and-int-unit-in-a-same-core-work-in-parallel/71086)
10. CUDA Programming: An Introduction to GPU Architecture | by muhammed ashraf | Medium, accessed February 19, 2026, [https://medium.com/@muhammedashraf2661/cuda-programming-an-introduction-to-gpu-architecture-dfd8dfffa13f](https://medium.com/@muhammedashraf2661/cuda-programming-an-introduction-to-gpu-architecture-dfd8dfffa13f)
11. Numerical behavior of NVIDIA tensor cores \- PMC, accessed February 19, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7959640/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7959640/)
12. Programming Tensor Cores in CUDA 9 | NVIDIA Technical Blog, accessed February 19, 2026, [https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
13. Understanding GPU Architecture \- GPU Example: Tesla V100 \- Tensor Cores, accessed February 19, 2026, [https://cvw.cac.cornell.edu/gpu-architecture/gpu-example-tesla-v100/tensor\_cores](https://cvw.cac.cornell.edu/gpu-architecture/gpu-example-tesla-v100/tensor_cores)
14. Mixed-precision iterative refinement using tensor cores on GPUs to accelerate solution of linear systems \- PMC, accessed February 19, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7735315/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7735315/)
15. volta tensor core training | ornl, accessed February 19, 2026, [https://www.olcf.ornl.gov/wp-content/uploads/2019/11/ORNL\_Tensor\_Core\_Training\_Aug2019.pdf](https://www.olcf.ornl.gov/wp-content/uploads/2019/11/ORNL_Tensor_Core_Training_Aug2019.pdf)
16. Questions about mma instruction with Nvidia ptx \- Stack Overflow, accessed February 19, 2026, [https://stackoverflow.com/questions/78747827/questions-about-mma-instruction-with-nvidia-ptx](https://stackoverflow.com/questions/78747827/questions-about-mma-instruction-with-nvidia-ptx)
17. CUDA C++ Programming Guide (Legacy) \- NVIDIA Documentation, accessed February 19, 2026, [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html\#wmma-api](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-api)
18. Reported number of hmma instructions by Nsight Compute \- NVIDIA Developer Forums, accessed February 19, 2026, [https://forums.developer.nvidia.com/t/reported-number-of-hmma-instructions-by-nsight-compute/255879](https://forums.developer.nvidia.com/t/reported-number-of-hmma-instructions-by-nsight-compute/255879)
19. The HMMA.884 tensor core instruction seems not match with its ..., accessed February 19, 2026, [https://forums.developer.nvidia.com/t/the-hmma-884-tensor-core-instruction-seems-not-match-with-its-cuda-warp-level-mma-instruction/304054](https://forums.developer.nvidia.com/t/the-hmma-884-tensor-core-instruction-seems-not-match-with-its-cuda-warp-level-mma-instruction/304054)
20. UC Santa Barbara \- eScholarship, accessed February 19, 2026, [https://escholarship.org/content/qt6d82g5r7/qt6d82g5r7.pdf](https://escholarship.org/content/qt6d82g5r7/qt6d82g5r7.pdf)
21. How many tensor cores to execute the wmma.mma.sync.aligned.{alayout}.{blayout}.m16n16k16 instruction？ \- CUDA Programming and Performance \- NVIDIA Developer Forums, accessed February 19, 2026, [https://forums.developer.nvidia.com/t/how-many-tensor-cores-to-execute-the-wmma-mma-sync-aligned-alayout-blayout-m16n16k16-instruction/353629](https://forums.developer.nvidia.com/t/how-many-tensor-cores-to-execute-the-wmma-mma-sync-aligned-alayout-blayout-m16n16k16-instruction/353629)
22. Numerical Behavior of the NVIDIA Tensor Cores Fasi ... \- MIMS EPrints, accessed February 19, 2026, [https://eprints.maths.manchester.ac.uk/2761/1/fhms20.pdf](https://eprints.maths.manchester.ac.uk/2761/1/fhms20.pdf)
23. NVIDIA Tensor Core Programmability, Performance & Precision \- arXiv, accessed February 19, 2026, [https://arxiv.org/abs/1803.04014](https://arxiv.org/abs/1803.04014)
24. NVIDIA Tensor Core Programmability, Performance & Precision \- arXiv, accessed February 19, 2026, [https://arxiv.org/pdf/1803.04014](https://arxiv.org/pdf/1803.04014)
25. NVIDIA V100, accessed February 19, 2026, [https://www.nvidia.com/en-au/data-center/v100/](https://www.nvidia.com/en-au/data-center/v100/)
26. CUTLASS: Fast Linear Algebra in CUDA C++ | NVIDIA Technical Blog, accessed February 19, 2026, [https://developer.nvidia.com/blog/cutlass-fast-linear-algebra-in-cuda-c/](https://developer.nvidia.com/blog/cutlass-fast-linear-algebra-in-cuda-c/)
27. Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers \- The Netlib, accessed February 19, 2026, [https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar\_fp16\_sc18.pdf](https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar_fp16_sc18.pdf)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIcAAAAYCAYAAADQ1+6cAAAE0klEQVR4Xu2YechtUxjGH/M8z5m6mUnmJFM3kjFzolCm+EPyBxkic+YhQ2TIVNyuKRIRIi7KkJmMITJkDCHx/O579v3WWWefc/bZ3/V9X7f11K9T+9377LXXeqe1pKKioqKioqKioqIpovnMdPOU+dX8a74yL5pZ5k3zu3nJ7GMWiMemlNY0j5pTzYKZbSK0rLncvKeYvz/N64r5A65/bi4xK3eeaSr++3QzLTe00PrmfPOyYo0/UIzzBLOSOdPsNefuREuZu83XZufMxuQ/aL43h5r5u82TqoXNGYoFucIs0m2eUO1ivjE3msWT68wt1/5ROMiiiW2YVjCXmg1ywwji/eeaX8yzZl+zdMe2jDnHfGjeNZt2rncJz3zFvGDWyGxoG/Ox+tsnQ2Q9xkVkfWvuMEt23dErniGC1s0NiQiGzTVaEJBRTzI/m6PU/SzvPML8Zu5XLEhTrWguU3vnWM7cpng3mbXOMZmLV81Divt7tJv5zlyt+j9gkE+YL8x2mW2ytJo5UJHpPlGUluW77qjXFooJ2zo3WBuae80eikVtKpzyLvOpwmFTMaYZ5g9zokYrzeNxjiqr4hjnqX5dEQ5xn6KsLJTZZl/AQNo5XPWTQk16UpE2SZ91IlqOVvQo1N4mfGS25OERxYfvrohyJu5t87xZPb1pgHZQOMh6ybVVzHXmYI2WNRDZ6A3zmGJBEf+xrSIifzBnKVL8KBqPc+yk6HWoCOtktlRLmFPUJ+jx7IcVC0VU1WljRU36TL2RMRki7W+vcGQchAb6LcUiNRELR4N9u2LiWASy5nGqiZ4hYgz7mZ8UmfUeRYN6jaLpY2w7du4bVW2dg57nBvO3InuM+k1zxETTkDyuMa9PxUcdpKino0Tn/yUi/BCNNVVVViOlb1Xd1EA4CItKH0ApoWtvM4lksYsU2QGHS51gVfOA+dLsmdlSLWauV29mHQRlvt/uhz6CTMaYKJGtxGApJWxjL1R8aK7UC89WuwmcW2J8jOEv9U7WoJLXT/QtLB6NNtmxjXBOAottf5650maU3eBElRUqABuI980mmS0VW3+ctnaXgsdSZ0mJB6jes0nfRCXpMa3Rudr0HGSsfqUsF2OjjhLhaXOF896s6JnIcHXfUCei7iZFF7+/osQM+r5+IluxEGSfKptVYk4YL83oTPXah6mtc0xT9BqMa9D84sxss2t7kqpev2M2ymyIl3BAxovaNI5zU5Szi9UbnZxtcMbBAhyvZs0kh0tkQ84d2GlUPcgtZu3kvmGqMi8l9zT1HsKtZZ5WZOZ8i9tEbZ2D4LlKkbGOUf0OidJzpzlM9fbZ20BOy0itTFgl0vfe5jXFFm2yzzaIuAsUBzY4QyoWhFJD2WNHkC9QLr6TiSNrpPt6JohDPrJQUwchaxF5nLPsmlzHCWjcn1E4BuMbdgZTp7bOgQhsyh3bfA6+Kgfgl1LCuk5XjcOyDWThObUjvf+oOFad1fnlyJeum2jKF2MixUJeq4hMxkkknKyxvoetIttHGi/slMdHzGYdey4mgghmJ1HXfDNxRBIHa4O+m4OsKzX2XiD7Mn9A/0FKv1XhJE1LXa7xOAeixznWPKcYEyfdZMsj1exMqGgKa7zOUTQPixKJY7BxKCoqKioqKioqKpr39R/kUgEd8HnxbwAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAXCAYAAACS5bYWAAAB90lEQVR4Xu2Wu0scURjFj1GjYvABojEgKXwgFhJFsDCVAUGIJhYi2imoWNiICQhBbFQkRRTjI0bCpguIqdMtKdKks/M/iJpCJaRRRM/h23HXcWZ21mJt5gc/hp17d+fc7z5mgYgIX1ppnHa6G7JIFd2hkzTX1XZNMV2nh/SFqy1bKNw4PaVTic+3yKG99C89wv2FbaJ79AwBYVX6LfqDHuB+whbSd3SXHsMnrG4M0zm6QP8gfVjNRAOtczekUEOf0QfuBh866Bodhc2wZ9haukmb6SzChRUt9AttczeQRvqNdsMGlo5SukRf0lfwCfuQTtMx2DRkElY8hwWuT7mnJfWR9iNcVZ39skhLaA98wqoqKv0TmofMwyqMfjwGm6EKugwbfH6yWyCPYYNrhwX3DPuIztO+RKe7hBUK/Bq2MTT1EwgfVGFG6FtakLjnGbaLvoetF3HXsKKafqe/YMdPWLRJdbbr6uAZVsEu6KWPv+lTp3MAlfQTrDqapRhuruEgFOwEt5/tqGPU822qDbaKzN5gZbDKaCdrWTlreBvhBupG39eyCHyDiSK6AXuDaYmkQ0E/wKpannJfDxikn5F5YIXVOfuPvoFHWHUYoPtIlv8//Qo7hrxwKrACOwHc6CFDdAbJjZMOnUw/6Tksg65x2J+riIiICHIFLNRj1rHq4WMAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEMAAAAUCAYAAADWQYA8AAACjUlEQVR4Xu2WyauNYRzHv2YiQ8lUsjAkyZRIhjKklHEhsaNMC0MyRJINEoXMQ7qmMsTKwu5mYWNnJ/+AaYFkQ+Lz7Xdf59z3Ovc877s4Z3M+9el0z3PuOe/zfX7P83ukFi1alGQWtuOS/EADGYlPcBf2yo01jIF4BT/i0txYo/Dkt+M33Nfxd8PpgavxM35S88KYgm/wp5oYhkvzBr7AD2pOGP3xKD7FL2pSGP7BzXgcT+J71Q/DlTQJJ+QHqhiLM7BnfqAG8/EyblVUaEoYXkSfc7U+NwTnKYJOYjxew2l4TGlhmJl4G2fnB2AyPsQViuDq4Yc+jStxjdLDGIUXcIO6fnYYXsTd2C839l/64n7cpkivSBhmgSKQiVXvebUu4XqlVUV2Xp3CwbhK6WGYcXgTl6kSvOdyCI/ggI736uJVdWmOwd4qHoYn64dvU1TYcDyvCLdP5WPd4tV1eHMVkykahpmK93CxYvJ7FZXmiktiEJ7AdYqHKBOGcSBrFQeft8ZOpQfhyW7Bg6qUcpkwzHR8hHfxLA7tPNw9y/GMKumVDcOMxmf4StEeU/Eh7LuNXzPKhuGK8OK+U3xHyln1D0/8N/6p4WvFfqzHCLyuWF1XWZs6nyHd4Yf+qq6/nek2n3Ibzs4IN4JFiurwghYKpBp/oU/fIjdQl6JX1vvT2y47Q24pLcg8/n9vmyI3UG/JPfhY0c6NzxAHslAlA3GZXVXcQL2F6uEgzimqwm0swxPYqDjhiwbiMHzP+I4HVD8MB7EDnyvaeTVz8L7i/pIciB/AffqtKuX5A+8o2uT/yFbQPd4dJI8nsQkPK7HHKzrbS/yleAa/tisuVbXwRB+o9jnlDuXu5q3cokWL8vwF3zR3RK8xPywAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGwAAAAXCAYAAADug6rPAAAFMElEQVR4Xu2YZ8xkUxzGH733XnfVVVaiS7QQwSISPojOm9UiSnxAwmqJiBDRO2vJaokShOgsi+gSItFt1FWCECR84PnNf847Z87cKSHzzpf7JL/svveee+bc86/nSrVq1apVq9ACZldzj3nWPGJOM8vlgyZIU8y15lXzvDnOLNY2oiXWfah50kwq7k2EllD8/k5m4ea1Ncy5Zvc0qNDiZjXF2ktx/Trzs/nJ3Gm2aBvR1DZmttleMdEq5m5zjVk6GzdM8bu85JvmdMWL7WFeM3tn43Kta15QPLNecW8itKJ53PxTcJdZPRu3oFnbHGaeU+ztstl9hLG4foPZ1IyZb8y7ZmprmLSoucCcpZaXIDZrrtk6uzZMrWmeUjgOL0NUXWn+NCcqXjrXImaG+UujNdhDZr75VbFfp5hlsjHs6RnmA4VxfzH3q91gKVOQ4TAc4rkzFe93lSKaGyKCZio2aoV00drOzDE7ZNeGJRZ8lPnBTFcYh2us4XxVGwNHute8odEZjP26UVFOBtEm5n11GoyguVjhnOc0/0a7mK8V7zi5eW18MKH8gKIW4L2kpQfVHtql2NSNzYbljUzrmC3VGSG5ljS3mO8Ukd1PS5kLzfEKDx+VwZZXpLD/azCiiQz3t7lDrTJEqfpMkRY3al5raDfzlcJo3GQRjyk2up+2MreZbcsbigUSBfuousgmkQbI7V8o0sczCiNcr3CIXMyDUS836yucapQGu9lcqmjUPjFvmWmqdtBuBkNE636KpiVpf0UD8oSirxgXFj5YUeQwGpYmb+Ypspd2Vhgt9wKMQLd3kKoXn4uofl3xu/cpNp/NuMm8qPZ5VzJXKBoUxgxiMH7/GPOHOhuEbrD5/eo3XTSNAgYjKqhdV5u3Vd3d9TJYKbIIDkttHFOxh6Q9CjwbwYalRd+q2JR+YjK84XazgVlZMR8pi/TaTzzzjvnNHKFWNBKZpMmzFfPwO4co6hpplA0bxGDD0kKK3x1vCBTR/625TNHp5hrUYLz/nuZLxTwYb1xMSg0jtJmEaDvAfGh+V3hmvwhBjOE56iBpkM5uEGOhtcxLirS8Y3adyKXoPm1WVUQiXpe8d9QGqxIl4lPF+/BeuXKD9Trj8i6ch6lnZJQ2EfavKPJnXmfYLAoeIV56SjeRf9nAl81mxb1eYlEcfj9Xey3EeBiR+Sabk81JajnCKA1GdI2Zj82xzb/R5uYj8546628yGE7dzWAEDedfuk8yVYf2UnxVoCPJRT7GwhgsD/luIgKoOZwdDlSkx7bOpodwCFIoKYCvBkl5hNGJYpyy1uTMUpE+mvovNQxDEC3dlHfX/JtacY4i8xR7ysE+Vz+Dsc+kf/Y9ncdwZpyUc2pDLAoPPVLtqQ/rktqmF9erRJ0jVV2iKL6pplEDJ2XjuonIJp0S0Rg7Rfq+5kfFvFWfp4joORpNhPGORys+JaUOjnWzfg7HdLFlZuplMLLGqeZhRfebRMTSABIQDTHpRYq8e4LCQ7nJAY6zUd5mVglj0awQXXlXSYrg9M4cgxiNZ2croojPOPzNnKTKbuc8asRcRVeWv+RECa8ndeFY1H4MR9NGx1tVEqYqegO+eORtOoaepsgm2IHPcYzD8EQw+5J/PWl0XBhrnuKhR83h6p8K8TIiEA+oyrcYje9nHAqrIqQUhmVx8xW1i0aoLNyIdZ2n6CBTCuP/M9Tp1cPWFIVjYSTOsLPU7jwYg2xDF8yxJa33e0WkEXXsDRFZpuUEGSal3Fq1atWqVatWrVq1Cv0LhS0hS/VinNMAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAYCAYAAACx4w6bAAACQklEQVR4Xu2WyctNcRjHv+Yx6rVRZBbylmGBEgkLLCzYECtlyO7NUIZio9j4AywkK1LKwlIpUxbmYWFOhsTCEIrE59tzr+753ft2zn3vpXdxPvXp1n3u79zzPM/v95wjlZSUlPwj+uEmvIhTsqFCDMAtuCT5vhn8v6fwFf7Gj3gdr+I1fIH3cSsOiSX5zMRb+AznJrEiDMRduCINNIkL3IWfFQn0rYmNx0v4CTdgn5pYQ0bgRjyH73BpNlwIJ7ZbrSc2DE8oujM/E5EG4VH8hfuxfzacxRVahgsUi1yptSpQjYRqYivTQJNMwht4AUcnsRl4G1+rQPEn43LFnt2HXxVnpXYLFKFdia3C93hYcU3jTq3Bm/gU1yunW8NxNY5SJLIdv+Fe5SxsQDsS8wA6oNhqd/A4Hql8vsXTOPHvr7vBibid8xTbzvpAflFsSVepEV63WVEAT64iPlGxgeQCn8d7OD2Jeb2/93RMYxk68a7qb8KeVHSzGdrRsTn4GM/gyCRWHRw/cKdiNtThKbgDZyXfL8Y3iqp1JLE8Wk3MO8aT2aN8j+qPwlA8hj/VzVFxpj582xR7uha328+xKzg2ieXRamLVjnxQ42ssUtzbc8UEz+CqLMSzODuJGe/dh/gApyWxPFpNbAxeVox6j/wqvq4fPz6nL3Gdkm55pPsZUD1Hfk2ZWom5zQcVryueSI47Qd/o4Mpv8uhpYn7c+MXgu+J//enkfH/2kWIWHMJxlTX/lZ4m1uvxNp+gGNklJSUlvZs/WBh2Gw3e8CUAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEkAAAAYCAYAAAC2odCOAAAENUlEQVR4Xu2XacweUxiGb4ra99r3UsQWQWJPxNLautGmVPSLqKSIP5ZYI02INWJfW2IJQluRiC4oXYTamjQNQdDYtyD8EES4L8855kz7fZ/3naY/JHMnV/LOzPPOzHnWM1KrVq1a9astzV3mR/ODeczsU7OI4zPNFul4DXO8udJsmI2Sdjd3mtfNy2aiGViz+J8JBz1u7jF7mh7zpVli9k42q5kx5mfzV8EX5hQzoLA7yrxlLjJrm6PNIjM02dTEH44zL5pvFTd9yWxV2GxrZpg/0vXvzbVm3cJmVYp3PM08oXAWIkMuMb+b28w6ye5U86ki25aZh8xB6VrWNmaOeVSRXWTPreZXM8msXpnWtal5SmHIA0arfuP8UnebjYrzvYnITDZPmgNUv08TrWWuU7wbZcMxOkKRJW+anRTPGWluVt8BxIZy/M6cpXAI53Dk1WbnynRF7avw7E2Kl5lmNimuEymyh7TtZNFEiKi8o7gn0WsqAnSZIpMfNuun8wTgY0XJ7ZbOjVA8ry8ncf4B842ixDoWix6riNZ+ioVRescWNpTcvarqv1NRtkSI5kj9l47vRvzvRLN1ce5kRdbPMoPSueGKhp4b8leKctw8Xadc5ypK8mJFm6E3USFDkk2voiYpjx5FqVyhiBqTJEfkSHO7mi+SEp2g6Cu8XO4tTbWeYmE06R5VfQQnLTSHKIJPSb6nCBTr3NG8oVjf04ry2tjcZ+arysgVRLTvNwemY0rvXfOB2V8xFc41F5o1k01TEYRx5nlzo9mlfrkjsXiy/DNF/8FhWQRxu2SDKM1HzIeK9Q02i80v5ozCjm0CJXi5+ljjwYpSyinLQng43qZRbmZuUR/jsaFovpTLPHO92aB+uV8RfSYw/Yl3609kD2uhz55jtldk2ufm0MLucMUQeEHVHutfkaZ0eRpjnhoIx32iaIpMjAfVLOq9iWcS1enmI8XLd7qJYyDcoQhq7jNZ9M3nzEyzQzpHVlxj/lS0Ecp8tmJtuXIQDsNxryoysSZSlbRnKpRTi/OUIDdfoGh+eaqsjNjlTjGvKDZ+Zan8l5iwlAMZlHsamXSeYnrSJigr2gQDCOU90G/mfMXzOKZUD0s2qN9MInWnmj2Wv2Ado6hTHnCBqh1rE/Fgpufb5lJFs+xGZATv8KzqGb2XIoDcn8nHNuYkVY2cHsV/yNi8qaQy2DqMSsfoBMUm+QYtl9Us+nRFly932FkshA0h23+mWxNx36sUY5qybpKNLGSYItIsls+H981Pii8AHENPwzHjFT2I53LMvo6sKdsJjuM/fEVQWhwz3SjDXZPNP6KbL1X1fbNM0UjLkuM33n5GUe/dCAdPVozVs9XMOVlElsFRfo+VEP3sAIYOWw0aO1PsNcUOu+y3iK0Ajvpa0YtoLd2ucaXEoiYqMqebqdWqVatWrVq1WqX6G0rLy1i8FzXEAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAXCAYAAAD3CERpAAACMElEQVR4Xu2VTYiOURTH/4yvJFESmayw8lUMWdiIhW/5/ih2SNRsLERsrJCoWdmYhpSFhTLNQlIoahpio0hEEomwEBK//3vunffOa+bFRqn3X7/e97n3PPece88955Ea+geaAdthfHoeAkvhEIzORoUGKWxH1k6gJlgBd+Er3IddqrH1AhvgI/woeAnrFItk+cV5cAwewrJizhoMG+EWrIVZcFHh/DAMy4Z2uh6ew3t4BmehJc1lTYGbcAN6FLbLi3nLu78AWxUBWFMVu34K89NYZeE1cEL9H1etfPSO+q1+dToTHiuCmp7GvOYZ+AKtKk5uNRxPBr9TPafT4AG8hkVpbDichG9wQPF+RavgPLTBHXgFp2FcNihUz6mPdA4sVHXxMXBJcWe2qEiZnTr5C9KgX/JFOaKItFQ9p/3Ja/m+XIWJ5cRYaFY1ilHQocjP3GyU9DdOXW7tig14Q3Xl3flifYadqt5Eq3TqehxIttujcOjgykrQJLgCXTA5jQ2Fo/AdDqpIfvpvp+80sFM78EW6rqjlMuiK8jV/pChmyzs9pbjme9W3QfyJ09nQqWguduggliiqpLKWk3tOsUCOyDm+DE8UTaJU6XRlzZzlZuCT2604Mcu/+2GT0jHb0TZFDiekZ0f4QlFXva0rKR/9B9isvrlysO3wCbrhnqKdug2+gcW9lmgE7IBrCsPbig9A6dA1687iHeb+7IJ3Q9+nsHWNut2VPTzjFOb0NdTQf6qfGIhxeUiXn6QAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAYCAYAAACvKj4oAAAD+ElEQVR4Xu2XaaiUVRzGn7RsMbVsN5M2aIMWSiEltIIIF6SEaHFDLbAooogWLKOsvpRCRGlpRhpokkpUH7KNooIWooWIFpG+ZAgthgj5oZ6f//c0Z85958KdZu79ch/4MTPveWfe89/PSIMacJ1gHjEvmNfM9ebA/Ib+1DjzhBlfLrSpg8wS86Q5ytxtPjCnVOvDzG1mYfW+qzraPGcWmaHFWrtKBm4wx5rbzafmjOye481qc40Zkl3vqDBosXneHFmsdUpHmLXmYXNIsTbFvGnOLa631HHmHrPJbFN47aPqWp0Bp5u3zGxzQLGG8PjLZrv5p3qdkK0fZh4zf1Xre81GM1bxexeZlWar6o1gT+sU0SbqLYVnSLHPzb3m8GztTPO+eVXx4CTS4kbztjk5u16KBy81f5p95kH1rJupCiNwWCmes8B8ojA4Fxl0qyIIefo2aaRZYb5VhLyMBJ+vNX8ovJ3SBCeQOk8rItFK1Oiz5gHzi/nanJ2tY8B8c58aho9QGIXhOOhys0Ph0LLeLjPfK/ZY7l0Hm/vNLnODen456Tzzo/lQjSieZN5VdLje2vf5Zo05x7xk/jZ3qvGdQ80yM1ONDdKV31FEnvW55jszuVrPRWdlX9jRtA9+7AqFVylU6q+VLlTUD95PqUC6fKPwfg/PVeL6deZRRZRnmN8UKX9idQ/GEOGzqs8IR7O35eZxRbe8SvV1Rh1uMc+oyCTS4EWFR+9Q6xbPJq9WpCi5niLIBn4ys6p76kQ6P6TIDu5hplHLuxWDG0NIMeYd3bIdYcf6Ct7/JyJBBHYqHtJKbJIhTpcj1YZX16cpcv/K6nOdxihm5AXVZwyaozCQTn2Mokkw53pL895E1FYpHDc6X0hp94PqW3ASazSgXxXGpGjxnrrozcCJiiZE5JI4glFfpCq19ZSiibQrDGSUvK7m5+xvyV8q0ix5uBQFToelvXMuzAftJQrDSd+6FCXlb1LM0Lx2uH6zYubx/FfUOIK1I9KSWci8HZUvpNRjyOLJsoOyEVr174omUNYITvnKzFO9gYwRfn+6eq6faj5TpD2jJqV9O2JfmxWZQECaROf6QvEwRkESnYkW/bNiPuWDP4nuRzfkYFBXP3TZN9R8ckkionxvjyKapXP7ojSu7lKLRnmawtPMEo5ntFzSBsPwdOn9JLxOA6ET592Lkw9jh7QmQjQU0rQ8vTAXccCk4npfdbEi1fM52hHhdbzPESqfYf0pDKIrv6cIVMfFketjRTP5P2nWrsgcRgTlVGZIR0TtcUjgCNY0g/pJlyrmH2XRNTGs6WCkSm2Rd0n8EaYH8Fet68+lo3Jm5PDQH+JPAplzi7qUmoMaaP0LeXCofNdmimgAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAYCAYAAACvKj4oAAAD90lEQVR4Xu2XWahWVRiG3wazcsxGTSUzaIIGyshENCWJBsQKabCUJqgooqgs1CJSbxRBI7XSoOHCoozELpoJKmggGqhouuimRGgwRLALex+/tTx7r7P/g+d4/nNuzgsPZ/977f/stb7vXd+3fmlA/a4J5knzpdliLjAH1Z7oQ403K82kcqCHGmKeMNek68XmQ3NKGj/M3GNuSddt1THmGXOrOaQY66lGmZcVQTvCTDM/KBacs3iCedbMNQene70uFnSH2WiOKsZ6SyyIhX1vpteH9n5+y5xV3G+p481C84p523xmPk73mhaAZd4x89S8P05VZOJXsyf9ZS9lHWmWm3/T+C6zyYytPDPCPG82pOuqmNMLZpEZVIzVdLjCYl+Yh83QythpCv+/ofqLscVt5l1zUuV+KV78qPnH/GceU+d9c5l5XR17LIvv4pA1CkuWwkF3K5JAMBs13Kwy3ylSXmaCz9eavxXRJhiIIDxnnlJkopXYo0+bJeZ38405ozJOoBaYR1RfOO+dau5TZO5kRWUtNcP8qJhjOXcNVlSo7eYGtd6sZ5ufzUfqyOI48755yBya7jXpHIW9zjQvmd3mfnV8hyJCxZyt+gTZVxSZOeYS86C5sDKexaKZF+uozYN/NksRVTYq+6+VzlPsH6KfrXC++VYR/U6RS+L+dWaZIstXmj8Vlj8xPUOLIcOnp8+IKkodYF9mqKIEqRT7cLNZq8JJwxSbl4hig1YlnklepbAoXs8ZJDi/mKvTM03Czo8r3MEzRyv28g5zvcIxWGy1GZm+012xjhcTXO8TmSADfyhe0kpMEqsQRaxG40WXK7x/afrcpDGKHnlu+syCblQskAwdqygS96prm3clsrZeETgyv0/Zdj+p6z7CGAVom2IxOVtcY5uuFniRogiRuazR5j2FVW9SHMdmVsa7Kxa4zmxV/T17S/JXCpvlCJeiAFBhKe9L1VFBERWOhWPfJoti+dsVPbTao7h/p6Ln8f5X1Vwd91fYkl5Iv631yWw9miyRLCsoE7nZ/KUoAuUeIShfm/lqXiBthP9/hTqPU/I/V9ieVpNt3xMxr9cUTiAhNVG5OK3zMlpBFpWJ5vyboj9VG38W1Y9qyMGgaf9QZd9U/eSSRUb53k5FNsvgdke5XT2gFoVyoiLS9BKOZ5RcbMPCiHQZ/SyiTgGhElerFycf2g62JkMUFGxanl4o+QRgSnG/u5qssHrZRw9YRJ3of6p6D+tLsSCq8geKRPW6OHJ9oigmB2Kzngrn0CLYTqVDekXsPQ4JHMFqPaiPdLGi/7Et2iaaNRUMqzRu8jbpOEUN4Kda299LRV2hODz0hfiRgHPuUpusOaD+1v/3MahrAoBDBgAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAUCAYAAADhj08IAAABNklEQVR4Xu3Vu0oDQRjF8eMNLcQLeEGIWHhBxEIhZV7AQtDCwpQWgg+g4CNYqSCCKGIlFmprl87aStBXsLK30P/Hp2QcCJIZIRJz4Nfst2zOzs5mpVaaOD0YRVs8aETaUUAZFVyi79sZDUgntvGEO7ziWn+gWJhZPKoJitk+nMFUPAgyjgX5dklOvcUsizhHMR7Ir3eFJWW+TCnFLCV5uengmL3ZR1hT5mpZUovZDy/jApMYwgE20VU9LT1hsf5o9lOs3Apu5I9vS79UyvJVzC5ebzHLGG5xj7lolpWcYiM4wQ5W5Y813HNZSS02gGPsoVfVPXeGieC85MzjWf4FGI5mtWKl9uWrNRgc78A6TpVYzv5f7O4e8Ib3Ty/ylbNVrBVbmQ0cyt/EOFbOvr+76I5mrfyPfADdjTBCiJbbLwAAAABJRU5ErkJggg==>