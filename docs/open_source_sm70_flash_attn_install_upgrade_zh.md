# 1Cat-vLLM + FlashAttention V100 安装与升级指南

这份文档面向 `Tesla V100 / SM70` 的开源用户，目标不是“最保守的公开发布默认值”，而是给出当前 `1Cat-vLLM` 在 V100 上更实用的推荐路径：

- attention 默认推荐走 `FLASH_ATTN`
- `FLASH_ATTN_V100` 作为次级后备
- `TRITON_ATTN` 作为稳定排障后备

如果你只想使用公开 release wheel，并接受文档里的保守默认值，请先看 [OPEN_SOURCE_SM70_GUIDE.md](OPEN_SOURCE_SM70_GUIDE.md)。  
如果你是 V100 用户，且希望优先拿到更好的 attention 性能，请按本文执行。

## 先说结论

当前仓库已经不是“装完 `flash-attention-v100` 之后，再手改 `site-packages/vllm/.../flash_attn.py`”的旧状态了。

当前仓库已经内置了这三件事：

1. `FLASH_ATTN` 的算力门槛已经放宽到 `SM70`
2. `SM70` 上的 attention 后端优先级已经是 `FLASH_ATTN -> FLASH_ATTN_V100 -> TRITON_ATTN`
3. CUDA rotary 路径已经避开了对顶层 `flash_attn` 包的无谓探测，减少 ABI 噪音

这意味着：

- 对当前仓库，`flash-attention-v100` 的推荐安装方式，是把它装进你实际运行 `1Cat-vLLM` 的同一个 `conda` 环境
- 不再推荐沿用 `flash-attention-v100` 仓库 README 里那种“先装一个单独 `.venv` 里的官方 `vllm`，再手改 `site-packages`”的方式

## 本文适用范围

- 操作系统：Linux x86_64
- GPU：`Tesla V100`，计算能力 `SM70`
- CUDA：`12.8`
- 安装形态：源码仓库 + `conda` 环境
- 推荐使用方式：`editable install` 的 `1Cat-vLLM` + 本地 `flash-attention-v100`

本文不以 release wheel 为主线。原因很简单：  
公开 release 文档当前仍以 `TRITON_ATTN` 作为保守默认值，而本文要解决的是“V100 上把 `flash-attention-v100` 当成推荐默认路径”。

## 当前推荐环境

下面这组版本是当前仓库已经实际验证过的组合：

- Python：`3.13`
- PyTorch：`2.10.0+cu128`
- Triton：`3.6.0`
- CUDA toolkit：`12.8`

如果你沿用自己的现有环境名，例如 `gptq`，可以直接把本文里的环境名替换掉。  
如果你更偏向公开 release 的 wheel 路线，请改看 [OPEN_SOURCE_SM70_GUIDE.md](OPEN_SOURCE_SM70_GUIDE.md)。

## 仓库关系先看清

`1Cat-vLLM` 当前有两条和本文强相关的依赖线：

1. `flash-attention-v100`
   - 用来提供 V100 上的 `FA2` 路径
   - 建议作为 V100 用户的 attention 默认路径
2. vendored `lmdeploy`
   - 主要用于 `SM70 AWQ GEMM` / TurboMind kernel 供给
   - 只有在你要升级 `SM70 AWQ` 底层 kernel 时，才需要碰它

换句话说：

- `vllm` 可以相对勤升级
- `lmdeploy` 建议保守升级

## 推荐安装路径

### 1. 准备源码

假设你有这两个本地目录：

```bash
/path/to/1Cat-vLLM
/path/to/flash-attention-v100
```

其中：

- `/path/to/1Cat-vLLM` 是当前仓库
- `/path/to/flash-attention-v100` 是本地 `flash-attention-v100` 源码目录

### 2. 创建并激活 conda 环境

下面用一个通用环境名 `1cat-vllm-fa2` 举例：

```bash
conda create -n 1cat-vllm-fa2 python=3.13 -y
conda activate 1cat-vllm-fa2
```

如果你已经有现成环境，例如 `gptq`，直接在那个环境里继续即可。

### 3. 安装 PyTorch 与基础构建依赖

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install cmake ninja packaging jinja2
```

如果你本机 CUDA 不是默认指向 `12.8`，建议显式导出：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
```

### 4. 安装 1Cat-vLLM

推荐用源码可编辑安装：

```bash
cd /path/to/1Cat-vLLM
python -m pip install -e . --no-build-isolation
```

如果你刚升级过 `vllm` 源码、或者改过 `csrc/` / `CMakeLists.txt` / CUDA 相关文件，也用这条命令重新安装。

### 5. 安装本地 flash-attention-v100

这是本文最关键的一步。  
推荐直接把本地 `flash-attention-v100` 装进同一个 `conda` 环境：

```bash
python -m pip uninstall -y flash-attn
python -m pip install --force-reinstall --no-build-isolation --no-deps -v \
  /path/to/flash-attention-v100
```

说明：

- 这里安装的是本地 `flash-attention-v100` 包，不是 PyPI 上的其他 `flash-attn`
- 推荐和 `1Cat-vLLM` 装在同一个环境里
- 不建议直接执行 `flash-attention-v100` 仓库里的 `build.sh`，除非你就是在它自己的 `.venv` 里工作
- 对当前 `1Cat-vLLM`，不推荐再去手改 `site-packages/vllm/.../flash_attn.py`

### 6. 为什么这里不建议沿用旧 README 的“手工 patch site-packages”

`flash-attention-v100` 原 README 的做法，是：

1. 在它自己的 `.venv` 里安装一个上游 `vllm`
2. 构建安装 `vllm-flash-attn`
3. 再手工把 `supports_compute_capability` 从 `8.0` 改到 `7.0`

这个流程对当前 `1Cat-vLLM` 已经不是推荐路径，因为当前仓库本身已经把这些改动内建进去了。  
对这个仓库，正确做法是：

1. 安装当前仓库自己的 `vllm`
2. 再把本地 `flash-attention-v100` 装进同一个环境

## 安装后如何验证

### 1. 检查包是否装到同一个环境里

```bash
python -m pip show vllm vllm-flash-attn
```

你至少应该看到：

- `vllm`
- `vllm-flash-attn`

### 2. 检查 SM70 上是否具备 FA2 条件

在一张 V100 上执行：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from vllm.platforms.cuda import CudaPlatform, _get_backend_priorities
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.vllm_flash_attn import flash_attn_interface as fai

cap = CudaPlatform.get_device_capability()
print("capability", (cap.major, cap.minor))
print("priorities", [x.name for x in _get_backend_priorities(False, cap)])
print("flash_attn_supports_cap", FlashAttentionBackend.supports_compute_capability(cap))
print("fa2_available", fai.FA2_AVAILABLE)
print("fa2_supported", fai.is_fa_version_supported(2))
PY
```

在 V100 上，期望结果接近：

```text
capability (7, 0)
priorities ['FLASH_ATTN', 'FLASH_ATTN_V100', 'TRITON_ATTN']
flash_attn_supports_cap True
fa2_available True
fa2_supported True
```

### 3. 注意一个常见误区

如果你直接执行：

```bash
python -c "import flash_attn"
```

即使这里报错，也不一定代表 `1Cat-vLLM` 的 attention 链路不可用。  
对当前仓库，更关键的是：

- `vllm.vllm_flash_attn.flash_attn_interface` 能否正常判断 `FA2_AVAILABLE=True`
- `vllm serve` 实际启动后，是否能选中 `FLASH_ATTN`

也就是说，不要把顶层 `import flash_attn` 当成唯一验收标准。

## 推荐启动方式

即使当前仓库在 `SM70` 上已经默认优先 `FLASH_ATTN`，仍然建议你显式指定它。原因很直接：

- 配置更清楚
- 升级后行为更稳定
- 排障时更容易判断当前到底走了哪条路径

示例：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
vllm serve /path/to/Qwen3.5-27B-AWQ \
  --tensor-parallel-size 2 \
  --quantization awq \
  --dtype float16 \
  --port 8000 \
  --attention-backend FLASH_ATTN \
  --skip-mm-profiling \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --max-num-seqs 4
```

如果你只是想快速 smoke test，也可以把模型换成你自己当前最熟悉的一份 AWQ 模型。

## 当 FLASH_ATTN 不可用时怎么办

建议按这个顺序回退：

1. 先检查 `fa2_available` / `fa2_supported`
2. 仍有问题时，改成 `--attention-backend TRITON_ATTN`
3. 如果你明确想测试 V100 专用后备路径，再试 `--attention-backend FLASH_ATTN_V100`

推荐判断方式：

- 首次 bring-up：优先先跑 `FLASH_ATTN`
- 如果服务起不来：先回退 `TRITON_ATTN`
- 如果只是想对比 V100 专用后备：再显式试 `FLASH_ATTN_V100`

## 升级 vllm 的方法

### 原则

`vllm` 是上层框架接入层，可以相对勤升级。  
但每次升级后，都要重新确认 `SM70 + FA2` 的几个接缝没有被上游改断。

### 推荐步骤

1. 先把你当前分支的本地改动提交或导出 patch
2. 合并或 rebase 上游 `vllm`
3. 重点检查下面这些冲突高发点
4. 重新安装当前仓库
5. 重新安装本地 `flash-attention-v100`
6. 跑最小回归

### 升级后的重点检查点

- `vllm/platforms/cuda.py`
  - `SM70` 的 backend priority 是否仍然合理
- `vllm/v1/attention/backends/flash_attn.py`
  - `supports_compute_capability` 是否仍接受 `SM70`
- `vllm/vllm_flash_attn/flash_attn_interface.py`
  - `FA2` 的导入逻辑和算力判断是否仍兼容 `SM70`
- `vllm/model_executor/layers/rotary_embedding/common.py`
  - CUDA rotary 是否又重新探测了顶层 `flash_attn`
- `tests/v1/attention/test_flash_attn_sm70.py`
- `tests/kernels/core/test_apply_rotary_emb.py`

### 升级后重装命令

```bash
conda activate 1cat-vllm-fa2

cd /path/to/1Cat-vLLM
python -m pip install -e . --no-build-isolation

python -m pip install --force-reinstall --no-build-isolation --no-deps -v \
  /path/to/flash-attention-v100
```

### 升级后的最小回归

如果当前环境还没有 `pytest`，先安装：

```bash
python -m pip install pytest
```

```bash
pytest -q tests/v1/attention/test_flash_attn_sm70.py
pytest -q tests/kernels/core/test_apply_rotary_emb.py -k skips_flash_attn_probe_on_cuda
```

然后再做一次最小服务烟测：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
vllm serve /path/to/your-awq-model \
  --tensor-parallel-size 1 \
  --quantization awq \
  --dtype float16 \
  --port 8000 \
  --attention-backend FLASH_ATTN
```

### 一个容易踩坑的旧习惯

升级 `vllm` 之后，不要再沿用“手工去改 `site-packages/vllm/.../flash_attn.py`”这种做法。  
对当前仓库，应该优先把修复留在源码树里，并重新做 `editable install`。

## 升级 lmdeploy 的方法

### 先说判断标准

除非你明确需要下面这类变化，否则不建议频繁升级 `lmdeploy`：

- 新的 `SM70` kernel 覆盖
- TurboMind `gemm` 调优
- V100 AWQ 形状覆盖补齐
- 需要和新的上游 `vllm` 对齐某个底层接口

原因很简单：  
在当前仓库里，`lmdeploy/` 不是“运行时可有可无的文档依赖”，而是 `SM70 AWQ` 构建链路的一部分。`vllm` 在编译时会直接从 vendored `lmdeploy/src/turbomind` 把相关源码并进扩展。

### 为什么它比升级 vllm 更保守

升级 `vllm`，你主要是在处理上层 Python / runtime / backend 接缝。  
升级 `lmdeploy`，你碰到的是：

- TurboMind kernel 文件列表
- `gemm` 调度与 cache 接口
- `SM70` kernel 文件名与路径
- `vllm` 这边的 C++/PyTorch op 桥接

所以：

- `vllm` 可以相对勤升级
- `lmdeploy` 建议按需升级

### 推荐步骤

1. 在新分支或新 worktree 里做
2. 准备一份单独的上游 `lmdeploy` 源码树
3. 把上游源码同步到当前仓库的 vendored `lmdeploy/` 目录
4. 检查根仓库里的集成点是否仍然成立
5. 重新编译并回归

### 同步 vendored lmdeploy 的常见做法

假设你另外有一份上游源码：

```bash
/path/to/lmdeploy-upstream
```

常见同步方式是：

```bash
rsync -a --delete --exclude '.git' \
  /path/to/lmdeploy-upstream/ \
  /path/to/1Cat-vLLM/lmdeploy/
```

这一步有明显覆盖行为，所以一定要先在新分支或新 worktree 里做。

### 同步后重点检查哪些位置

至少检查下面这些位置：

- `CMakeLists.txt`
  - `lmdeploy/src/turbomind/...` 的源文件列表是否仍然存在
- `csrc/ops.h`
- `csrc/torch_bindings.cpp`
- `vllm/model_executor/layers/quantization/...`
- `vllm/model_executor/layers/linear.py`

特别是：

- `lmdeploy/src/turbomind/kernels/gemm/gemm.cu`
- `lmdeploy/src/turbomind/kernels/gemm/dispatch_cache.cu`
- `lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu`
- `lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_16.cu`

如果这些文件改名、拆分、移动，根仓库的 CMake 和 op 桥接很可能要同步改。

### 升级 lmdeploy 后的最小回归

先重新安装当前仓库：

```bash
cd /path/to/1Cat-vLLM
python -m pip install -e . --no-build-isolation
```

然后至少做这三类检查：

1. `AWQ SM70` 相关单测
2. `FLASH_ATTN SM70` 相关单测
3. 一次真实 `Qwen3.5-AWQ` 服务烟测

如果 `lmdeploy` 升级后只是想验证“编译没断”，那还不够。  
至少要确认一份真实 AWQ 模型能加载并生成一次。

## 推荐维护策略

如果你是开源用户，最省成本的维护策略是：

1. 优先维护好 `vllm + flash-attention-v100` 这条线
2. 只有在 `SM70 AWQ` 底层 kernel 确实不够用时，再升级 `lmdeploy`

也就是：

- attention 性能问题，先看 `FLASH_ATTN`
- AWQ kernel / GEMM 覆盖问题，再看 `lmdeploy`

## 常见问题

### 1. 为什么公开 release 文档还在写 `TRITON_ATTN`，而本文推荐 `FLASH_ATTN`

因为两份文档的目标不同：

- [OPEN_SOURCE_SM70_GUIDE.md](OPEN_SOURCE_SM70_GUIDE.md) 讲的是公开发布的保守默认值
- 本文讲的是 V100 用户的推荐默认路径

前者更保守，后者更偏性能与实际使用体验。

### 2. 我已经装了 `vllm-flash-attn`，为什么顶层 `import flash_attn` 还是报错

这通常意味着你环境里残留了另一个顶层 `flash-attn` 包，或者它和当前 `torch` ABI 不兼容。  
对当前 `1Cat-vLLM`，真正关键的是：

- `vllm.vllm_flash_attn.flash_attn_interface` 能工作
- `FA2_AVAILABLE=True`
- `vllm serve` 能实际选中 `FLASH_ATTN`

不要把顶层 `import flash_attn` 当成唯一标准。

### 3. 什么时候应该直接切回 `TRITON_ATTN`

下面几种情况都合理：

- 你只是想先把服务跑起来
- 你在排查一个不确定是不是 attention 引起的问题
- 你升级了 `vllm` 或 `lmdeploy`，想先确认基础链路是否正常

### 4. `FLASH_ATTN_V100` 还有没有意义

有，但更适合作为后备路径。  
对当前仓库，V100 上优先推荐还是 `FLASH_ATTN`。只有当你明确要测试 V100 专用后备实现、或者在排查兼容性问题时，再显式尝试 `FLASH_ATTN_V100`。
