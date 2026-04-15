# v0.19.0 默认打包版本固定设计

日期：2026-04-15

## 1. 背景与目标

当前 `feature/vllm-0190-upstream-split` 的源码基线已经升级到 `vLLM 0.19.0`，但默认打包版本仍由 `setuptools_scm` 和本地构建环境共同决定，通常会生成带有 `dev`、git sha 或 CUDA 后缀的版本号。

这不符合本次对外发布目标：默认构建出来的包版本应稳定显示为 `0.19.0`。

本次设计目标只有一个：

- 在不修改源码包名、模块名、CLI 名称和推理路径的前提下，让仓库默认构建出的包版本固定为 `0.19.0`。

## 2. 非目标

本次设计明确不做以下事情：

- 不修改 Python 包名 `vllm`
- 不修改 CLI 命令 `vllm`
- 不修改 GitHub 仓库发布流程以外的运行时行为
- 不调整 `SM70 / AWQ / MiniMax / compressed-tensors` 相关推理逻辑
- 不把版本策略扩展为新的多变体命名体系

## 3. 当前版本生成链

当前版本由 `setup.py` 的 `get_vllm_version()` 负责生成，流程是：

1. 如果设置了 `VLLM_VERSION_OVERRIDE`，则把它写入 `SETUPTOOLS_SCM_PRETEND_VERSION`
2. 否则调用 `setuptools_scm.get_version(write_to="vllm/_version.py")`
3. 对 CUDA / ROCm / precompiled / empty 等构建场景继续追加后缀
4. `setup()` 最终使用该结果作为对外包版本

这个实现对 upstream 持续开发友好，但不适合本仓库当前“固定发布为 0.19.0”的要求。

## 4. 候选方案

### 方案 A：只在发布命令中设置 `VLLM_VERSION_OVERRIDE=0.19.0`

优点：

- 改动最小
- 不影响当前版本生成逻辑

缺点：

- 默认构建行为仍不稳定
- 每次发布都依赖人工记忆
- 不满足“仓库本身默认打包版本就是 0.19.0”的目标

### 方案 B：修改默认版本生成逻辑，直接固定为 `0.19.0`

优点：

- 仓库默认构建结果稳定
- 不依赖发布时额外环境变量
- 更符合对外发布仓库的预期

缺点：

- 需要调整 `setup.py` 的默认版本逻辑
- 会弱化当前 `setuptools_scm` 的动态版本语义

### 方案 C：保留动态版本，只用 Git tag 标记 `v0.19.0`

优点：

- 改动最少

缺点：

- 包元数据仍不是 `0.19.0`
- 不满足本次要求

## 5. 推荐方案

推荐采用方案 B。

具体策略如下：

1. 默认构建版本直接固定为 `0.19.0`
2. 仍保留 `VLLM_VERSION_OVERRIDE` 作为更高优先级的显式覆盖入口，方便后续特殊发布
3. 默认情况下不再根据 CUDA / ROCm / precompiled 场景附加版本后缀
4. 继续写出 `vllm/_version.py`，确保运行时 `vllm.__version__` 与包元数据一致

## 6. 详细设计

### 6.1 修改点

只改动版本生成入口，不扩散到运行时其他模块。

目标文件：

- `setup.py`

预期调整：

1. 将 `get_vllm_version()` 的默认返回值改为固定字符串 `0.19.0`
2. 保留 `VLLM_VERSION_OVERRIDE` 逻辑，使其可以显式覆盖默认值
3. 保留写出 `vllm/_version.py` 的行为，避免运行时版本与包元数据不一致
4. 删除默认路径上对 CUDA/ROCm/precompiled/empty 后缀的追加

### 6.2 兼容性要求

修改后必须保证：

- `import vllm`
- `from vllm.version import __version__`
- CLI 中 `--version`
- 任何读取 wheel / dist metadata 的逻辑

都能看到一致的 `0.19.0`。

### 6.3 风险控制

主要风险是默认版本固定后，某些调试场景失去“从版本号区分构建变体”的能力。

控制方式：

- 不改包名，不改 CLI，不改运行时逻辑
- 保留 `VLLM_VERSION_OVERRIDE`
- 通过验证命令确认运行时版本和打包元数据一致

## 7. 验证口径

最低验证包含：

1. 本地构建元数据检查：
   - `python setup.py --version`
2. 运行时版本检查：
   - `python -c "import vllm; print(vllm.__version__)"`
3. CLI 版本检查：
   - `python -m vllm.entrypoints.cli.main --version`

三者都必须返回 `0.19.0`。

## 8. 实施边界

这次改动应作为一次独立、小范围的发布整理提交处理，不应与新的推理修复、模型支持或构建系统重构混在一起。
