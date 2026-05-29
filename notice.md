# A800 环境配置记录

本文记录 A800 云服务器上成功创建项目环境的指令流水线。这里只记录环境创建、Jittor 初始化和 CUDA/cuDNN 相关处理，不包含数据缓存、训练或推理步骤。

## 服务器配置

```text
GPU: A800 x 1
显存: 80GB
CPU 型号: 2 x Intel 8358P 32C
平台显示 CPU: 14 核心
内存: 245GB
资源组: 64 组
资源组名称: nva800normal
镜像: jupyterlab-pytorch:2.2.0-py3.10-cuda12.1-ubuntu22.04-devel
```

## 成功指令流水线

以下命令在项目根目录执行。

```bash
conda env create -f environment.yml
```

作用：根据项目里的 `environment.yml` 创建 conda 环境。该环境名称为 `jittor`。

```bash
source activate jittor
```

作用：激活刚创建的 `jittor` 环境。本服务器上可用的是 `source activate jittor`，而不是 `conda activate jittor`。

```bash
conda install -c conda-forge libgomp -y
```

作用：安装 GNU OpenMP 运行库。Jittor 首次导入时会下载并编译 MKL-DNN/DNNL 相关依赖，这些依赖需要 `libgomp.so.1`。缺少它时会出现 `libgomp.so.1 not found` 或 `GOMP_*` 未定义符号错误。

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

作用：把当前 conda 环境的 `lib` 目录加入动态链接库搜索路径，使 Jittor 能找到 `libgomp.so.1` 和环境内的其他共享库。

```bash
python3.9 -m jittor_utils.install_cuda
```

作用：让 Jittor 自动安装它兼容的 CUDA/cuDNN 运行时到 Jittor 缓存目录。本服务器上 Jittor 检测到 CUDA driver version 为 `12.9`，并安装了：

```text
~/.cache/jittor/jtcuda/cuda11.2_cudnn8_linux
```

这一步解决了 Jittor 导入时的 cuDNN 问题：

```text
CUDA found but cudnn is not loaded
Develop version of CUDNN not found
```

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

作用：在安装 Jittor CUDA/cuDNN 后再次导出库路径，确保 conda 环境库目录仍在动态链接搜索路径前面。

```bash
python -c "import jittor as jt; print(jt.__version__)"
```

作用：验证 Jittor 能正常导入。该服务器上成功输出：

```text
1.3.8.5
```

## 最终可复制命令

同配置新服务器可以按以下顺序执行：

```bash
conda env create -f environment.yml
source activate jittor
conda install -c conda-forge libgomp -y
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python3.9 -m jittor_utils.install_cuda
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python -c "import jittor as jt; print(jt.__version__)"
```

## 备注

- Jittor 第一次导入时可能会下载依赖并编译算子，耗时几分钟属于正常现象。
- 本服务器使用 `source activate jittor` 激活环境。
- `libgomp` 是 Jittor 编译 MKL-DNN/DNNL 依赖时需要的 OpenMP 运行库。
- `python3.9 -m jittor_utils.install_cuda` 会安装 Jittor 自管理的 CUDA/cuDNN，避免依赖系统中缺失的 cuDNN 开发文件。
