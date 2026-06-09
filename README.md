# Jittor 点云去噪

这是一个基于 Jittor 的 patch 点云去噪项目。仓库里目前保留两条模型流程：

```text
1. VM 主线：global token SSL + EDM 条件化 self-attention 去噪模型
2. EdgeConv Baseline：迁入官方 starter code 风格的 EdgeConv 基线
```

EdgeConv baseline 仍预测每个点的三维位移：

```text
target = pc_clean - pc_noisy
pc_pred = pc_noisy + displacement
```

VM 主线当前以 global token 预训练模型为基线，并切到 EDM 路线 C：模型输入 noisy patch 和噪声强度 `sigma`，经 EDM preconditioning 后预测 clean estimate；`sigma` 通过 FiLM 注入每层 local self-attention block，但暂不注入 global token generator。global token 仍由原始 noisy patch 生成，不使用 `c_in` 缩放输入。当前 VM 不使用显式 relative position bias。

README 下面把两条流程分开说明，避免把 VM 和 EdgeConv baseline 的数据、配置、训练和推理命令混在一起。

## 环境

推荐使用 CUDA 环境：

```bash
conda env create -f environment.yml
conda activate jittor
python -m pip install -r requirements.txt
```

也可以手动创建环境：

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
python -m pip install -r requirements.txt
```

Windows 示例命令中的 Python 解释器可以替换成自己的环境路径，例如：

```powershell
C:\Users\Lenovo\anaconda3\envs\jittor\python.exe run.py --task configs\task\train_vm_ssl.yaml --seed 123
```

## 数据目录

训练用 clean mesh 目录：

```text
dataset_clean/
  shapenet/
    <synset_id>/
      <model_id>/
        models/
          model_normalized.obj
```

测试用 noisy 点云目录：

```text
test_noisy/
  shapenet/
    <synset_id>/
      <model_id>/
        noisy.npy
```

## 公共 clean 点云缓存

VM 训练前需要先把 mesh 采样成 `clean.npy` 缓存。缓存里只需要 clean 点云，不需要几何标签或几何向量文件。

```bash
python scripts/cache_clean_points.py \
  --input_dataset_dir dataset_clean \
  --output_dir cache_clean_points \
  --datalist datalist/train.txt datalist/validate.txt \
  --workers 8 \
  --seed 123 \
  --overwrite
```

Windows 示例：

```powershell
C:\Users\Lenovo\anaconda3\envs\jittor\python.exe scripts\cache_clean_points.py `
  --input_dataset_dir E:\Code\competition2_EdgeConv\dataset_clean `
  --output_dir cache_clean_points `
  --datalist datalist\train.txt datalist\validate.txt `
  --workers 8 `
  --seed 123 `
  --overwrite
```

输出结构：

```text
cache_clean_points/
  shapenet/
    <synset_id>/
      <model_id>/
        clean.npy
```

注意：EdgeConv baseline 不走这个缓存流程，它会按官方 baseline 的方式直接从 OBJ mesh 在线采样。

## VM 主线流程

VM 是当前自己设计的主线模型，对应配置名为 `vm`。它使用 4 层 multi-scale local self-attention，每层 local attention 都额外读取一个由 SSL 预训练模块生成的 global token。当前主线采用 EDM preconditioning，并把噪声强度 `sigma` 通过零初始化 FiLM 注入每层 attention/FFN；global token generator 暂不接收 `sigma`，也不使用 `c_in` 缩放后的输入。

### VM 数据流

训练时先读取 `cache_clean_points/` 里的 `clean.npy`，然后在线加噪、切 patch：

```text
clean.npy
  -> normalize
  -> add_noise: Gaussian 噪声，train 使用 [0, 0.025] 截断 log-normal sigma
  -> 随机选 seed point
  -> 在 noisy 点云中取 seed 周围 KNN 1000 个点
  -> 保存本次加噪强度 sigma
  -> patch 坐标减 noisy seed
  -> 输入 VM
```

推理时对整云做 patch-based denoise：

```text
noisy.npy
  -> FPS 选 seed
  -> KNN 切 patch
  -> EDM 少步 refinement 预测 clean estimate
  -> 融合回整云
  -> denoised.npy
```

### VM 模型结构

```text
patch: (1000, 3)
  -> 原始 noisy patch 生成 global token
  -> c_in * noisy patch 进入去噪 SA 主干
  -> 4 层 multi-scale local self-attention + sigma FiLM
       attention_knn: [8, 16, 32]
       第 1 层使用坐标 KNN，后续层使用当前特征动态 KNN
  -> decoder: 256 -> 128 -> 64 -> 3
  -> EDM clean estimate: (1000, 3)
```

当前 EDM 形式：

```text
D(x, sigma) = c_skip * x + c_out * F(c_in * x, c_noise)
c_noise = log(sigma) / 4

sigma_data: 训练开始前从 train patch 的 clean 坐标标准差自动估计
edm_sampler: heun
edm_inference_sigmas: [0.020, 0.010, 0.005, 0.0]
```

当前 VM 主线没有 EdgeConv，没有显式 relative position bias，也没有几何分类头。global token generator 使用 SSL 预训练权重初始化，主训练前 8 个 epoch 冻结 `input_proj_1`、`input_proj_2` 和 `global_token_generator`；从 epoch 8 开始解冻，并以 `global_lr_scale: 0.2` 缩小这部分梯度。

主要配置：

```text
configs/task/train_vm.yaml
configs/task/train_vm_ssl.yaml
configs/task/predict_vm.yaml
configs/data/train.yaml
configs/data/predict.yaml
configs/transform/vm.yaml
configs/model/vm.yaml
configs/train/vm.yaml
configs/system/vm.yaml
configs/system/vm_ssl.yaml
```

当前训练 loss：

```yaml
displacement_loss: 1.0  # EDM 模式下实际为 weighted clean-estimate MSE
```

### VM 训练

训练前需要确认 global token SSL 预训练权重存在：

```text
outputs/checkpoints/global_token_ssl/checkpoint_best.pkl
```

如果权重在别的位置，修改 `configs/system/vm_ssl.yaml` 里的 `ssl_pretrained_ckpt`。

```bash
python run.py --task configs/task/train_vm_ssl.yaml --seed 123
```

Windows 示例：

```powershell
C:\Users\Lenovo\anaconda3\envs\jittor\python.exe run.py `
  --task configs\task\train_vm_ssl.yaml `
  --seed 123
```

训练输出：

```text
outputs/checkpoints/vm_ssl/checkpoint_best.pkl
outputs/vm/runs/
```

### VM 推理

先确认 `configs/task/predict_vm.yaml` 里的 `load_ckpt` 指向要使用的 checkpoint：

```yaml
load_ckpt: outputs/checkpoints/vm_ssl/checkpoint_best.pkl
```

然后运行：

```bash
python run.py --task configs/task/predict_vm.yaml --seed 123
```

也可以使用默认加载 VM 配置的脚本：

```bash
python scripts/infer.py --seed 123
```

推理输出：

```text
outputs/vm/result/test_noisy/
  shapenet/
    <synset_id>/
      <model_id>/
        denoised.npy
outputs/vm/result/result.zip
```

## EdgeConv Baseline 流程

EdgeConv baseline 对应配置名为 `edgeconv_baseline`，用于复现/对照官方 starter code 风格的基线。它不使用 `cache_clean_points/`，而是直接从 OBJ mesh 在线采样 clean 点云。

### EdgeConv 数据流

训练数据来源：

```text
dataset_clean/
  shapenet/
    <synset_id>/
      <model_id>/
        models/model_normalized.obj
```

训练 transform：

```text
OBJ mesh
  -> sample: 表面采样 32768 点，并混入 1024 个原始顶点
  -> normalize_pc
  -> add_noise: Laplace 噪声，std 从 [0.005, 0.020] 随机采样
  -> linear: 随机旋转/缩放配置保留官方设置
  -> patch: noisy KNN 取 1000 点，并生成 pc_mix
```

训练目标保持官方 baseline 写法：

```text
input = pc_mix
target = pc_clean - pc_noisy
loss = MSE(pred_dir, target) / dsm_sigma
```

推理同样使用 patch-based denoise 和 hard patch fusion。

### EdgeConv 模型结构

```text
patch pc_mix: (1000, 3)
  -> DynamicEdgeConv(3 -> 32), KNN k=16
  -> DynamicEdgeConv(32 -> 64), KNN k=16
  -> concat(x1, x2): (1000, 96)
  -> DynamicEdgeConv(96 -> 256), KNN k=16
  -> decoder: 256 -> 256 -> 64 -> 3
  -> displacement: (1000, 3)
```

主要配置：

```text
configs/task/train_edgeconv_baseline.yaml
configs/task/predict_edgeconv_baseline.yaml
configs/data/edgeconv_baseline.yaml
configs/transform/edgeconv_baseline.yaml
configs/model/edgeconv_baseline.yaml
configs/train/edgeconv_baseline.yaml
configs/system/edgeconv_baseline.yaml
```

### EdgeConv 训练

```bash
python run.py --task configs/task/train_edgeconv_baseline.yaml --seed 123
```

Windows 示例：

```powershell
C:\Users\Lenovo\anaconda3\envs\jittor\python.exe run.py `
  --task configs\task\train_edgeconv_baseline.yaml `
  --seed 123
```

训练输出：

```text
outputs/EdgeConv/checkpoints/checkpoint_best.pkl
outputs/EdgeConv/runs/
```

### EdgeConv 推理

先确认 `configs/task/predict_edgeconv_baseline.yaml` 里的 `load_ckpt` 指向要使用的 checkpoint：

```yaml
load_ckpt: outputs/EdgeConv/checkpoints/checkpoint_best.pkl
```

然后运行：

```bash
python run.py --task configs/task/predict_edgeconv_baseline.yaml --seed 123
```

推理输出：

```text
outputs/EdgeConv/result/test_noisy/
  shapenet/
    <synset_id>/
      <model_id>/
        denoised.npy
outputs/EdgeConv/result/result.zip
```

## 项目结构

```text
configs/       配置文件
datalist/      train / validate / test 列表
scripts/       缓存、训练、推理、评估脚本
src/data/      数据路径、Dataset、Transform、Augment
src/model/     VM、EdgeConv 模型和公共特征模块
src/system/    训练、验证、推理流程
tools/         分析工具
outputs/       checkpoint、日志、推理结果，按模型分为 vm/、EdgeConv/
```

## 说明

当前仓库不包含数据集、训练权重和大缓存文件。`cache_clean_points/`、`outputs/` 都是运行时产物。
