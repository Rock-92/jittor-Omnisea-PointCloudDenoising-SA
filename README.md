# Jittor 点云去噪

这是一个基于 Jittor 的 patch 点云去噪项目。仓库里目前保留两条模型流程：

```text
1. VM 主线：3 层 SA + 在线几何 token 调制的 self-attention 去噪模型
2. EdgeConv Baseline：迁入官方 starter code 风格的 EdgeConv 基线
```

三个模型最终都预测每个点的三维位移：

```text
target = pc_clean - pc_noisy
pc_pred = pc_noisy + displacement
```

其中 VM 主线当前退回原版单步 displacement 训练：模型输入 noisy patch，直接预测 `pc_clean - pc_noisy` 位移；推理时单步输出 denoised patch。EdgeConv baseline 也保留 displacement 预测表述。

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
C:\Users\Lenovo\anaconda3\envs\jittor\python.exe run.py --task configs\task\train_vm.yaml --seed 123
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

VM 是当前自己设计的主线模型，对应配置名为 `vm`。它使用 3 层 multi-scale local self-attention，并采用噪声强度条件调制每层 attention/FFN；几何 token 从当前 noisy patch 在线计算。

### VM 数据流

训练时先读取 `cache_clean_points/` 里的 `clean.npy`，然后在线加噪、切 patch：

```text
clean.npy
  -> normalize
  -> add_noise
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
  -> patch 内预测 displacement
  -> 融合回整云
  -> denoised.npy
```

### VM 模型结构

```text
patch: (1000, 3)
  -> input projection: 3 -> 128 -> 256
  -> 3 层 multi-scale local self-attention
       attention_knn: [8, 16, 32]
  -> displacement decoder: 256 -> 128 -> 64 -> 3
  -> displacement: (1000, 3)
```

几何 token 调制和 sigma 条件模块仍保留在代码里，但当前 VM 配置关闭：

```text
use_geometry_tokens: false
use_bridge_condition: false
denoise_num_steps: 1
```

当前 VM 主线没有 EdgeConv，没有显式 relative position bias，也没有几何分类头或几何预训练分支。

主要配置：

```text
configs/task/train_vm.yaml
configs/task/predict_vm.yaml
configs/data/train.yaml
configs/data/predict.yaml
configs/transform/vm.yaml
configs/model/vm.yaml
configs/train/vm.yaml
configs/system/vm.yaml
```

当前训练 loss：

```yaml
displacement_loss: 0.9  # 预测 pc_clean - pc_noisy
normalized_surface_loss: 0.1
```

### VM 训练

```bash
python run.py --task configs/task/train_vm.yaml --seed 123
```

Windows 示例：

```powershell
C:\Users\Lenovo\anaconda3\envs\jittor\python.exe run.py `
  --task configs\task\train_vm.yaml `
  --seed 123
```

训练输出：

```text
outputs/vm/checkpoints/checkpoint_best.pkl
outputs/vm/runs/
```

### VM 推理

先确认 `configs/task/predict_vm.yaml` 里的 `load_ckpt` 指向要使用的 checkpoint：

```yaml
load_ckpt: outputs/vm/checkpoints/checkpoint_best.pkl
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
