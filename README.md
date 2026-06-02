# Jittor 点云去噪

这是一个基于 Jittor 的 patch 点云去噪项目。当前主线已经简化为：

```text
noisy patch
  -> 在线几何 token 计算
  -> 几何调制的局部 self-attention encoder
  -> displacement decoder
  -> denoised patch
```

模型输入带噪 patch，预测每个点的三维位移：

```text
target = pc_clean - pc_noisy
pc_pred = pc_noisy + displacement
```

当前没有 EdgeConv，没有显式 relative position bias，也没有几何分类头或几何预训练分支。几何信息会在训练和推理 forward 中直接从输入 noisy patch 在线计算。

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

## 数据目录

clean mesh 数据目录：

```text
dataset_clean/
  shapenet/
    <synset_id>/
      <model_id>/
        models/
          model_normalized.obj
```

测试 noisy 点云目录：

```text
test_noisy/
  shapenet/
    <synset_id>/
      <model_id>/
        noisy.npy
```

## 生成 clean 点云缓存

训练不直接从 OBJ mesh 读取，而是先把 mesh 采样成 `clean.npy` 缓存。现在缓存里只需要 clean 点云，不再需要任何几何标签或几何向量文件。

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

## 当前模型结构

训练时会先从 `clean.npy` 读取 clean 点云，然后在线加噪、切 patch：

```text
clean.npy
  -> normalize
  -> add_noise
  -> 随机选 seed point
  -> 在 noisy 点云中取 seed 周围 KNN 1000 个点
  -> patch 坐标减 seed
  -> 输入模型
```

模型内部结构：

```text
patch: (1000, 3)
  -> input projection: (1000, 256)
  -> 从 noisy patch 在线计算每点局部几何统计
  -> GeometryTokenEncoder: (1000, 256)
  -> 4 层 multi-scale local self-attention
  -> fuse: (1000, 256)
  -> decoder
  -> displacement: (1000, 3)
```

在线几何 token 默认使用 `geometry_token_knn: 32`，会从每个点的局部邻域中计算：

```text
curvature
linearity
planarity
scattering
normal variation
local radius mean
local radius std
local radius max
```

这些几何统计经过 MLP 投影成 256 维 token，用于调制 attention 的多尺度权重和 attention temperature。

## 训练

```bash
python run.py --task configs/task/train_vm.yaml --seed 123
```

主训练配置链路：

```text
configs/task/train_vm.yaml
configs/data/train.yaml
configs/transform/vm.yaml
configs/model/vm.yaml
configs/train/vm.yaml
configs/system/vm.yaml
```

当前训练 loss：

```yaml
displacement_loss: 0.9
normalized_surface_loss: 0.1
```

训练输出 checkpoint：

```text
outputs/checkpoints/vm/checkpoint_best.pkl
```

## 推理

确认 `configs/task/predict_vm.yaml` 中的 `load_ckpt` 指向要使用的 checkpoint，例如：

```yaml
load_ckpt: outputs/checkpoints/vm/checkpoint_best.pkl
```

然后运行：

```bash
python scripts/infer.py --seed 123
```

推理输出：

```text
outputs/result/test_noisy/
  shapenet/
    <synset_id>/
      <model_id>/
        denoised.npy
```

推理时会对整云做 patch-based denoise：FPS 选 seed，KNN 切 patch，patch 内预测 displacement，再融合回整云。

## 项目结构

```text
configs/       配置文件
datalist/      train / validate / test 列表
scripts/       缓存、训练、推理、评估脚本
src/data/      数据路径、Dataset、Transform、Augment
src/model/     去噪模型、特征 encoder、decoder
src/system/    训练、验证、推理流程
tools/         分析工具
outputs/       checkpoint、日志、推理结果
```

## 说明

当前仓库不包含数据集、训练权重和大缓存文件。`cache_clean_points/`、`outputs/` 都是运行时产物。
