# Jittor Point Cloud Denoising

这是一个基于 Jittor 的点云去噪项目。模型输入 noisy patch，预测每个点的去噪位移：

```text
target = pc_clean - pc_noisy
pc_pred = pc_noisy + displacement
```

当前主线模型包含：

```text
4-layer global token encoder
+ 4-layer local denoising encoder
+ MLP displacement decoder
```

其中 global token encoder 可以先用“几何伪标签 + SwAV prototype consistency”做自监督预训练，再接入主去噪训练。

## 环境

推荐使用 NVIDIA CUDA 环境：

```bash
conda env create -f environment.yml
conda activate jittor
python -m pip install -r requirements.txt
```

如果手动创建环境：

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
python -m pip install -r requirements.txt
```

ROCm/HIP/DCU 后端可能触发 Jittor fused operator 编译失败。训练建议使用 A800、4090、3090 等 NVIDIA CUDA 机器。

## 数据目录

clean mesh 目录：

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

如果原始数据放在隔壁项目，例如：

```text
E:\Code\competition2_EdgeConv\dataset_clean
```

下面命令里的 `--input_dataset_dir` 改成对应路径即可。

## Step 1: 生成主训练 clean 缓存

主去噪训练读取的是 clean 点云缓存，不直接从 mesh 训练。先从 OBJ 采样生成 `clean.npy`：

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

命令中默认加入了 `--overwrite`，会重建已有缓存；如果想复用已有缓存，可以去掉 `--overwrite`。

## Step 2: 生成 SSL 几何缓存

global token 自监督预训练使用单独的几何缓存 `geometry_ssl_cache`。它会从真实 OBJ 采样 clean patch，计算 PCA、局部曲率、normal variation 等几何特征，再用 k-means 生成几何伪标签。

推荐第一版配置：

```bash
python scripts/build_geometry_ssl_cache.py \
  --input_dataset_dir dataset_clean \
  --datalist datalist/train.txt \
  --output_dir geometry_ssl_cache \
  --num_shapes 5000 \
  --patches_per_shape 1 \
  --num_geom_classes 12 \
  --seed 123 \
  --overwrite
```

Windows 示例：

```powershell
C:\Users\Lenovo\anaconda3\envs\jittor\python.exe scripts\build_geometry_ssl_cache.py `
  --input_dataset_dir E:\Code\competition2_EdgeConv\dataset_clean `
  --datalist datalist\train.txt `
  --output_dir geometry_ssl_cache `
  --num_shapes 5000 `
  --patches_per_shape 1 `
  --num_geom_classes 12 `
  --seed 123 `
  --overwrite
```

输出结构：

```text
geometry_ssl_cache/
  samples/
    00000000.npz
    00000001.npz
    ...
  train.txt
  patches_clean.npy
  labels.npy
  geom_features.npy
  kmeans.json
  sources.json
```

其中：

- `samples/*.npz`：单个 clean patch 和几何伪标签，供 dataloader 训练使用。
- `labels.npy`：k-means 伪标签。
- `geom_features.npy`：几何描述符。
- `kmeans.json`：聚类中心、特征名、每类数量。

## Step 3: 预训练 global token encoder

生成 `geometry_ssl_cache` 后，启动 global token SSL 预训练：

```bash
python run.py --task configs/task/train_global_token_ssl.yaml --seed 123
```

训练配置：

```text
configs/model/global_token_ssl.yaml
configs/data/global_token_ssl.yaml
configs/train/global_token_ssl.yaml
configs/task/train_global_token_ssl.yaml
```

默认设置：

```text
epochs = 30
num_geom_classes = 12
num_prototypes = 24
swav_weight = 0.2
batch_size = 16
lr = 1e-4
```

每个 clean patch 会生成两个增强 view：

```text
view_a = dropout/resample + Laplace noise + small jitter
view_b = another dropout/resample + Laplace noise + small jitter
```

预训练 loss：

```text
loss = geom_loss + 0.2 * swav_loss
```

输出 checkpoint：

```text
outputs/checkpoints/global_token_ssl/checkpoint_best.pkl
outputs/checkpoints/global_token_ssl/checkpoint_<epoch>.pkl
```

## Step 4: 使用 SSL 权重开始主去噪训练

SSL 预训练完成后，使用下面入口开始主训练：

```bash
python run.py --task configs/task/train_vm_ssl.yaml --seed 123
```

这个入口会：

```text
1. 创建正常 VelocityModule 去噪模型
2. 从 outputs/checkpoints/global_token_ssl/checkpoint_best.pkl 加载：
   - input_proj_1
   - input_proj_2
   - global_token_generator
3. 前 8 个 epoch 冻结 global token encoder
4. 第 9 个 epoch 起解冻，并使用 0.2x global gradient scale
5. 按原 displacement + surface loss 训练主去噪模型
```

对应配置：

```text
configs/system/vm_ssl.yaml
configs/task/train_vm_ssl.yaml
```

关键参数：

```yaml
ssl_pretrained_ckpt: outputs/checkpoints/global_token_ssl/checkpoint_best.pkl
freeze_global_epochs: 8
global_lr_scale: 0.2
```

主训练仍然读取 `cache_clean_points`：

```text
configs/data/train.yaml
input_dataset_dir: ./cache_clean_points
loader: clean_npy
data_name: clean.npy
```

输出 checkpoint：

```text
outputs/checkpoints/vm_ssl/checkpoint_best.pkl
```

## 不使用 SSL 的原始主训练

如果不想使用 global token 预训练，仍然可以走原来的训练入口：

```bash
python run.py --task configs/task/train_vm.yaml --seed 123
```

或者：

```bash
python scripts/train.py --seed 123
```

默认 checkpoint 输出：

```text
outputs/checkpoints/vm/checkpoint_best.pkl
```

## 推理

确认 `configs/task/predict_vm.yaml` 中的 `load_ckpt` 指向要使用的 checkpoint，例如：

```yaml
load_ckpt: outputs/checkpoints/vm_ssl/checkpoint_best.pkl
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

打包提交：

```bash
cd outputs/result/test_noisy
zip -r ../../../result.zip shapenet/
```

## 推荐完整流程

```bash
# 1. 生成主训练 clean 缓存
python scripts/cache_clean_points.py \
  --input_dataset_dir dataset_clean \
  --output_dir cache_clean_points \
  --datalist datalist/train.txt datalist/validate.txt \
  --workers 8 \
  --seed 123 \
  --overwrite

# 2. 生成 SSL 几何缓存
python scripts/build_geometry_ssl_cache.py \
  --input_dataset_dir dataset_clean \
  --datalist datalist/train.txt \
  --output_dir geometry_ssl_cache \
  --num_shapes 5000 \
  --patches_per_shape 1 \
  --num_geom_classes 12 \
  --seed 123 \
  --overwrite

# 3. 预训练 global token encoder
python run.py --task configs/task/train_global_token_ssl.yaml --seed 123

# 4. 加载 SSL 权重进行主去噪训练
python run.py --task configs/task/train_vm_ssl.yaml --seed 123

# 5. 推理
python scripts/infer.py --seed 123
```

## 诊断 global token

可以使用：

```bash
python tools/analyze_global_token.py \
  --mesh-root dataset_clean \
  --datalist datalist/validate.txt \
  --checkpoint outputs/checkpoints/vm_ssl/checkpoint_best.pkl \
  --num-shapes 16 \
  --patches-per-shape 2 \
  --num-patches 32 \
  --seed 123 \
  --output analysis_outputs/global_token_vm_ssl_real.json
```

它会检查：

- global token 是否按真实 patch 几何特征聚类。
- denoising encoder 是否关注 global token。
- local attention 中 global token 权重是否高于 uniform baseline。

## 项目结构

```text
configs/       配置文件
datalist/      train / validate / test 列表
scripts/       缓存、训练、推理、评估脚本
src/           核心代码
tools/         诊断工具
outputs/       checkpoint、日志、推理结果
```

## 说明

仓库不包含数据集、训练权重和大缓存文件。`cache_clean_points/`、`geometry_ssl_cache/`、`outputs/` 都是运行时产物。
