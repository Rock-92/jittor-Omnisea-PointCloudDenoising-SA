# Jittor Point Cloud Denoising

这是一个基于 Jittor 的点云去噪项目。主模型输入 noisy patch，不直接预测 clean point，而是预测每个点的去噪位移：

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

其中 global token encoder 可以先用轻量 Point-MAE 风格任务做自监督预训练，再接入主去噪训练。

## Environment

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

本地 Windows 常用 Python：

```text
C:\Users\Lenovo\anaconda3\envs\jittor\python.exe
```

## Data Layout

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

如果原始数据在隔壁项目，例如：

```text
E:\Code\competition2_EdgeConv\dataset_clean
E:\Code\competition2_EdgeConv\test_noisy
```

把下面命令里的 `--input_dataset_dir` 或配置里的 `input_dataset_dir` 改成对应路径即可。

## Step 1: Build Clean Cache

主去噪训练读取 clean 点云缓存，不直接从 mesh 训练。先从 OBJ 采样生成 `clean.npy`：

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

## Step 2: Build SSL Patch Cache

global token 自监督预训练使用单独缓存 `geometry_ssl_cache`。当前 MAE 预训练只需要 `samples/*.npz` 里的 clean patch；脚本仍会额外保存几何特征、k-means 标签和统计文件，方便后续诊断或对比实验。

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

## Step 3: Pretrain Global Token Encoder

当前 SSL 预训练是轻量 Point-MAE 风格：

```text
clean patch: 1000 points
-> 随机 visible 400 点，加入轻微 noise/jitter
-> masked 600 点作为重建目标
-> visible points 输入 global token encoder
-> 输出 global token: (B, 256)
-> mae_decoder 重建 masked point set
-> Chamfer L2 loss
```

启动预训练：

```bash
python run.py --task configs/task/train_global_token_ssl.yaml --seed 123
```

默认关键配置：

```text
epochs = 30
batch_size = 16
lr = 1e-4
patch_size = 1000
mae_visible_points = 400
mae_mask_points = 600
mae_decoder_hidden_dim = 512
mae_noise_std = 0.005
mae_jitter_std = 0.001
```

loss：

```text
loss = mae_loss
```

输出 checkpoint：

```text
outputs/checkpoints/global_token_ssl/checkpoint_best.pkl
outputs/checkpoints/global_token_ssl/checkpoint_<epoch>.pkl
```

注意：`checkpoint_best.pkl` 按最低 train loss 保存，因为 SSL 当前没有验证集。

## Step 4: Train VM With SSL Weights

SSL 预训练完成后，使用下面入口开始主去噪训练：

```bash
python run.py --task configs/task/train_vm_ssl.yaml --seed 123
```

这个入口会：

```text
1. 创建正常 VelocityModule 去噪模型
2. 从 outputs/checkpoints/global_token_ssl/checkpoint_best.pkl 加载：
   - encoder.input_proj_1
   - encoder.input_proj_2
   - encoder.global_token_generator
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

## Train Without SSL

如果不使用 global token 预训练，使用原始主训练入口：

```bash
python run.py --task configs/task/train_vm.yaml --seed 123
```

默认 checkpoint：

```text
outputs/checkpoints/vm/checkpoint_best.pkl
```

## Inference

确认 `configs/task/predict_vm.yaml` 中的 `load_ckpt` 指向要使用的 checkpoint，例如：

```yaml
load_ckpt: outputs/checkpoints/vm_ssl/checkpoint_best.pkl
```

然后运行：

```bash
python scripts/infer.py --seed 123
```

如果测试数据在隔壁项目，修改：

```yaml
# configs/data/predict.yaml
input_dataset_dir: ../competition2_EdgeConv/test_noisy
```

## Recommended Flow

```bash
# 1. 生成主训练 clean 缓存
python scripts/cache_clean_points.py \
  --input_dataset_dir dataset_clean \
  --output_dir cache_clean_points \
  --datalist datalist/train.txt datalist/validate.txt \
  --workers 8 \
  --seed 123 \
  --overwrite

# 2. 生成 SSL patch 缓存
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

## Diagnose Global Token

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

## Project Layout

```text
configs/       配置文件
datalist/      train / validate / test 列表
scripts/       缓存、训练、推理、评估脚本
src/           核心代码
tools/         诊断工具
outputs/       checkpoint、日志、推理结果
```

## Notes

仓库不包含数据集、训练权重和大缓存文件。`cache_clean_points/`、`geometry_ssl_cache/`、`outputs/` 都是运行时产物。
