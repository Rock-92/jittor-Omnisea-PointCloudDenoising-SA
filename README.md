# Jittor Point Cloud Denoising

这是一个基于 Jittor 的点云去噪项目。模型输入 noisy patch，预测每个点的去噪位移：

```text
target = pc_clean - pc_noisy
pc_pred = pc_noisy + displacement
```

当前主线模型包含：

```text
point-wise EdgeConv geometry conditioner
+ 4-layer token-conditioned local denoising encoder
+ MLP displacement decoder
```

当前主线不再使用 16 个 local token 或 SSL 预训练。模型先用 EdgeConv 为每个点提取一个 256 维 point-wise local geometry feature，再用这个 feature 调制后续 local attention 的 `scale_gate / temperature / rel_gate`。EdgeConv 分支还带一个辅助 displacement head，用同一个 `pc_clean - pc_noisy` 目标约束它不要跑偏。

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

## Step 2: 训练 EdgeConv 调制去噪模型

生成 `cache_clean_points` 后，直接启动主训练：

```bash
python run.py --task configs/task/train_vm_ssl.yaml --seed 123
```

这个入口会：

```text
1. 创建正常 VelocityModule 去噪模型
2. 用 input_proj 得到每点初始特征
3. EdgeConvConditioner 对每个点的 KNN 邻域提取 point-wise local feature
4. 每个点自己的 EdgeConv feature 调制 local attention
5. 主 decoder 输出 displacement
6. EdgeConv auxiliary head 也输出辅助 displacement
7. 按 displacement + edge_aux_displacement + surface loss 端到端训练
```

对应配置：

```text
configs/system/vm_edgeconv.yaml
configs/task/train_vm_ssl.yaml
```

关键参数：

```yaml
edgeconv_knn: 24
edgeconv_blocks: 2
edgeconv_hidden_dim: 256
edge_aux_hidden_dims: [128, 64]
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
outputs/checkpoints/vm_edgeconv/checkpoint_best.pkl
```

当前 loss 权重：

```yaml
displacement_loss: 0.9
edge_aux_displacement_loss: 0.2
normalized_surface_loss: 0.1
```

## 推理

确认 `configs/task/predict_vm.yaml` 中的 `load_ckpt` 指向要使用的 checkpoint，例如：

```yaml
load_ckpt: outputs/checkpoints/vm_edgeconv/checkpoint_best.pkl
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

# 2. 训练 EdgeConv point-wise condition 去噪模型
python run.py --task configs/task/train_vm_ssl.yaml --seed 123

# 3. 推理
python scripts/infer.py --seed 123
```

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
