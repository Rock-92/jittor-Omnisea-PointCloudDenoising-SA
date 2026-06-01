# Jittor Point Cloud Denoising

这是一个基于 Jittor 的点云去噪项目。模型输入 noisy patch，预测每个点的去噪位移：

```text
target = pc_clean - pc_noisy
pc_pred = pc_noisy + displacement
```

当前主线模型：

```text
point-wise EdgeConv geometry teacher
+ geometry-aware gate matching
+ 4-layer token-conditioned local denoising encoder
+ MLP displacement decoder
```

模型先用 `EdgeConvConditioner` 为每个点提取 256 维 point-wise local geometry feature。这个 feature 一方面作为几何 teacher 通过分类头学习局部曲率、尖锐法向变化和转角等几何伪标签，另一方面用于调制后续 local attention 的 `scale_gate / temperature / rel_gate`。主训练中还加入 gate matching loss，要求几何分类头认为相近的点具有相近 gate，几何分类头认为较远的点具有较远 gate。

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

如果原始数据在隔壁项目，例如：

```text
E:\Code\competition2_EdgeConv\dataset_clean
```

下面命令里的 `--input_dataset_dir` 改成对应路径即可。

## Step 1: 生成 clean 缓存

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

## Step 2: 预训练 EdgeConv 几何 teacher

先预训练 EdgeConv 几何 teacher，让 EdgeConv condition feature 和几何分类头能够输出明确的局部几何信息。几何伪标签由 clean patch 在线估计，主要来自局部曲率和法向转角分桶。

```bash
python run.py --task configs/task/train_edge_geom_pretrain.yaml --seed 123
```

输出 checkpoint：

```text
outputs/checkpoints/vm_edgegeom_pretrain/checkpoint_best.pkl
```

对应配置：

```text
configs/task/train_edge_geom_pretrain.yaml
configs/model/vm_edgegeom_pretrain.yaml
configs/train/edge_geom_pretrain.yaml
configs/system/vm_edgegeom_pretrain.yaml
```

预训练阶段只优化：

```yaml
edge_geom_cls_loss: 1.0
```

## Step 3: 主去噪训练

预训练完成后，启动带 gate matching 的主训练：

```bash
python run.py --task configs/task/train_vm_edgegeom.yaml --seed 123
```

这个入口会加载：

```text
outputs/checkpoints/vm_edgegeom_pretrain/checkpoint_best.pkl
```

主训练流程：

```text
1. 创建 VelocityModule 去噪模型
2. input_proj 得到每点初始特征
3. EdgeConvConditioner 提取每点 point-wise local geometry feature
4. edge_geom_head 预测局部几何类别
5. scale_gate / temperature / rel_gate 根据 EdgeConv feature 调制 local attention
6. edge_gate_match_loss 约束 gate 的距离结构匹配几何分类头的距离结构
7. decoder 输出 displacement
8. EdgeConv auxiliary decoder 输出辅助 displacement
9. 联合 displacement、surface、geometry classification、gate matching loss 训练
```

主训练前 20 个 epoch 会冻结 EdgeConv geometry teacher：

```text
model.encoder.edge_conditioner
model.edge_geom_head
```

这段时间 gate projection、denoising encoder 和 decoder 正常训练，gate 会向固定几何 teacher 学习。第 20 个 epoch 后自动解冻，全模型联合微调。

主训练 loss 权重：

```yaml
displacement_loss: 0.9
edge_aux_displacement_loss: 0.2
normalized_surface_loss: 0.1
edge_geom_cls_loss: 0.05
edge_gate_match_loss: 0.02
```

关键参数：

```yaml
edgeconv_knn: 24
edgeconv_blocks: 2
edgeconv_hidden_dim: 256
num_edge_geom_classes: 12
edge_geom_hidden_dim: 128
edge_geom_knn: 32
edge_geom_match_temperature: 1.0
edge_geom_freeze_epochs: 20
edge_aux_hidden_dims: [128, 64]
```

主训练输出 checkpoint：

```text
outputs/checkpoints/vm_edgeconv/checkpoint_best.pkl
```

如果不想使用几何 teacher 预训练，也可以跑旧入口：

```bash
python run.py --task configs/task/train_vm_ssl.yaml --seed 123
```

但当前推荐流程是 `train_edge_geom_pretrain.yaml` 后接 `train_vm_edgegeom.yaml`。

## 诊断 EdgeConv 和 gate 几何区分度

训练后可以检查 EdgeConv condition feature 是否包含曲率、尖锐法向变化、转角信息，以及 gate 是否真正学到了几何调制：

```bash
python tools/diagnose_edgeconv_geometry.py \
  --checkpoint outputs/checkpoints/vm_edgeconv/checkpoint_best.pkl \
  --out-dir analysis_outputs/edgeconv_geometry \
  --patches 40 \
  --points-per-patch 384 \
  --seed 20260601
```

输出：

```text
analysis_outputs/edgeconv_geometry/
  summary.json
  edge_condition_geometry.csv
  edge_gate_geometry.csv
```

重点看：

```text
edge_condition_*_r2_pc16
edge_condition_*_best_pc_auc_q75_vs_q25
gate_*_r2_pc16
gate_*_best_pc_auc_q75_vs_q25
```

如果 `edge_condition` 指标高但 `gate` 指标低，说明 EdgeConv 学到了几何，但调制没有跟上。加入 gate matching 后，期望 gate 的 AUC/R2 明显提高。

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
# 1. 生成 clean 缓存
python scripts/cache_clean_points.py \
  --input_dataset_dir dataset_clean \
  --output_dir cache_clean_points \
  --datalist datalist/train.txt datalist/validate.txt \
  --workers 8 \
  --seed 123 \
  --overwrite

# 2. 预训练 EdgeConv geometry teacher
python run.py --task configs/task/train_edge_geom_pretrain.yaml --seed 123

# 3. 主去噪训练，前 20 epoch 冻结 geometry teacher
python run.py --task configs/task/train_vm_edgegeom.yaml --seed 123

# 4. 诊断 EdgeConv / gate 几何区分度
python tools/diagnose_edgeconv_geometry.py \
  --checkpoint outputs/checkpoints/vm_edgeconv/checkpoint_best.pkl \
  --out-dir analysis_outputs/edgeconv_geometry

# 5. 推理
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
