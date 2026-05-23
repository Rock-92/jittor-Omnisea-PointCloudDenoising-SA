# Jittor Point Cloud Denoising Baseline

本项目是点云降噪赛题的 Jittor baseline。模型从 ShapeNet 干净网格采样点云、添加噪声，并学习逐点位移 `dx, dy, dz`，将含噪点云推回物体表面附近。

## 环境安装

推荐环境：

- Python 3.9
- Jittor
- CUDA GPU 环境

安装示例：

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
python -m pip install -r requirements.txt
pip install point-cloud-utils
```

## 数据准备

训练数据放在项目根目录下，结构如下：

```text
dataset_train/
  shapenet/
    <synset_id>/
      <model_id>/
        models/
          model_normalized.obj
```

测试数据结构如下：

```text
dataset_test_noisy/
  shapenet/
    <synset_id>/
      <model_id>/
        noisy.npy
```

数据列表在 `datalist/` 下。数据根目录配置在：

- `configs/data/train.yaml`
- `configs/data/predict.yaml`

## 项目结构

```text
configs/      配置文件
data/         数据说明，不提交原始数据
datalist/     训练、验证、测试样本列表
scripts/      训练、推理、评测入口
src/          核心代码
tools/        可选工具脚本
outputs/      日志、权重、结果，默认不提交
```

## 训练

默认训练命令：

```bash
python scripts/train.py --seed 123
```

等价兼容命令：

```bash
python run.py --task configs/task/train_vm.yaml --seed 123
```

训练超参数在 `configs/train/vm.yaml`：

```yaml
optimizer:
  lr: 0.00001

trainer:
  epochs: 100
```

模型结构参数在 `configs/model/vm.yaml`。默认权重保存到：

```text
outputs/checkpoints/vm/
```

每次运行会在 `outputs/runs/<mode>/<timestamp>/` 保存：

- `config.json`：实际加载的配置
- `command.txt`：运行命令

## 推理

先确认 `configs/task/predict_vm.yaml` 里的 `load_ckpt` 指向已训练权重，然后运行：

```bash
python scripts/infer.py --seed 123
```

默认输出：

```text
outputs/results/dataset_test_noisy/
  shapenet/
    <synset_id>/
      <model_id>/
        denoised.npy
```

打包提交：

```bash
cd outputs/results/dataset_test_noisy
zip -r ../../../result.zip shapenet/
```

## 评测

本地评测需要干净点云 GT 和原始网格：

```bash
python scripts/evaluate.py \
  --pred_dir outputs/results/dataset_test_noisy \
  --gt_dir test_gt \
  --noisy_dir dataset_test_noisy \
  --mesh_dir dataset_train \
  --workers 8
```

评测指标：

- `CD`：降噪点云与干净点云之间的 Chamfer Distance。
- `P2S`：降噪点云到原始网格表面的 Point-to-Surface Distance。

最终分数为 CD 分数和 P2S 分数的平均值；如果不提供 `mesh_dir`，只计算 CD。

## 结果说明

当前模型使用局部 self-attention encoder 和两层 MLP decoder。训练输入为含噪 patch `pc_noisy`，监督目标为 `pc_clean - pc_noisy` 的逐点位移残差。

线上分数可能和本地分数有差异，原因包括测试 GT 不公开、P2S 依赖网格和 `point-cloud-utils` 版本、硬件数值差异等。

## 开源说明

本仓库不包含数据集、训练权重和中间产物。第三方依赖见 `requirements.txt`，额外声明见 `NOTICE`。
