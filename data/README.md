# Data

This directory is for dataset notes only. Do not commit raw datasets or generated
point clouds.

Expected training data layout:

```text
dataset_train/
  shapenet/
    <synset_id>/
      <model_id>/
        models/
          model_normalized.obj
```

Expected test data layout:

```text
dataset_test_noisy/
  shapenet/
    <synset_id>/
      <model_id>/
        noisy.npy
```

The dataset roots are configured in:

- `configs/data/train.yaml`
- `configs/data/predict.yaml`
