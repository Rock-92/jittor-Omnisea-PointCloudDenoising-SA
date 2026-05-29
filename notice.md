# A800 Environment Setup Notes

This note records the successful environment setup pipeline used on the A800 cloud server. It only covers environment creation and Jittor/CUDA initialization, not dataset caching, training, or inference.

## Server Configuration

```text
GPU: A800 x 1
GPU memory: 80GB
CPU model: 2 x Intel 8358P 32C
CPU cores shown by platform: 14 cores
RAM: 245GB
Resource group: 64 group
Resource group name: nva800normal
Image: jupyterlab-pytorch:2.2.0-py3.10-cuda12.1-ubuntu22.04-devel
```

## Successful Setup Pipeline

Run these commands from the project root.

```bash
conda env create -f environment.yml
```

Creates the project conda environment from `environment.yml`. In this project the environment name is `jittor`.

```bash
source activate jittor
```

Activates the newly created environment. On this server, `source activate` is used instead of `conda activate`.

```bash
conda install -c conda-forge libgomp -y
```

Installs the GNU OpenMP runtime. This is needed by Jittor's downloaded MKL-DNN/DNNL dependency; without it, Jittor import can fail with missing `libgomp.so.1` or unresolved `GOMP_*` symbols.

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

Adds the active conda environment library directory to the dynamic linker path so Jittor can find `libgomp.so.1` and other shared libraries installed into the environment.

```bash
python3.9 -m jittor_utils.install_cuda
```

Lets Jittor install its own compatible CUDA/cuDNN runtime under the Jittor cache directory. On this server, Jittor detected CUDA driver version `12.9` and installed:

```text
~/.cache/jittor/jtcuda/cuda11.2_cudnn8_linux
```

This step resolves the Jittor import error where CUDA is found but cuDNN development files are not loaded.

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

Exports the library path again after the CUDA/cuDNN installation step. This keeps the conda environment libraries at the front of the runtime search path.

```bash
python -c "import jittor as jt; print(jt.__version__)"
```

Verifies that Jittor imports correctly. The successful output on this server was:

```text
1.3.8.5
```

## Final Command Block

For a fresh server with the same image and resource configuration, the successful setup sequence is:

```bash
conda env create -f environment.yml
source activate jittor
conda install -c conda-forge libgomp -y
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python3.9 -m jittor_utils.install_cuda
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python -c "import jittor as jt; print(jt.__version__)"
```

## Notes

- The first Jittor import may compile operators and download dependencies. This can take several minutes.
- `source activate jittor` was the working activation command on this server.
- `libgomp` is required before a clean Jittor import because Jittor's MKL-DNN/DNNL setup links against OpenMP.
- `python3.9 -m jittor_utils.install_cuda` installs a Jittor-managed CUDA/cuDNN runtime and avoids relying on missing system cuDNN development files.
