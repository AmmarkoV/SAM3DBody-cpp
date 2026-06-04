# Setup for WSL2 (Ubuntu)

This guide gets the CUDA build of **SAM3DBody-cpp** running under **WSL2** on
Windows. WSL has two quirks that trip people up:

1. You need a CUDA **12.x** toolkit + **cuDNN 9.x** that match the ONNX Runtime
   1.20.1 GPU build (see [DEPENDENCIES.md](DEPENDENCIES.md)).
2. WSL prepends every Windows `PATH` entry to your Linux environment, so the
   loader can pick up the *Windows* GPU driver instead of the WSL one unless you
   fix `PATH` / `LD_LIBRARY_PATH` (Step 6).

> **Do not install the full Linux NVIDIA driver inside WSL.** The GPU driver lives
> on the *Windows* side. Inside WSL you install only the **CUDA toolkit** and
> **cuDNN**.

---

## Prerequisites

- Windows 10/11 with **WSL2** and an Ubuntu distro installed.
- A recent **NVIDIA driver installed on Windows** (the WSL GPU passthrough comes
  from it).
- Verify the GPU is visible from inside WSL **before** continuing:

  ```bash
  nvidia-smi
  ```

  This must list your GPU. If it doesn't, fix the Windows driver / WSL setup
  first — nothing below will work otherwise.

---

## Step 1 — Enable the `lunar` universe repo (for gcc-12)

The build needs **gcc-12**, which isn't in the default repos on newer Ubuntu
releases. Add the archived `lunar` (23.04) universe repo to pull it in.

Open the sources file:

```bash
sudo nano /etc/apt/sources.list.d/ubuntu.sources
```

Paste this at the end:

```
Types: deb
URIs: http://old-releases.ubuntu.com/ubuntu/
Suites: lunar
Components: universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
```

Save, then refresh:

```bash
sudo apt update
```

> If your Ubuntu already provides `gcc-12` (`apt-cache policy gcc-12` shows a
> candidate), you can skip this step.

---

## Step 2 — Install gcc-12

```bash
sudo apt install gcc-12
```

---

## Step 3 — Install CUDA Toolkit 12.x

Install the **WSL-Ubuntu** CUDA toolkit (this guide uses 12.2):

https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

Pick **Linux → x86_64 → WSL-Ubuntu** so you get the WSL package (not the full
desktop driver).

Verify:

```bash
nvcc --version
```

Expected (version line is what matters):

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Cuda compilation tools, release 12.2, V12.2.91
Build cuda_12.2.r12.2/compiler.32965470_0
```

---

## Step 4 — Install cuDNN 9.x (for CUDA 12)

ONNX Runtime 1.20.1 requires **cuDNN 9** — cuDNN 8 will *not* work and is the #1
cause of `Failed to load library libonnxruntime_providers_cuda.so`.

Get the install command for your distro here (choose **Linux → x86_64 →
Ubuntu**, cuDNN for CUDA 12):

https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_type=deb_local

After installing, refresh the loader cache and confirm cuDNN 9 is present:

```bash
sudo ldconfig
ldconfig -p | grep -i cudnn
```

You should see `libcudnn.so.9` listed.

---

## Step 5 — Set your GPU's CUDA architecture in `scripts/build.sh`

Set `-DCMAKE_CUDA_ARCHITECTURES` for your card
(**89** = Ada: RTX 40-series incl. RTX 4060; `86` = Ampere RTX 30-series;
`75` = Turing RTX 20-series):

```bash
#!/bin/bash

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89

make -j$(nproc)

exit 0
```

Then build:

```bash
bash scripts/build.sh
```

---

## Step 6 — Fix the WSL GPU library path before running

WSL appends all Windows `PATH` entries to your Linux environment, so the loader
can resolve the *Windows* GPU driver path instead of the WSL one. Export clean
paths so Linux sees the WSL GPU libs (`/usr/lib/wsl/lib`) and your CUDA install:

```bash
export PATH=/usr/local/cuda-12.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-12.2/lib64
```

> Adjust `cuda-12.2` to your installed version. To make this permanent, append
> the two `export` lines to `~/.bashrc` (then `source ~/.bashrc`).

### Verify CUDA EP can load

Before running the pipeline, confirm the ONNX Runtime CUDA provider has all its
dependencies (no output = success):

```bash
ldd build/onnxruntime_dl/lib/libonnxruntime_providers_cuda.so | grep -i "not found"
```

If anything shows up here, revisit Step 4 (cuDNN) and Step 6 (paths). See
[DEPENDENCIES.md](DEPENDENCIES.md) for the full troubleshooting flow, including
why a CPU fallback fails with `Could not find an implementation for Expand(13)`.

Now run, e.g.:

```bash
scripts/webcam.sh
```

---

## Blender tip — MakeHuman/MPFB installed via Blender extensions

If you installed Blender manually and the MakeHuman (MPFB) extension *through
Blender's extension system*, edit `blender/blender_bvh_plugin.py`.

Find:

```python
if __name__ == "__main__":
```

and inside the loop over add-ons add this `elif`:

```python
elif "mpfb" in addon.lower():   # MPFB installed as a Blender extension
    haveMPFB = True
```
