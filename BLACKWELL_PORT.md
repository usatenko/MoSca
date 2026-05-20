# MoSca on CUDA 13 / Blackwell (sm_120) — port notes

This fork makes MoSca run on NVIDIA Blackwell GPUs (e.g. RTX 5060 Ti, compute capability sm_120),
which require **CUDA 13 + PyTorch 2.11 (cu130)**. The upstream pinned cu11/cu12 toolchain does not
run on sm_120. Built/tested on vast.ai, PyTorch 2.11.0+cu130, Python 3.12.

## Build the CUDA extensions (sm_120)
Set before building every native module:
```
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
pip install --no-build-isolation ./lib_render/simple-knn
pip install --no-build-isolation ./lib_render/diff-gaussian-rasterization-alphadep
pip install --no-build-isolation ./lib_render/diff-gaussian-rasterization-alphadep-add3
pip install --no-build-isolation ./lib_render/gof-diff-gaussian-rasterization
# pytorch3d, torch-scatter/sparse/cluster, mmcv all built from source with same flags
```

## Source patches in this fork
- **CUDA headers**: added `#include <cstdint>` to all `cuda_rasterizer/*.h` + simple-knn headers
  (CUDA 13 / newer GCC no longer pulls it in transitively).
- **xformers -> torch SDPA**: Metric3D ViT (`ViT_DINO_reg.py`, fetched via torch.hub) used
  `xformers.memory_efficient_attention`, unsupported on sm_120. Replaced with
  `torch.nn.functional.scaled_dot_product_attention` (see hub-cache patch in pipeline docs).
- **cupy**: `softsplat.py` used removed `cupy.cuda.compile_with_cache(...)` -> `cupy.RawModule`.
  Install `cupy-cuda13x`.
- **numpy 2.0**: `np.NaN`->`np.nan`, `ndarray.tostring()`->`tobytes()`,
  `np.fromstring(bytes,sep="")`->`np.frombuffer` shim (in mosca_precompute/reconstruct headers).
- **matplotlib 3.10**: `FigureCanvasAgg.tostring_rgb()` removed -> `buffer_rgba()` shim.
- **torch.hub trust prompt**: monkeypatched `_check_repo_is_trusted` (headless, no stdin).
- **broken committed symlink**: `lib_mosca/camera.py` was a text file containing a path -> real symlink.
- **viz_utils.py**: `torch.from_numpy(T_cw_viz)` guarded for already-tensor input.
- **spatracker viz**: wrapped in try/except (imageio mp4 plugin issues).

## Extra Python deps needed
`timm cupy-cuda13x lpips evo imageio-ffmpeg roma plyfile natsort scikit-image opencv-python kornia tensorboardX transforms3d easydict open3d`
(imageio-ffmpeg is REQUIRED or MoSca writes "mp4" files that are actually multi-page TIFFs.)

## Memory (16 GB VRAM)
- Inputs downscaled to ~640x360 (720p OOMs DepthCrafter).
- SpaTracker track counts halved (8192) in profile yaml; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Run
```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python mosca_precompute.py   --cfg ./profile/demo/demo_prep.yaml --ws ./demo/<scene>
python mosca_reconstruct.py  --cfg ./profile/demo/demo_fit.yaml  --ws ./demo/<scene>
```
Outputs: `logs/.../photometric_{s,d}_model_*.pth` (static + dynamic Gaussians), viz mp4s.

## Export to standard 3DGS .ply
`export_ply.py` (full scene at time t), `export_dyn_seq.py` / `export_dyn_pruned.py` (dynamic players per frame).
See `football_pipeline/` for the monocular-broadcast football pipeline (SAM2 masks, viewers).
