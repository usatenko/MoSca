# Monocular broadcast football -> 4DGS pipeline

End-to-end pipeline built on this Blackwell port of MoSca (see ../BLACKWELL_PORT.md).
Goal: single broadcast clip -> dynamic 4D Gaussian scene + hybrid stadium/player viewers.

## Steps
1. **Frames**: extract ~50 frames from a clip, downscale to 640x360 (16GB VRAM limit), into
   `demo/football/images/00000.jpg ...`
2. **Player masks** (`gen_sam2_masks.py`): YOLO11 detects players on frame 0, SAM2 video predictor
   propagates instance masks across all frames, written as RGB-instance PNGs to
   `demo/football/vos_deva/Annotations/` (MoSca DEVA/VOS format; RGB2INST = R+G*256+B*65536).
3. **MoSca**: `mosca_precompute.py` then `mosca_reconstruct.py` on `demo/football`. The vos masks
   focus dynamic Gaussians on players (static pitch handled by the static model).
4. **Export real splats**: `../export_ply.py` (full scene @ t), `../export_dyn_pruned.py`
   (pruned dynamic players per frame). Static stadium = static model export.
5. **Viewers** (hybrid, fast/real-time, reuse tracking+masks, no optimization):
   - `make_stadium_viewer.py` -> self-contained three.js: procedural stadium + player billboard sprites.

## Honest limits
- Mono broadcast = PTZ camera (rotation/zoom, little translation) -> no parallax -> cannot reconstruct
  a true 3D stadium; static stadium is best sourced synthetic or from dedicated capture.
- Real-splat players from mono are mushy; billboard sprites (real pixels) look sharper.
- Heuristic image->pitch projection is uncalibrated; TVCalib is the accuracy upgrade.
