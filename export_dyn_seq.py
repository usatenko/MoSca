import os, sys, os.path as osp, numpy as np, torch
os.environ.setdefault("GS_BACKEND","native_add3")
from lib_render.render_helper import GS_BACKEND
from lib_mosca.dynamic_gs import DynSCFGaussian
from plyfile import PlyData, PlyElement
import roma
logdir=sys.argv[1]; outdir=sys.argv[2]; os.makedirs(outdir, exist_ok=True)
dev="cuda"
d=DynSCFGaussian.load_from_ckpt(torch.load(osp.join(logdir,f"photometric_d_model_{GS_BACKEND.lower()}.pth")),device=dev).to(dev).eval()
T=d.T; print("frames(T):",T)
def q_wxyz(R):
    q=roma.rotmat_to_unitquat(R); return torch.cat([q[:,3:4],q[:,:3]],-1)
def write(path,mu,fr,s,o,sph):
    xyz=mu.cpu().numpy().astype(np.float32); quat=q_wxyz(fr).cpu().numpy().astype(np.float32)
    sc=np.log(np.clip(s.cpu().numpy(),1e-8,None)).astype(np.float32)
    op=o.cpu().numpy().reshape(-1); op=np.log(np.clip(op,1e-6,1-1e-6)/(1-np.clip(op,1e-6,1-1e-6))).astype(np.float32).reshape(-1,1)
    sp=sph.cpu().numpy().astype(np.float32); fdc=sp[:,:3]; frest=sp[:,3:] if sp.shape[1]>3 else np.zeros((sp.shape[0],0),np.float32)
    N=xyz.shape[0]
    flds=["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2"]+[f"f_rest_{i}" for i in range(frest.shape[1])]+["opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
    arr=np.zeros(N,dtype=[(f,"f4") for f in flds])
    arr["x"],arr["y"],arr["z"]=xyz[:,0],xyz[:,1],xyz[:,2]
    arr["f_dc_0"],arr["f_dc_1"],arr["f_dc_2"]=fdc[:,0],fdc[:,1],fdc[:,2]
    for i in range(frest.shape[1]): arr[f"f_rest_{i}"]=frest[:,i]
    arr["opacity"]=op[:,0]; arr["scale_0"],arr["scale_1"],arr["scale_2"]=sc[:,0],sc[:,1],sc[:,2]
    arr["rot_0"],arr["rot_1"],arr["rot_2"],arr["rot_3"]=quat[:,0],quat[:,1],quat[:,2],quat[:,3]
    PlyData([PlyElement.describe(arr,"vertex")]).write(path)
with torch.no_grad():
    for t in range(T):
        mu,fr,s,o,sph=d(t)
        write(osp.join(outdir,f"players_{t:03d}.ply"),mu,fr,s,o,sph)
print("wrote", T, "dynamic player plys to", outdir)
import glob; print("total size MB:", round(sum(os.path.getsize(f) for f in glob.glob(outdir+"/*.ply"))/1e6,1))
