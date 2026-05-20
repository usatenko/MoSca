import os, sys, os.path as osp, numpy as np, torch
os.environ.setdefault("GS_BACKEND","native_add3")
from lib_render.render_helper import GS_BACKEND
from lib_mosca.static_gs import StaticGaussian
from lib_mosca.dynamic_gs import DynSCFGaussian
from plyfile import PlyData, PlyElement

logdir = sys.argv[1]; t = int(sys.argv[2]) if len(sys.argv)>2 else 25
out = sys.argv[3] if len(sys.argv)>3 else f"/workspace/scene_t{t}.ply"
dev="cuda"
s_model = StaticGaussian.load_from_ckpt(torch.load(osp.join(logdir,f"photometric_s_model_{GS_BACKEND.lower()}.pth")), device=dev).to(dev).eval()
d_model = DynSCFGaussian.load_from_ckpt(torch.load(osp.join(logdir,f"photometric_d_model_{GS_BACKEND.lower()}.pth")), device=dev).to(dev).eval()

def mat2quat_wxyz(R):
    import roma
    q = roma.rotmat_to_unitquat(R)      # xyzw
    return torch.cat([q[:, 3:4], q[:, :3]], -1)  # wxyz

with torch.no_grad():
    parts=[]
    sm = s_model(); parts.append(sm)
    dm = d_model(t); parts.append(dm)
    mu = torch.cat([p[0] for p in parts],0)
    fr = torch.cat([p[1] for p in parts],0)
    s  = torch.cat([p[2] for p in parts],0)   # post-activation scale
    o  = torch.cat([p[3] for p in parts],0)   # post-sigmoid opacity
    sph= torch.cat([p[4] for p in parts],0)   # SH features (DC first 3)

xyz = mu.detach().cpu().numpy().astype(np.float32)
quat= mat2quat_wxyz(fr).detach().cpu().numpy().astype(np.float32)
# revert activations to raw (.ply stores pre-activation; viewers re-apply)
scale_raw = np.log(np.clip(s.detach().cpu().numpy(),1e-8,None)).astype(np.float32)
opa = o.detach().cpu().numpy().reshape(-1)
opacity_raw = np.log(np.clip(opa,1e-6,1-1e-6)/(1-np.clip(opa,1e-6,1-1e-6))).astype(np.float32).reshape(-1,1)
sphn = sph.detach().cpu().numpy().astype(np.float32)
f_dc = sphn[:, :3]
f_rest = sphn[:, 3:] if sphn.shape[1] > 3 else np.zeros((sphn.shape[0],0),np.float32)
N = xyz.shape[0]
print("total gaussians:", N, "f_rest dim:", f_rest.shape[1])

fields = ["x","y","z","nx","ny","nz"] + ["f_dc_0","f_dc_1","f_dc_2"] + [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
dtype=[(f,"f4") for f in fields]
arr=np.zeros(N,dtype=dtype)
arr["x"],arr["y"],arr["z"]=xyz[:,0],xyz[:,1],xyz[:,2]
arr["f_dc_0"],arr["f_dc_1"],arr["f_dc_2"]=f_dc[:,0],f_dc[:,1],f_dc[:,2]
for i in range(f_rest.shape[1]): arr[f"f_rest_{i}"]=f_rest[:,i]
arr["opacity"]=opacity_raw[:,0]
arr["scale_0"],arr["scale_1"],arr["scale_2"]=scale_raw[:,0],scale_raw[:,1],scale_raw[:,2]
arr["rot_0"],arr["rot_1"],arr["rot_2"],arr["rot_3"]=quat[:,0],quat[:,1],quat[:,2],quat[:,3]
PlyData([PlyElement.describe(arr,"vertex")]).write(out)
print("wrote", out, os.path.getsize(out)//1024,"KB")
