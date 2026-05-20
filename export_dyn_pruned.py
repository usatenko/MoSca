import os,sys,os.path as osp,numpy as np,torch,glob
os.environ.setdefault("GS_BACKEND","native_add3")
from lib_render.render_helper import GS_BACKEND
from lib_mosca.dynamic_gs import DynSCFGaussian
from plyfile import PlyData,PlyElement
import roma
logdir=sys.argv[1]; outdir=sys.argv[2]; OPA=float(sys.argv[3]) if len(sys.argv)>3 else 0.2
os.makedirs(outdir,exist_ok=True); dev="cuda"
d=DynSCFGaussian.load_from_ckpt(torch.load(osp.join(logdir,f"photometric_d_model_{GS_BACKEND.lower()}.pth")),device=dev).to(dev).eval()
T=d.T

def q_wxyz(R):
    q=roma.rotmat_to_unitquat(R)
    return torch.cat([q[:,3:4],q[:,:3]],-1)

def write(path,mu,fr,s,o,sph):
    N=mu.shape[0]
    xyz=mu.cpu().numpy().astype(np.float32); quat=q_wxyz(fr).cpu().numpy().astype(np.float32)
    sc=np.log(np.clip(s.cpu().numpy(),1e-8,None)).astype(np.float32)
    op=o.cpu().numpy().reshape(-1); op=np.log(np.clip(op,1e-6,1-1e-6)/(1-np.clip(op,1e-6,1-1e-6))).astype(np.float32).reshape(-1,1)
    sp=sph.cpu().numpy().astype(np.float32); fdc=sp[:,:3]; fr2=sp[:,3:] if sp.shape[1]>3 else np.zeros((N,0),np.float32)
    flds=["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2"]+[f"f_rest_{i}" for i in range(fr2.shape[1])]+["opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
    arr=np.zeros(N,dtype=[(f,"f4") for f in flds])
    arr["x"],arr["y"],arr["z"]=xyz[:,0],xyz[:,1],xyz[:,2]
    arr["f_dc_0"],arr["f_dc_1"],arr["f_dc_2"]=fdc[:,0],fdc[:,1],fdc[:,2]
    for i in range(fr2.shape[1]): arr[f"f_rest_{i}"]=fr2[:,i]
    arr["opacity"]=op[:,0]; arr["scale_0"],arr["scale_1"],arr["scale_2"]=sc[:,0],sc[:,1],sc[:,2]
    arr["rot_0"],arr["rot_1"],arr["rot_2"],arr["rot_3"]=quat[:,0],quat[:,1],quat[:,2],quat[:,3]
    PlyData([PlyElement.describe(arr,"vertex")]).write(path)

with torch.no_grad():
    mu0,_,_,o0,_=d(T//2)
    keep0=o0.reshape(-1)>OPA
    c=mu0[keep0]
    lo=torch.quantile(c,0.02,0); hi=torch.quantile(c,0.98,0); pad=2.0
    print("crop lo",lo.cpu().numpy().round(1),"hi",hi.cpu().numpy().round(1),"midframe kept",int(keep0.sum()))
    for t in range(T):
        mu,fr,s,o,sph=d(t)
        op=o.reshape(-1)
        inb=((mu>=lo-pad)&(mu<=hi+pad)).all(-1)
        keep=(op>OPA)&inb
        write(osp.join(outdir,f"players_{t:03d}.ply"),mu[keep],fr[keep],s[keep],o[keep],sph[keep])

fs=sorted(glob.glob(outdir+"/*.ply"))
print("wrote",len(fs),"plys, total %.1f MB"%(sum(os.path.getsize(f) for f in fs)/1e6),"frame0 gaussians:",PlyData.read(fs[0]).elements[0].count)
