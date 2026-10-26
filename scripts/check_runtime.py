"""Check existing GPU runtime without downloads or package changes."""
import json,sys
import torch,mmcv,mmdet,mmdet3d,mmseg,spconv,torch_scatter,flash_attn_cuda
if not torch.cuda.is_available():
 raise SystemExit("CUDA is unavailable. Check Windows NVIDIA driver and WSL2 GPU support.")
x=torch.arange(16,dtype=torch.float32,device="cuda").reshape(4,4)
assert torch.isfinite(x@x).all()
print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,
 mmcv=mmcv.__version__,mmdet=mmdet.__version__,mmdet3d=mmdet3d.__version__,
 mmseg=mmseg.__version__,spconv=spconv.__version__,
 gpu=torch.cuda.get_device_name(0),cuda_tensor_operation="PASS"),indent=2))
