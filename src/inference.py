"""Stable deployment kernels across independent processes on the same runtime."""
import os
import platform
import torch

def load_deployed_model(checkpoint,device):
    from .model import build_model
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=build_model(ck['config']['model'])
    model.load_state_dict(ck['state_dict'],strict=True)
    return model.to(device,memory_format=torch.channels_last).eval(),ck

def forward_probabilities(model,x,device,tta=False):
    x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last)
    with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
        p=model(x).float().sigmoid()
        if tta:p=(p+model(x.flip(-1)).float().sigmoid())/2
    return p

def configure_deterministic_inference():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)

def runtime_information(device):
    import torchvision
    import timm
    import PIL
    return {'python':platform.python_version(),'operating_system':platform.system(),'torch':str(torch.__version__),
            'torchvision':str(torchvision.__version__),'timm':timm.__version__,'Pillow':PIL.__version__,
            'device':str(device),'device_name':torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',
            'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'inference_dtype':'float16' if device.type=='cuda' else 'float32'}
