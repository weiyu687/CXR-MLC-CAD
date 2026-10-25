"""Stable deployment kernels across independent processes on the same runtime."""
import os
import platform
import hashlib
import json
import torch

def deployment_memory_format(device):
    if device.type=='cuda':return torch.channels_last
    # The Windows CPU regression exposed a native convolution access violation
    # in the accelerated path. Use the standard CPU layout/backend instead.
    torch.backends.mkldnn.enabled=False
    return torch.contiguous_format


def model_input_fingerprint(checkpoint):
    """Identity of fixed parameters/buffers and the deterministic input setup."""
    config=checkpoint.get('config',{});state=checkpoint.get('state_dict')
    if not state or 'model' not in config or 'size' not in config:return None
    digest=hashlib.sha256(b'model-input-v1\0')
    digest.update(json.dumps([config['model'],config['size'],config.get('normalization','legacy')]).encode())
    for key in sorted(state):
        value=state[key]
        if not isinstance(value,torch.Tensor):return None
        try:data=value.detach().cpu().numpy().tobytes(order='C')
        except (TypeError,RuntimeError):return None
        description=json.dumps([key,str(value.dtype),list(value.shape)]).encode()
        digest.update(len(description).to_bytes(8,'little'));digest.update(description)
        digest.update(len(data).to_bytes(8,'little'));digest.update(data)
    return digest.hexdigest()


def load_deployed_model(checkpoint,device):
    from .model import build_model
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=build_model(ck['config']['model'])
    model.load_state_dict(ck['state_dict'],strict=True)
    return model.to(device,memory_format=deployment_memory_format(device)).eval(),ck

def forward_probabilities(model,x,device,tta=False):
    x=x.to(device,non_blocking=True).to(memory_format=deployment_memory_format(device))
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
    info={'python':platform.python_version(),'operating_system':platform.system(),'torch':str(torch.__version__),
            'torchvision':str(torchvision.__version__),'timm':timm.__version__,'Pillow':PIL.__version__,
            'device':str(device),'device_name':torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',
            'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'inference_dtype':'float16' if device.type=='cuda' else 'float32'}
    if device.type=='cpu':
        info.update(inference_memory_format='contiguous',cpu_mkldnn_enabled=torch.backends.mkldnn.enabled)
    return info
