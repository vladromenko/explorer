#!/usr/bin/env python3
"""Exercise official ACT and CUDA without actuators or a robot dataset."""
import importlib.metadata
import json
from pathlib import Path
import time

root=Path('/home/vlad/Explorer');result=dict(at=time.time(),ready=False)
try:
    import torch
    from lerobot.configs import FeatureType,PolicyFeature
    from lerobot.policies.act import ACTConfig,ACTPolicy
    import lerobot.datasets
    torch.set_num_threads(2)
    result.update(lerobot=importlib.metadata.version('lerobot'),torch=torch.__version__,
                  cuda_available=torch.cuda.is_available(),cuda=torch.version.cuda)
    if not torch.cuda.is_available():raise RuntimeError('CUDA недоступна')
    config=ACTConfig(device='cuda',chunk_size=1,n_action_steps=1,pretrained_backbone_weights=None,
        dim_model=64,n_heads=4,dim_feedforward=128,n_encoder_layers=1,n_vae_encoder_layers=1,
        input_features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(6,)),
                        'observation.images.wrist':PolicyFeature(type=FeatureType.VISUAL,shape=(3,64,64))},
        output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(6,))})
    model=ACTPolicy(config).cuda().train()
    batch={'observation.state':torch.zeros(1,6,device='cuda'),
           'observation.images.wrist':torch.zeros(1,3,64,64,device='cuda'),
           'action':torch.zeros(1,1,6,device='cuda'),
           'action_is_pad':torch.zeros(1,1,device='cuda',dtype=torch.bool)}
    loss,_=model(batch);loss.backward();torch.cuda.synchronize()
    model.eval();action=model.select_action(batch)
    if not torch.isfinite(loss) or not torch.isfinite(action).all():raise RuntimeError('ACT produced nonfinite values')
    result.update(ready=False,gpu=torch.cuda.get_device_name(),act_forward_backward=True,
                  test_data='synthetic tensors; not a learned robot skill',
                  peak_gpu_mb=round(torch.cuda.max_memory_allocated()/1024**2),
                  action_shape=list(action.shape),autonomous_execution=False,
                  error='Вычисления ACT проверены; ожидается проверка полного цикла LeRobot',
                  compatibility_note='PyTorch warns that this aarch64 build does not officially list Orin CC 8.7; only the exercised ACT path is validated')
    integration=root/'data/learning-selftest-result.json'
    if integration.exists():
        report=json.loads(integration.read_text())
        if report.get('torch')==torch.__version__ and report.get('lerobot')==result['lerobot']:
            result.update(ready=True,dataset_train_save_reload_validate=True)
            result.pop('error',None)
except Exception as exc:result['error']=str(exc)
path=root/'data/learning-backend.json';temporary=path.with_suffix('.tmp')
temporary.write_text(json.dumps(result));temporary.replace(path)
print(json.dumps(result))
