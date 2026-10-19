"""LeRobot entry point for explicit warm starts with unchanged normalization.

LeRobot 0.6.1 refreshes processor statistics when fine-tuning. Explorer preserves
the source checkpoint's coordinate system when adding verified demonstrations.
Measured dataset statistics on disk are never overwritten.
"""
from pathlib import Path
import os


def projected_horizon_query(reader,query_indices):
    """Read requested columns before applying image decoding/transforms.

    datasets 4.8.5 Column[list] applies the full row transform, so requesting
    twenty future ACTIONS also decodes twenty unused camera frames. Projection
    preserves those exact action tensors without accessing unrelated columns.
    """
    import torch
    if getattr(reader,'_explorer_projection_source',None) is not reader.hf_dataset:
        reader._explorer_projection_source=reader.hf_dataset;reader._explorer_projections={}
    result={}
    for key,indices in query_indices.items():
        if key not in reader._meta.video_keys:
            relative=indices if reader._absolute_to_relative_idx is None else [reader._absolute_to_relative_idx[i] for i in indices]
            if key not in reader._explorer_projections:
                reader._explorer_projections[key]=reader.hf_dataset.select_columns([key])
            result[key]=torch.stack(reader._explorer_projections[key][relative][key])
    return result


def install_query_projection():
    from lerobot.datasets.dataset_reader import DatasetReader
    DatasetReader._query_hf_dataset=projected_horizon_query


# Darwin DataLoader uses spawn: each child imports this entry module anew.
# The opt-in is inherited only by this training tree, never another process.
if os.environ.get('EXPLORER_PROJECTED_HORIZON_READ')=='1':install_query_projection()


def frozen_processor_call(original, reference):
    reference=Path(reference).resolve()
    def call(*args,**kwargs):
        path=kwargs.get('pretrained_path')
        if not path or Path(path).resolve()!=reference:
            raise ValueError('Unexpected normalization reference')
        kwargs.pop('dataset_stats',None)
        for kind in ('preprocessor_overrides','postprocessor_overrides'):
            kwargs[kind]={name:{key:value for key,value in options.items() if key!='stats'}
                          for name,options in kwargs.get(kind,{}).items()}
        return original(*args,**kwargs)
    return call


def measured_updates(original,output,warmup=20,samples=20):
    """Bounded synchronized measurements of the real update, not a second learner."""
    import time
    import torch
    rows=[];calls=0
    def wrapped(metrics,policy,batch,optimizer,grad_clip_norm,accelerator,*args,**kwargs):
        nonlocal calls
        calls+=1
        device=next(policy.parameters()).device
        if not warmup<calls<=warmup+samples:
            return original(metrics,policy,batch,optimizer,grad_clip_norm,accelerator,*args,**kwargs)
        def sync():
            if device.type=='mps':torch.mps.synchronize()
            elif device.type=='cuda':torch.cuda.synchronize(device)
        times={};forward_start=[None]
        def before(*_):sync();forward_start[0]=time.perf_counter()
        def after(*_):sync();times['forward_s']=time.perf_counter()-forward_start[0]
        pre=policy.register_forward_pre_hook(before);post=policy.register_forward_hook(after)
        old_backward,old_step=accelerator.backward,optimizer.step
        def timed(name,function):
            def call(*a,**k):
                sync();started=time.perf_counter();result=function(*a,**k);sync()
                times[name]=time.perf_counter()-started;return result
            return call
        accelerator.backward=timed('backward_s',old_backward)
        optimizer.step=timed('optimizer_s',old_step)
        sync();started=time.perf_counter()
        try:result=original(metrics,policy,batch,optimizer,grad_clip_norm,accelerator,*args,**kwargs)
        finally:
            sync();times['update_s']=time.perf_counter()-started
            accelerator.backward=old_backward;optimizer.step=old_step;pre.remove();post.remove()
        tensors={key:dict(device=str(value.device),shape=list(value.shape),dtype=str(value.dtype))
            for key,value in batch.items() if isinstance(value,torch.Tensor)}
        rows.append(dict(relative_update=calls,parameter_device=str(device),batch=tensors,**times))
        if len(rows)==samples:
            from .learning import write_json
            write_json(output,dict(kind='actual_learner_synchronized_window',pid=os.getpid(),
                warmup_updates=warmup,samples=rows,extra_training=False,
                limitation='Synchronized hooks measure a bounded window and change pipelining; CPU fallback operators are not profiled',
                at=time.time()))
        return result
    return wrapped


def main():
    from lerobot.scripts import lerobot_train
    reference=os.environ.get('EXPLORER_FROZEN_NORMALIZATION')
    os.environ['EXPLORER_PROJECTED_HORIZON_READ']='1';install_query_projection()
    # LeRobot's checkpoint resume already restores authoritative processors.
    # Only a weight-only warm start needs suppression of refreshed statistics.
    if reference and os.environ.get('EXPLORER_CHECKPOINT_RESUME')!='1':
        lerobot_train.make_pre_post_processors=frozen_processor_call(lerobot_train.make_pre_post_processors,reference)
    if os.environ.get('EXPLORER_UPDATE_MEASUREMENT'):
        lerobot_train.update_policy=measured_updates(lerobot_train.update_policy,os.environ['EXPLORER_UPDATE_MEASUREMENT'])
    lerobot_train.train()


if __name__=='__main__':main()
