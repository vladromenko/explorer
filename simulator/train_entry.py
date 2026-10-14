"""LeRobot entry point for explicit warm starts with unchanged normalization.

LeRobot 0.6.1 refreshes processor statistics when fine-tuning. Explorer preserves
the source checkpoint's coordinate system when adding verified demonstrations.
Measured dataset statistics on disk are never overwritten.
"""
from pathlib import Path


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


def main():
    import os
    from lerobot.scripts import lerobot_train
    reference=os.environ.get('EXPLORER_FROZEN_NORMALIZATION')
    if not reference:raise ValueError('Warm-start normalization reference is required')
    lerobot_train.make_pre_post_processors=frozen_processor_call(lerobot_train.make_pre_post_processors,reference)
    lerobot_train.train()


if __name__=='__main__':main()
