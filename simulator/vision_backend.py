"""One FP32 pinned detector; measured MPS acceleration with bounded fallback."""
import sys
import time


def equivalent(a,b):
    import numpy as np
    labels=lambda r:list(r.get('text_labels',r.get('labels',[])))
    if labels(a)!=labels(b) or len(a['scores'])!=len(b['scores']):return False
    if not len(a['scores']):return True
    array=lambda t:t.detach().cpu().numpy() if hasattr(t,'detach') else np.asarray(t)
    return bool(np.max(np.abs(array(a['scores'])-array(b['scores'])))<.001 and
        np.max(np.abs(array(a['boxes'])-array(b['boxes'])))<.05)


class PinnedDetectorBackend:
    def __init__(self,torch,model,processor):
        self.torch,self.model,self.processor=torch,model,processor
        self.device='cpu';self.checked=False;self.measurement=dict(device='cpu',dtype='float32',calibrated=False)

    def _infer(self,inputs,size,device):
        batch={k:v.to(device) for k,v in inputs.items()}
        if device=='mps':self.torch.mps.synchronize()
        started=time.perf_counter()
        with self.torch.inference_mode():output=self.model(**batch)
        if device=='mps':self.torch.mps.synchronize()
        elapsed=time.perf_counter()-started
        result=self.processor.post_process_grounded_object_detection(output,batch['input_ids'],threshold=.35,
            text_threshold=.25,target_sizes=[size])[0]
        # All returned tensors are on CPU before releasing this inference.
        result={k:v.detach().cpu() if hasattr(v,'detach') else v for k,v in result.items()}
        return result,elapsed

    def infer(self,inputs,size):
        if not self.checked:
            self.checked=True
            # Warm CPU and accelerator on the exact same observed RGB. This
            # cannot create a demonstration or grant task qualification.
            first,_=self._infer(inputs,size,'cpu')
            if sys.platform=='darwin' and self.torch.backends.mps.is_available():
                from .resource_budget import memory_budget
                memory=memory_budget()
                if memory['available_bytes']/memory['total_bytes']>=.25:
                    try:
                        cpu,cpu_s=self._infer(inputs,size,'cpu')
                        self.model.to('mps');self._infer(inputs,size,'mps')
                        mps,mps_s=self._infer(inputs,size,'mps')
                        same=equivalent(cpu,mps);speedup=cpu_s/max(.001,mps_s)
                        after=memory_budget()
                        enough=after['available_bytes']/after['total_bytes']>=.20
                        if same and speedup>=1.2 and enough:self.device='mps'
                        else:self.model.to('cpu');self.torch.mps.empty_cache()
                        self.measurement.update(device=self.device,calibrated=True,speedup=speedup,
                            same_recorded_detections=same,memory_headroom=enough)
                    except (RuntimeError,NotImplementedError) as error:
                        self.model.to('cpu');self.torch.mps.empty_cache()
                        self.measurement.update(device='cpu',fallback=type(error).__name__+': '+str(error))
            return first
        try:return self._infer(inputs,size,self.device)[0]
        except (RuntimeError,NotImplementedError) as error:
            if self.device=='cpu':raise
            self.model.to('cpu');self.device='cpu';self.torch.mps.empty_cache()
            self.measurement.update(device='cpu',fallback=type(error).__name__+': '+str(error))
            return self._infer(inputs,size,'cpu')[0]
