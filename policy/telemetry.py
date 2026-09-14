"""Shared opt-in chunk telemetry, distinct from exhaustive parity attestation.

No operation interception, tensor copies, RNG manipulation or claimed continuous
precision proof. Off and lightweight use the same optional timing boundaries.
"""
from contextlib import ExitStack
from time import perf_counter

MODES = ('off', 'lightweight')


def tensor_metadata(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device)}
    if hasattr(value, 'items'):
        result = {}
        for key, child in value.items():
            metadata = tensor_metadata(child)
            if metadata is not None:
                result[str(key)] = metadata
        return result
    if isinstance(value, (tuple, list)):
        return [metadata for child in value if (metadata := tensor_metadata(child)) is not None]
    return None


class ChunkObserver:
    def __init__(self, mode='lightweight', *, synchronize=None):
        if mode not in MODES:
            raise ValueError(f'Unknown observer mode: {mode!r}')
        self.mode = mode
        self.synchronize = synchronize or (lambda: None)
        self.last = None

    def run(self, model, predict, *args, **kwargs):
        record = {'observer_mode': self.mode, 'coverage': {'off': 'none', 'lightweight': 'module_boundaries'}[self.mode],
                  'exhaustive_attestation': False, 'status': 'failed'}
        self.last = record
        with ExitStack() as stack:
            if self.mode == 'lightweight':
                record['flow_steps'] = 0
                def before_backbone(module, inputs, kwargs):
                    record['backbone_inputs'] = tensor_metadata({'args': inputs, 'kwargs': kwargs})
                def after_backbone(module, inputs, kwargs, output):
                    record['backbone_outputs'] = tensor_metadata(output)
                def before_step(module, inputs):
                    record['flow_steps'] += 1
                stack.callback(model.backbone.register_forward_pre_hook(before_backbone, with_kwargs=True).remove)
                stack.callback(model.backbone.register_forward_hook(after_backbone, with_kwargs=True).remove)
                stack.callback(model.action_head.action_encoder.register_forward_pre_hook(before_step).remove)
            self.synchronize()
            start = perf_counter()
            try:
                result = predict(*args, **kwargs)
                self.synchronize()
            finally:
                record['generation_ms'] = (perf_counter() - start) * 1000
            record['status'] = 'complete'
            return result
