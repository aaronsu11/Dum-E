"""Optional exhaustive operation tracing shared with native diagnostics.

Copied from the reviewed serving observer without changing RNG or model calls.
This supplies telemetry, not a standalone release attestation.
"""
import contextlib
import torch
from policy_guard.replay_contract import floating_dtypes

class ServingObservation:
    """Observe actual operations and sampler facts without changing RNG or outputs.

    Pinned PyTorch modes forward each operation exactly once. Module hooks return
    None; no upstream method is rebound, and no extra random sample is drawn.
    """

    def __init__(self, model):
        from torch.overrides import TorchFunctionMode
        from torch.utils._python_dispatch import TorchDispatchMode

        self.model = model
        self.flow_steps = 0
        self.sdpa_calls = 0
        self.noise_draws = 0
        self.noise_shape = None
        self.noise_dtype = None
        self.noise_device = None
        self.compute_dtypes = set()
        self.input_dtypes = set()
        self.backbone_dtypes = set()
        self.kernels = set()
        self.floating_operation_count = 0
        self.autocast = False
        self.tf32 = self.tf32_matmul = self.tf32_cudnn = False
        owner = self

        class FunctionObserver(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                owner.observe_context()
                result = func(*args, **kwargs)
                if func is torch.randn:
                    owner.noise_draws += 1
                    owner.noise_shape = list(result.shape)
                    owner.noise_dtype = str(result.dtype)
                    owner.noise_device = str(result.device)
                if func is torch.nn.functional.scaled_dot_product_attention:
                    owner.sdpa_calls += 1
                return result

        class DispatchObserver(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                owner.observe_context()
                dtypes = floating_dtypes(args) | floating_dtypes(kwargs)
                result = func(*args, **kwargs)
                dtypes.update(floating_dtypes(result))
                if dtypes:
                    owner.floating_operation_count += 1
                    owner.compute_dtypes.update(dtypes)
                if "scaled_dot_product" in str(func):
                    owner.kernels.add(str(func))
                return result

        self.function_mode = FunctionObserver()
        self.dispatch_mode = DispatchObserver()

    def observe_context(self):
        self.autocast |= torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")
        self.tf32_matmul |= torch.backends.cuda.matmul.allow_tf32
        self.tf32_cudnn |= torch.backends.cudnn.allow_tf32
        self.tf32 = self.tf32_matmul or self.tf32_cudnn

    def __enter__(self):
        self.stack = contextlib.ExitStack()
        try:
            def before_backbone(module, args):
                self.input_dtypes.update(floating_dtypes(args))

            def after_backbone(module, args, result):
                self.backbone_dtypes.update(floating_dtypes(result))

            def before_step(module, args):
                self.flow_steps += 1

            for hook in (
                self.model.backbone.register_forward_pre_hook(before_backbone),
                self.model.backbone.register_forward_hook(after_backbone),
                self.model.action_head.action_encoder.register_forward_pre_hook(before_step),
            ):
                self.stack.callback(hook.remove)
            self.stack.enter_context(self.function_mode)
            self.stack.enter_context(self.dispatch_mode)
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


