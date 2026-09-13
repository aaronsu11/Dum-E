"""LeRobot serving with off/lightweight chunk telemetry and serialized decode."""
import json
import os
from pathlib import Path
import sys
import threading
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'docker/lerobot-policy')]
from policy_guard.chunk_observer import ChunkObserver


def main():
    attestation = os.getenv('DUME_PARITY_ATTESTATION_PATH')
    mode = os.environ.get('DUME_CHUNK_OBSERVER', 'lightweight')
    if attestation:
        raise RuntimeError('Historical parity attestation is archived; unset DUME_PARITY_ATTESTATION_PATH')
    # Validate before importing/starting the server.
    ChunkObserver(mode)
    import torch
    import entrypoint
    class ObservedServer(entrypoint.DumEGrootPolicyServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pipeline_lock = threading.RLock()

        def SendPolicyInstructions(self, request, context):
            with self._pipeline_lock:
                return super().SendPolicyInstructions(request, context)

        def _predict_action_chunk(self, observation):
            # The relative-action anchor lives on the processor instance.
            # Hold through preprocessing, inference and decode, including failures.
            with self._pipeline_lock:
                return self._observed_chunk(observation)

        def _observed_chunk(self, observation):
            observer = ChunkObserver(mode, synchronize=torch.cuda.synchronize)
            try:
                return observer.run(self.policy._groot_model, self._predict_action_chunk_impl, observation)
            finally:
                self.logger.info('Chunk telemetry | %s', json.dumps({'backend': 'lerobot', **observer.last}))
    entrypoint.DumEGrootPolicyServer = ObservedServer
    return entrypoint.main()


if __name__ == '__main__':
    raise SystemExit(main())
