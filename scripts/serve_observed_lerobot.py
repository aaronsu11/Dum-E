"""LeRobot serving with shared off/lightweight telemetry; no parity attestation.

All ordinary entrypoint/SAFE-01 preflight checks remain. This explicitly separate
mode cannot satisfy the exhaustive Phase 7 physical-run attestation contract.
"""
import json
import os
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'docker/lerobot-policy')]
from policy_guard.chunk_observer import ChunkObserver


def main():
    if os.getenv('DUME_PARITY_ATTESTATION_PATH'):
        raise RuntimeError('Lightweight telemetry cannot replace exhaustive parity attestation; use the standard entrypoint for that contract')
    mode = os.environ.get('DUME_CHUNK_OBSERVER', 'lightweight')
    # Validate before importing/starting the server.
    ChunkObserver(mode)
    import torch
    import entrypoint
    class ObservedServer(entrypoint.DumEGrootPolicyServer):
        def _predict_action_chunk(self, observation):
            if getattr(self, '_parity_attestor', None) is not None:
                raise RuntimeError('Exhaustive attestation cannot be bypassed')
            observer = ChunkObserver(mode, synchronize=torch.cuda.synchronize)
            try:
                return observer.run(self.policy._groot_model, self._predict_action_chunk_impl, observation)
            finally:
                self.logger.info('Chunk telemetry | %s', json.dumps({'backend': 'lerobot', **observer.last}))
    entrypoint.DumEGrootPolicyServer = ObservedServer
    return entrypoint.main()


if __name__ == '__main__':
    raise SystemExit(main())
