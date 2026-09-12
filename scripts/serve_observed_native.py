"""Native ZMQ serving with shared off/lightweight chunk telemetry."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_guard.chunk_observer import ChunkObserver, MODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--observer-mode', choices=MODES, default='lightweight')
    parser.add_argument('--cpu-threads', type=int, default=1)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer
    from scripts.replay_groot_native import pinned_native_cache
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(args.cpu_threads)
    with pinned_native_cache():
        policy = Gr00tPolicy(EmbodimentTag.NEW_EMBODIMENT, str(args.model_path.resolve()), device='cuda:0', strict=True)
    observer = ChunkObserver(args.observer_mode, synchronize=torch.cuda.synchronize)
    server = PolicyServer(policy, host=args.host, port=args.port)
    def get_action(*a, **kw):
        try:
            return observer.run(policy.model, policy.get_action, *a, **kw)
        finally:
            print(json.dumps({'backend': 'native', **observer.last}), flush=True)
    server.register_endpoint('get_action', get_action)
    server.run()


if __name__ == '__main__':
    main()
