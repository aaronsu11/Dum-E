"""One transactional observation/action exchange per client, thread-safe task updates."""
import threading
from policy.backends.lerobot.backend import LeRobotPolicyBackend


class SerializedLeRobotPolicyBackend(LeRobotPolicyBackend):
    def __init__(self, **kwargs):
        if kwargs.get('host','127.0.0.1') not in ('localhost','127.0.0.1'):
            raise ValueError('Async pickle transport must stay on loopback')
        self._exchange_lock=threading.RLock()
        self._task_lock=threading.Lock()
        super().__init__(**kwargs)
        if self._device not in ('cuda','cuda:0') or self._actions_per_chunk!=16:
            raise ValueError('Async inference requires CUDA and 16-action chunks')

    def prepare_execution(self, observation, instruction, *, deadline_s):
        """Warm the model, then install the measured deadline under the exchange lock."""
        with self._exchange_lock:
            self.get_action(observation, instruction)
            self._session.configure_requests(deadline_s=deadline_s, max_attempts=1)

    def set_task(self, instruction):
        if not isinstance(instruction,str) or not instruction.strip():
            raise ValueError('Instruction must be nonempty')
        with self._task_lock:self._language_instruction=instruction

    def set_lang_instruction(self, instruction):
        self.set_task(instruction)

    def get_action(self, observation_dict, lang=None):
        with self._task_lock:instruction=lang or self._language_instruction
        with self._exchange_lock:
            return super().get_action(observation_dict,instruction)

    def reset(self):
        with self._exchange_lock:return super().reset()
