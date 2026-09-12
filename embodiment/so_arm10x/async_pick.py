"""Opt-in async pick driver; the supplied controller keeps hardware ownership."""
import copy
import json
import math
from pathlib import Path
import threading
import time
import uuid
from policy.lerobot.async_chunks import AsyncChunks,AsyncSettings,ChunkPending,InferenceStopped
from policy_guard.replay_contract import JOINT_ORDER,sha256_file


def load_settings(path):
    data=json.loads(Path(path).read_text())
    if data.get('kind')!='async_request_latency' or data.get('status')!='complete' or data.get('sample_count',0)<100:
        raise ValueError('Completed measured request-latency evidence required')
    if data.get('observer_mode')!='lightweight' or data.get('policy_device')!='cuda':
        raise ValueError('Lightweight CUDA measurement required')
    samples=data.get('samples_s',[])
    if len(samples)!=data['sample_count'] or not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in samples):
        raise ValueError('Complete finite request samples required')
    measured=sorted(samples)[math.ceil(.99*(len(samples)-1))]
    if measured!=data['request_p99_s']:
        raise ValueError('Recorded p99 does not match request samples')
    expected_sources={'scripts/serve_observed_lerobot.py','policy_guard/chunk_observer.py',
                      'policy/lerobot/serialized_backend.py','docker/lerobot-policy/server.py'}
    if set(data.get('source_files',{}))!=expected_sources:
        raise ValueError('Exact measured source identities required')
    root=Path(__file__).resolve().parents[2]
    for name,digest in data['source_files'].items():
        if sha256_file(root/name)!=digest:raise ValueError('Latency configuration is stale: '+name)
    settings=AsyncSettings(**data['settings'])
    if settings.request_p99_s!=data['request_p99_s']:raise ValueError('Latency settings mismatch')
    return settings


class AsyncPickSkill:
    def __init__(self,controller,policy,settings,*,clock=time.monotonic,sleep=time.sleep,on_failure=None,trace_directory=None):
        self.controller,self.policy,self.settings=controller,policy,settings
        self.clock,self.sleep=clock,sleep
        self.on_failure=on_failure
        self.trace_directory=Path(trace_directory) if trace_directory else None
        self._motion_lock=threading.Lock();self._state_lock=threading.RLock()
        self._active=None;self._fault=None;self.last_events=[]

    @property
    def active(self):
        with self._state_lock:return self._active is not None

    def check(self):
        with self._state_lock:
            if self._fault:raise InferenceStopped(self._fault)
            latch=getattr(self.controller,'stop',None)
            if latch is not None:latch.check()

    def stop(self,reason='Operator stopped async picking'):
        # Latch raw hardware dispatch before waiting for a scheduler lock held
        # by an in-flight write. Existing bus guards then refuse retries/enables.
        latch=getattr(self.controller,'stop',None)
        if latch is not None:latch.trip(reason)
        with self._state_lock:
            self._fault=self._fault or reason
            if self._active is not None:self._active.stop(reason)
        # Closing the channel cancels an outstanding RPC; it never touches robot hardware.
        self.policy.close()

    def set_task(self,instruction):
        with self._state_lock:
            self.check()
            if self._active is None:raise RuntimeError('No async pick is active to retarget')
            self.policy.set_task(instruction)
            self._active.set_task(instruction)

    def other_motion(self,call,*args,**kwargs):
        if not self._motion_lock.acquire(blocking=False):raise RuntimeError('A pick is already controlling the arm')
        try:self.check();return call(*args,**kwargs)
        finally:self._motion_lock.release()

    def run(self,item=None,pose='initial',actions_to_execute=10,action_horizon=16,language_instruction=None):
        if action_horizon!=16 or type(actions_to_execute) is not int or actions_to_execute<1:
            raise ValueError('Positive fixed budget and 16 actions required')
        if not self._motion_lock.acquire(blocking=False):raise RuntimeError('A pick is already active')
        scheduler=None
        state_samples=[]
        from contextlib import ExitStack
        protection=ExitStack()
        if getattr(self.controller,'stop',None) is not None:
            from scripts.run_checkpoint_sanity import armed_stop
            protection.enter_context(armed_stop(self.controller.stop))
        try:
            self.check()
            instruction=language_instruction or self.policy.language_instruction
            # Slow load/handshake occurs in this worker before reset or timed execution.
            if not self.policy._handshaken:self.policy._handshake()
            # Warm CUDA kernels before applying the measured *warm* deadline.
            # This prediction is never sent to the arm.
            self.policy.get_action(copy.deepcopy(self.controller.get_observation()),instruction)
            self.policy._session._deadline_s=self.settings.request_deadline_s
            self.policy._session._max_attempts=1
            self.check()
            if pose=='initial':
                self.controller.move_to_initial_pose();self.check();self.controller.move_to_ready_pose()
            elif pose!='resume':raise ValueError('Unknown reset mode')
            scheduler=AsyncChunks(self.policy.get_action,JOINT_ORDER,self.settings,instruction,clock=self.clock)
            with self._state_lock:self._active=scheduler
            step=0;next_tick=self.clock()
            while step<actions_to_execute*16:
                self.check();scheduler.poll(step)
                if scheduler.wants_observation(step):
                    observation=copy.deepcopy(self.controller.get_observation())
                    scheduler.request(observation,step)
                try:command=scheduler.next_action(step)
                except ChunkPending:
                    self.sleep(min(.005,self.settings.period_s));next_tick=self.clock();continue
                # Deadline and epoch are rechecked while holding the dispatch lock.
                try:sent=scheduler.dispatch(command,self.controller.set_target_state)
                except ChunkPending:continue
                if self.trace_directory is not None:
                    state=self.controller.get_current_state()
                    state_samples.append({'step':step,'at':self.clock(),'state':list(map(float,state))})
                step+=1;next_tick+=self.settings.period_s
                delay=next_tick-self.clock()
                if delay>0:self.sleep(delay)
                else:next_tick=self.clock()  # Never catch up with a burst of writes.
            return self.controller.get_current_images()
        except BaseException as exc:
            with self._state_lock:self._fault=str(exc) or type(exc).__name__
            if scheduler is not None:scheduler.stop(self._fault)
            latch=getattr(self.controller,'stop',None)
            if latch is not None:latch.trip(self._fault)
            self.policy.close()
            if self.on_failure is not None:self.on_failure(self._fault)
            raise
        finally:
            if scheduler is not None:
                self.last_events=list(scheduler.events);scheduler.close()
            with self._state_lock:self._active=None
            try:
                if self.trace_directory is not None:
                    from policy_guard.replay_contract import write_evidence, now
                    write_evidence(self.trace_directory, 'async-trace-'+uuid.uuid4().hex+'.json', {
                        'recorded_at':now(), 'settings':vars(self.settings), 'events':self.last_events,
                        'state_samples':state_samples, 'fault':self._fault,
                        'status':'failed' if self._fault else 'complete'})
            finally:
                protection.close()
                self._motion_lock.release()
