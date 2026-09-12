"""Bounded async action queue. This module never owns or reads robot hardware."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
import math
import time


class InferenceStopped(RuntimeError):
    """Latched fault: no later action may be dispatched by this scheduler."""


class ChunkPending(Exception):
    """No action yet; only allowed while awaiting the first/re-targeted chunk."""


@dataclass(frozen=True)
class AsyncSettings:
    request_p99_s: float
    period_s: float = .05
    actions_per_chunk: int = 16
    chunk_size_threshold: float = .5
    margin_s: float = .1
    aggregate_fn_name: str = 'weighted_average'

    def __post_init__(self):
        if self.actions_per_chunk != 16 or self.chunk_size_threshold != .5 or self.period_s != .05:
            raise ValueError('Reviewed async protocol requires 16 actions, 0.5 threshold and 50ms pacing')
        if not all(math.isfinite(v) and v > 0 for v in (self.request_p99_s,self.margin_s)):
            raise ValueError('Measured p99 and positive margin required')
        if self.request_deadline_s >= self.prefetch_remaining * self.period_s:
            raise ValueError('Measured request latency plus margin does not fit the half-chunk overlap')
        if self.aggregate_fn_name != 'weighted_average':
            raise ValueError('Explicit weighted_average aggregation required')

    @property
    def prefetch_remaining(self):return int(self.actions_per_chunk*self.chunk_size_threshold)
    @property
    def request_deadline_s(self):return self.request_p99_s+self.margin_s
    @property
    def max_action_age_s(self):return self.actions_per_chunk*self.period_s


@dataclass(frozen=True)
class Command:
    step: int
    epoch: int
    sampled_at: float
    target: dict
    previous: object = None
    raw_target: object = None


class AsyncChunks:
    def __init__(self, infer, joints, settings, instruction, *, executor=None, clock=time.monotonic):
        if not instruction or not isinstance(instruction,str):raise ValueError('Instruction required')
        self.infer,self.joints,self.settings,self.clock=infer,tuple(joints),settings,clock
        self._executor=executor or ThreadPoolExecutor(max_workers=1,thread_name_prefix='policy-inference')
        self._owns_executor=executor is None
        self._lock=RLock();self._epoch=0;self._instruction=instruction
        self._future=None;self._request=None;self._queue={};self._fault=None;self._closed=False;self._awaiting_initial=True
        self.events=[]

    def _check(self):
        if self._fault:raise InferenceStopped(self._fault)
        if self._closed:raise InferenceStopped('Inference session closed')

    def stop(self,reason):
        with self._lock:
            if self._fault is None:self._fault=str(reason) or 'Inference stopped'
            self._queue.clear();self.events.append({'event':'stop','at':self.clock(),'reason':self._fault})

    def set_task(self,instruction):
        if not isinstance(instruction,str) or not instruction.strip():raise ValueError('Instruction required')
        with self._lock:
            self._check();self._epoch+=1;self._instruction=instruction;self._queue.clear();self._awaiting_initial=True
            self.events.append({'event':'retarget','epoch':self._epoch,'at':self.clock()})
            # An old in-flight call remains the sole worker until it returns or
            # times out. Its result is discarded; no concurrent anchor mutation.

    def _fail(self,reason):
        self.stop(reason);raise InferenceStopped(self._fault)

    def poll(self,step):
        with self._lock:
            self._check()
            if self._future is None:return
            epoch,base,sampled=self._request
            if self.clock()-sampled > self.settings.request_deadline_s:
                self._fail('Policy request deadline exceeded')
            if not self._future.done():return
            future=self._future;self._future=None;self._request=None
            try:actions=future.result()
            except Exception as exc:self._fail('Policy server failed: '+str(exc))
            if epoch!=self._epoch:
                self.events.append({'event':'discard_old_epoch','epoch':epoch});return
            if not isinstance(actions,list) or len(actions)!=16:self._fail('Policy must return exactly 16 actions')
            for offset,target in enumerate(actions):
                if not isinstance(target,dict) or set(target)!=set(self.joints):self._fail('Action joint mapping changed')
                if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in target.values()):
                    self._fail('Policy returned a nonfinite or nonnumeric action')
                index=base+offset
                if index<step:continue
                values=dict(target);sample_time=sampled
                old=self._queue.get(index)
                if old is not None and self.clock()-old.sampled_at<=self.settings.max_action_age_s:
                    # Exact pinned upstream weighted_average formula.
                    values={k:.3*old.target[k]+.7*values[k] for k in self.joints}
                    sample_time=sampled
                self._queue[index]=Command(index,epoch,sample_time,values,old,dict(target))
            self._awaiting_initial=False
            self.events.append({'event':'chunk','epoch':epoch,'base':base,'received_at':self.clock(),'sampled_at':sampled})

    def wants_observation(self,step):
        with self._lock:
            self.poll(step)
            return self._future is None and sum(i>=step for i in self._queue)<=self.settings.prefetch_remaining

    def request(self,observation,step):
        with self._lock:
            self.poll(step)
            if self._future is not None:raise RuntimeError('Only one request may be in flight')
            epoch,task,sampled=self._epoch,self._instruction,self.clock()
            self._request=(epoch,step,sampled)
            self._future=self._executor.submit(self.infer,observation,task)

    def next_action(self,step):
        with self._lock:
            self.poll(step)
            command=self._queue.pop(step,None)
            if command is None:
                if self._future is not None and self._awaiting_initial:raise ChunkPending()
                self._fail('Action queue starved')
            self._validate_command(command)
            return command

    def _validate_command(self,command):
        self._check()
        if command.epoch!=self._epoch:raise ChunkPending('Retargeted action discarded before dispatch')
        if self.clock()-command.sampled_at>self.settings.max_action_age_s:self._fail('Action observation is stale')

    def _effective_target(self,command):
        raw=command.raw_target if command.raw_target is not None else command.target
        previous=command.previous
        if previous is None or self.clock()-previous.sampled_at>self.settings.max_action_age_s:
            return dict(raw)
        old=self._effective_target(previous)
        return {key:.3*old[key]+.7*raw[key] for key in self.joints}

    def dispatch(self,command,send):
        with self._lock:
            self.poll(command.step);self._validate_command(command)
            target=self._effective_target(command)
            result=send(target)
            # Match the controller/upstream clamp threshold. Subtracting and
            # re-adding present position can change a float without clipping it.
            if not isinstance(result,dict) or set(result)!=set(target) or any(
                isinstance(result[k],bool) or not isinstance(result[k],(int,float)) or
                not math.isfinite(result[k]) or not math.isclose(result[k],v,rel_tol=0.,abs_tol=1e-4)
                for k,v in target.items()
            ):
                self._fail('Controller clamped or changed an async target')
            self.events.append({'event':'action','step':command.step,'epoch':command.epoch,'at':self.clock(),'target':dict(target)})
            return result

    def close(self):
        with self._lock:
            self._closed=True;self._queue.clear()
            if self._future is not None:self._future.cancel()
        if self._owns_executor:self._executor.shutdown(wait=False,cancel_futures=True)
