"""Bounded file-based control relay for the main-thread guarded physical runner."""
import re
import threading
import time
from pathlib import Path
from policy_guard.replay_contract import read_json,write_evidence,now


def canonical_instruction(text):
    if not isinstance(text,str):raise ValueError('Instruction must name banana or apple.')
    words=set(re.findall(r"[a-z']+",text.lower()))
    if words & {'not',"don't",'stop','cancel'}:
        raise ValueError('Use cancel_task to stop; negative retarget instructions are not accepted.')
    objects=words & {'banana','apple'}
    if len(objects)!=1:raise ValueError('Name exactly one target: banana or apple.')
    return 'Grab '+next(iter(objects))+' and put it on the plate'


class ControlRelay:
    def __init__(self,skill,workspace,*,clock=time.monotonic):
        self.skill=skill;self.workspace=Path(workspace);self.done=threading.Event()
        self.seen=set();self.active_announced=False
        self.clock=clock;self.active_since=None
        self.thread=threading.Thread(target=self._loop,name='voice-control-relay',daemon=True)
    def poll(self):
        if self.skill.active and self.active_since is None:self.active_since=self.clock()
        if self.active_since is not None and self.clock()-self.active_since>=140:
            self.skill.stop('Closeout active-time limit reached');self.done.set();return
        if self.skill.active and not self.active_announced:
            write_evidence(self.workspace/'runtime','active.json',{
                'at':now(),'message':'Inference is active. You may now change the target to the apple.'})
            self.active_announced=True
        for path in sorted((self.workspace/'controls').glob('*.json')):
            if path.name in self.seen:continue
            command=read_json(path)
            try:
                instruction=canonical_instruction(command.get('instruction'))
                self.skill.set_task(instruction)
                event={'at':now(),'requested_at':command['requested_at'],'status':'applied',
                       'instruction':instruction,'message':'Retargeted the running pick to the '+('apple' if 'apple' in instruction else 'banana')+'.'}
            except Exception as exc:
                event={'at':now(),'status':'rejected','message':'Could not retarget: '+str(exc)}
            write_evidence(self.workspace/'runtime','ack-'+path.name,event)
            self.seen.add(path.name)
    def _loop(self):
        try:
            while not self.done.is_set():self.poll();self.done.wait(.02)
        except BaseException as exc:
            self.skill.stop('Control relay failed: '+str(exc))
    def start(self):self.thread.start()
    def close(self):self.done.set();self.thread.join(timeout=2)


def install_controlled_pick(workspace):
    # Installed only in the isolated physical child before the guarded runner
    # imports AsyncPickSkill. Production scheduler/dispatch behavior is unchanged.
    import embodiment.so_arm10x.async_pick as driver
    original=driver.AsyncPickSkill
    class ControlledPick(original):
        def run(self,*args,**kwargs):
            relay=ControlRelay(self,workspace);relay.start()
            try:return super().run(*args,**kwargs)
            finally:relay.close()
    driver.AsyncPickSkill=ControlledPick
