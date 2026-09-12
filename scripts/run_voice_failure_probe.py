"""One voice-requested simulated inference failure through the real agent/MCP path.

No hardware controller or GPU model is constructed. Attach only to the existing
voice stack's shared memory; exit after the first test task has failed.
"""
import argparse
import asyncio
from contextlib import suppress
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from embodiment.so_arm10x.agent import SO10xRobotAgent,create_robot_tools,_agent_worker_loop
from embodiment.so_arm10x.async_pick import AsyncPickSkill
from policy.lerobot.async_chunks import AsyncSettings
from policy_guard.replay_contract import JOINT_ORDER,write_evidence,now
from shared import MessageType
from shared.task_manager import get_shared_memory_task_manager_from_env
from shared.message_broker import get_shared_memory_broker_from_env
from shared.fleet_manager import get_shared_memory_fleet_manager_from_env


class SimulatedController:
    id='voice-test-arm'
    def __init__(self):self.connected=False;self.actions=[];self.resets=[]
    def connect(self):self.connected=True
    def disconnect(self):self.connected=False
    def is_connected(self):return self.connected
    def get_current_images(self):return {'front':np.zeros((4,4,3),dtype=np.uint8)}
    def get_observation(self):return {'step':len(self.actions)}
    def move_to_initial_pose(self):self.resets.append('initial')
    def move_to_ready_pose(self):self.resets.append('ready')
    def set_target_state(self,target):self.actions.append(dict(target));return dict(target)


class SimulatedPolicy:
    language_instruction='simulated banana';_handshaken=True
    def __init__(self):self._session=SimpleNamespace();self.calls=0;self.closed=False
    def set_lang_instruction(self,text):self.language_instruction=text
    def set_task(self,text):self.language_instruction=text
    def get_action(self,observation,instruction):
        self.calls+=1;time.sleep(.04)
        if self.calls>=3:
            raise RuntimeError('Simulated inference server disconnected. This is the voice integration test; no hardware moved.')
        return [dict.fromkeys(JOINT_ORDER,.1) for _ in range(16)]
    def close(self):self.closed=True


async def main(workspace):
    tm=get_shared_memory_task_manager_from_env();broker=get_shared_memory_broker_from_env()
    fleet=get_shared_memory_fleet_manager_from_env()
    if not all((tm,broker,fleet)):raise RuntimeError('Existing voice stack shared-memory bindings required')
    if (workspace/'failure-probe.json').exists():raise RuntimeError('Use a new attempt workspace')
    controller=SimulatedController();policy=SimulatedPolicy()
    agent=SO10xRobotAgent(controller,policy,task_manager=tm,message_broker=broker)
    agent.async_pick=AsyncPickSkill(controller,policy,AsyncSettings(.15),on_failure=agent._notify_async_failure)
    agent._robot_tools=create_robot_tools(controller,policy,async_pick=agent.async_pick)
    pick=next(t for t in agent._robot_tools if t.tool_name=='start_pick')
    class ScriptedToolSelection:
        async def stream_async(self,instruction):
            await pick._tool_func(item='a simulated banana')
            yield {'message':{'role':'assistant','content':[{'text':'Unexpected simulation completion'}]}}
    async def get_model():return ScriptedToolSelection()
    agent._get_strands_agent=get_model
    await fleet.register_robot(controller.id,'Voice Test Robot',{'simulated':True,'purpose':'one failure notification test; no physical hardware'})
    worker=asyncio.create_task(_agent_worker_loop(agent,tm,broker,controller.id,'voice-failure-probe'))
    print('READY: Voice Test Robot (voice-test-arm), simulated hardware; one failure test.',flush=True)
    try:
        async with asyncio.timeout(600):
            async for message in broker.subscribe(message_types=[MessageType.TASK_FAILED]):
                if message.task_id!=agent._active_task_id:continue
                while controller.connected:await asyncio.sleep(.01)
                task=await tm.get_task(message.task_id)
                write_evidence(workspace,'failure-probe.json',{'recorded_at':now(),'status':'failure_reached_broker',
                    'task_id':message.task_id,'task_status':task.status.value,'error':message.data,
                    'policy_calls':policy.calls,'simulated_actions':len(controller.actions),
                    'resets':controller.resets,'policy_closed':policy.closed,'hardware':'simulated_no_robot_access',
                    'robot_model_tool_selection':'scripted; voice LLM and MCP are live','audible_observation':'pending'})
                print('DONE: Expected failure published through real agent task/broker path.',flush=True)
                break
    finally:
        if agent.async_pick.active:agent.async_pick.stop('Probe cleanup')
        worker.cancel()
        with suppress(asyncio.CancelledError):await worker
        await fleet.set_enabled(controller.id,False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True)
    asyncio.run(main(p.parse_args().workspace))
