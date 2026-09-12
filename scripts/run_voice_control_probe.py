"""One three-minute simulated task for live spoken retarget/cancel checks.

Uses the real agent worker, async scheduler, MCP and voice services. No hardware
controller or GPU model is constructed; robot-model tool selection is scripted.
"""
import argparse
import asyncio
from contextlib import suppress
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_voice_failure_probe import SimulatedController
from embodiment.so_arm10x.agent import SO10xRobotAgent,create_robot_tools,_agent_worker_loop
from embodiment.so_arm10x.async_pick import AsyncPickSkill
from policy.lerobot.async_chunks import AsyncSettings
from policy_guard.replay_contract import JOINT_ORDER,write_evidence,now
from shared import MessageType
from shared.task_manager import get_shared_memory_task_manager_from_env
from shared.message_broker import get_shared_memory_broker_from_env
from shared.fleet_manager import get_shared_memory_fleet_manager_from_env


class ControlPolicy:
    language_instruction='simulated banana';_handshaken=True
    def __init__(self):self._session=SimpleNamespace();self.calls=[];self.closed=False
    def set_lang_instruction(self,text):self.language_instruction=text
    def set_task(self,text):self.language_instruction=text
    def get_action(self,observation,instruction):
        self.calls.append({'at':time.monotonic(),'instruction':instruction})
        time.sleep(.04)
        value=.2 if 'apple' in instruction.lower() else .1
        return [dict.fromkeys(JOINT_ORDER,value) for _ in range(16)]
    def close(self):self.closed=True


async def main(workspace):
    tm=get_shared_memory_task_manager_from_env();broker=get_shared_memory_broker_from_env()
    fleet=get_shared_memory_fleet_manager_from_env()
    if not all((tm,broker,fleet)):raise RuntimeError('Existing voice stack shared memory required')
    if (workspace/'control-probe.json').exists():raise RuntimeError('Existing attempt must be preserved')
    workspace.mkdir(parents=True,exist_ok=True)
    controller=SimulatedController();policy=ControlPolicy()
    agent=SO10xRobotAgent(controller,policy,task_manager=tm,message_broker=broker)
    agent.async_pick=AsyncPickSkill(controller,policy,AsyncSettings(.15),on_failure=agent._notify_async_failure)
    run=agent.async_pick.run
    def bounded_run(*args,**kwargs):return run(*args,**dict(kwargs,actions_to_execute=225))
    agent.async_pick.run=bounded_run
    agent._robot_tools=create_robot_tools(controller,policy,async_pick=agent.async_pick)
    pick=next(t for t in agent._robot_tools if t.tool_name=='start_pick')
    class ScriptedToolSelection:
        async def stream_async(self,instruction):
            await pick._tool_func(item='a simulated banana')
            yield {'result':SimpleNamespace(message={'role':'assistant','content':[{'text':'Three-minute simulated control test completed.'}]})}
    async def get_model():return ScriptedToolSelection()
    agent._get_strands_agent=get_model
    await fleet.register_robot(controller.id,'Voice Test Robot',{'simulated':True,'purpose':'one 180-second spoken retarget/cancel test; no hardware'})
    await fleet.set_enabled(controller.id,True)
    messages=[];ticks=[]
    async def capture():
        async for message in broker.subscribe():
            if message.task_id and message.task_id==agent._active_task_id:
                entry={'at':time.monotonic(),'type':message.message_type.value,'task_id':message.task_id,'data':message.data}
                messages.append(entry)
                with (workspace/'live-events.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
    async def heartbeat():
        while True:
            if agent.async_pick.active:ticks.append(time.monotonic())
            await asyncio.sleep(.05)
    worker=asyncio.create_task(_agent_worker_loop(agent,tm,broker,controller.id,'voice-control-probe'))
    listener=asyncio.create_task(capture());pulse=asyncio.create_task(heartbeat())
    print('READY: Voice Test Robot (voice-test-arm); one simulated 180-second task, spoken retarget/cancel enabled.',flush=True)
    try:
        async with asyncio.timeout(600):
            while not agent._active_task_id:await asyncio.sleep(.05)
            print('ACTIVE task: '+agent._active_task_id,flush=True)
            while True:
                task=await tm.get_task(agent._active_task_id)
                if task.status.value in ('failed','cancelled','completed') and not controller.connected:break
                await asyncio.sleep(.05)
            await asyncio.sleep(.1)
            write_evidence(workspace,'control-probe.json',{'recorded_at':now(),'status':'completed_awaiting_voice_review',
                'task_id':task.task_id,'task_status':task.status.value,'policy_calls':policy.calls,
                'simulated_action_count':len(controller.actions),'apple_target_sent':any(a[JOINT_ORDER[0]]==.2 for a in controller.actions),
                'policy_closed':policy.closed,'resets':controller.resets,'scheduler_events':agent.async_pick.last_events,
                'messages':messages,'event_loop_heartbeat_s':ticks,'hardware':'simulated_no_robot_access',
                'model':'simulated 40ms inference; no GPU load','robot_model_tool_selection':'scripted; voice LLM and MCP live'})
            print('DONE: '+task.status.value+'; evidence saved.',flush=True)
    finally:
        if agent.async_pick.active:agent.async_pick.stop('Probe cleanup')
        for job in (worker,listener,pulse):job.cancel()
        for job in (worker,listener,pulse):
            with suppress(asyncio.CancelledError):await job
        await fleet.set_enabled(controller.id,False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True)
    asyncio.run(main(p.parse_args().workspace))
