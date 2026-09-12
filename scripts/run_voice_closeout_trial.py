"""One approved 128-second physical voice closeout trial through the guarded runner.

The voice model selects the task. Physical execution uses the fixed approved
initial banana instruction, 2560-action budget and original stop/clamp composition.
"""
import argparse
import asyncio
from contextlib import suppress
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import importlib.util
_spec=importlib.util.spec_from_file_location('closeout_physical_base',Path(__file__).with_name('run_async_physical_trial.py'))
physical=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(physical)
from policy_guard.replay_contract import write_evidence,read_json,now
from shared import Message,MessageType,TaskStatus
from shared.task_manager import get_shared_memory_task_manager_from_env
from shared.message_broker import get_shared_memory_broker_from_env
from shared.fleet_manager import get_shared_memory_fleet_manager_from_env

ROBOT_ID='physical-test-arm'
physical.PROTOCOL={**physical.PROTOCOL,'chunks':160,'trigger':'one voice/MCP closeout trial','retarget_allowed':True,'retarget_objects':['banana','apple'],'nominal_action_duration_s':128,'active_time_limit_s':140}
physical.SOURCES=(*physical.SOURCES,'scripts/run_voice_closeout_trial.py','scripts/voice_trial_control.py','mcp_server.py','pipecat_server.py')


def accepts_instruction(text):
    text=text.lower() if isinstance(text,str) else ''
    if any(word in text for word in ("don't",'do not','cancel','stop')):return False
    return 'banana' in text and any(word in text for word in ('pick','grab'))


async def serve(workspace):
    tm=get_shared_memory_task_manager_from_env();broker=get_shared_memory_broker_from_env()
    fleet=get_shared_memory_fleet_manager_from_env()
    if not all((tm,broker,fleet)):raise RuntimeError('Live voice stack bindings required')
    approval=read_json(workspace/'approval.json')
    physical.validate_approval(approval,await asyncio.to_thread(physical.snapshot))
    physical.gate.require(sys.stdin.isatty(),'Interactive stop terminal required')
    physical.gate.require(not (workspace/'voice-dispatch.json').exists(),'Existing voice attempt must be preserved')
    await fleet.register_robot(ROBOT_ID,'Physical Test Arm',{'physical':True,'managed_trial':True,'ready_for_task':True,'scope':'one approved 128-second closeout trial; banana/apple retarget only'})
    await fleet.set_enabled(ROBOT_ID,True)
    proc=None;controls=None;task_id=None;cancelled=False;control_events=[]
    runtime_monitor=None;command_count=0
    async def publish(kind,data):
        await broker.publish(Message(message_type=kind,task_id=task_id,timestamp=datetime.now(),data=data))
    async def control_loop():
        nonlocal cancelled,command_count
        async for message in broker.subscribe(message_types=[MessageType.STATUS_UPDATE],task_id=task_id):
            data=message.data
            if not isinstance(data,dict) or data.get('source')!='mcp_server':continue
            control_events.append({'at':now(),'data':data})
            if data.get('status')=='cancelled':
                cancelled=True
                if proc is not None and proc.returncode is None:proc.send_signal(signal.SIGTERM)
            elif data.get('action')=='retarget':
                from scripts.voice_trial_control import canonical_instruction
                try:
                    instruction=canonical_instruction(data.get('instruction'))
                    if not (workspace/'runtime'/'active.json').exists():
                        raise ValueError('Inference is not active yet; wait for the ready message.')
                    command_count+=1
                    write_evidence(workspace/'controls',f'{command_count:04d}.json',{'requested_at':now(),'instruction':instruction})
                except ValueError as exc:
                    await publish(MessageType.TASK_PROGRESS,{'message':'Retarget rejected: '+str(exc)})
    async def relay_runtime():
        seen=set()
        while True:
            for path in sorted((workspace/'runtime').glob('*.json')):
                if path.name in seen:continue
                event=read_json(path);seen.add(path.name)
                await publish(MessageType.TASK_PROGRESS,{'message':event['message']})
            await asyncio.sleep(.02)
    print('READY: Physical Test Arm; one 128-second closeout trial. Banana/apple retarget enabled.',flush=True)
    try:
        async with asyncio.timeout(600):
            async for message in broker.subscribe(message_types=[MessageType.TASK_CREATED]):
                data=message.data
                if not isinstance(data,dict) or data.get('source')!='mcp_server' or data.get('robot_id')!=ROBOT_ID:continue
                task_id=message.task_id
                if not accepts_instruction(data.get('instruction')):
                    await tm.update_task(task_id,TaskStatus.FAILED,'Only the approved banana pick is enabled.')
                    await publish(MessageType.TASK_FAILED,{'error':'Only the approved banana pick is enabled; no motion occurred.'})
                    task_id=None;continue
                if not await tm.claim_task(task_id,'voice-physical-trial'):continue
                task=await tm.get_task(task_id)
                if task.status==TaskStatus.CANCELLED:continue
                await fleet.register_robot(ROBOT_ID,metadata={'physical':True,'managed_trial':True,'ready_for_task':False})
                await fleet.set_enabled(ROBOT_ID,False)
                write_evidence(workspace,'voice-dispatch.json',{'at':now(),'task_id':task_id,'instruction':data['instruction'],'robot_id':ROBOT_ID,'protocol':physical.PROTOCOL})
                controls=asyncio.create_task(control_loop())
                runtime_monitor=asyncio.create_task(relay_runtime())
                await asyncio.sleep(0)
                if cancelled:
                    await publish(MessageType.TASK_FAILED,{"error":"Trial cancelled before motion."})
                    break
                await publish(MessageType.TASK_STARTED,{'source':ROBOT_ID,'instruction':data['instruction']})
                # Inherit the PTY so the child keeps its own main-thread signal
                # handlers, raw hardware stop latch and reset-inclusive guards.
                proc=await asyncio.create_subprocess_exec(sys.executable,__file__,'execute','--workspace',str(workspace))
                if cancelled and proc.returncode is None:proc.send_signal(signal.SIGTERM)
                code=await proc.wait()
                result=read_json(workspace/'live-run.json') if (workspace/'live-run.json').exists() else {}
                task=await tm.get_task(task_id)
                cancelled=cancelled or task.status==TaskStatus.CANCELLED
                if code==0 and result.get('status')=='awaiting_operator_observation' and not cancelled:
                    await tm.update_task(task_id,TaskStatus.COMPLETED,'Bounded motion finished; operator outcome pending.')
                    await publish(MessageType.TASK_COMPLETED,{'result':'Physical trial motion finished. Waiting for Aaron to confirm the outcome.'})
                else:
                    reason=result.get('stop_reason') or result.get('error') or 'Physical trial stopped before completion.'
                    if not cancelled:await tm.update_task(task_id,TaskStatus.FAILED,reason)
                    await publish(MessageType.TASK_FAILED,{'error':reason})
                write_evidence(workspace,'voice-execution.json',{'at':now(),'task_id':task_id,'returncode':code,'cancelled':cancelled,'controls':control_events,'runner_status':result.get('status'),'execution':'fixed guarded runner; voice LLM/MCP live; GPU inference and controller real','operator_observation':'pending'})
                break
    finally:
        if proc is not None and proc.returncode is None:
            proc.send_signal(signal.SIGTERM)
            await proc.wait()
        for job in (controls,runtime_monitor):
            if job is not None:
                job.cancel()
                with suppress(asyncio.CancelledError):await job
        await fleet.register_robot(ROBOT_ID,metadata={'physical':True,'managed_trial':True,'ready_for_task':False})
        await fleet.set_enabled(ROBOT_ID,False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['serve','execute','snapshot'])
    parser.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='snapshot':print(json.dumps(physical.snapshot()))
    elif args.command=='execute':
        from scripts.voice_trial_control import install_controlled_pick
        install_controlled_pick(args.workspace)
        raise SystemExit(physical.run(args.workspace))
    else:asyncio.run(serve(args.workspace))
