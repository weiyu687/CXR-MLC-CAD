"""Observe GPU load before timing; never terminate or modify other processes."""
import json
import math
from pathlib import Path
import subprocess
import time


def query_gpu_load(index=0,runner=None):
    command=['nvidia-smi','--id='+str(index),
             '--query-gpu=index,utilization.gpu,memory.free','--format=csv,noheader,nounits']
    try:
        result=(runner or subprocess.run)(command,capture_output=True,text=True,timeout=10)
    except (OSError,subprocess.TimeoutExpired) as error:
        raise RuntimeError('GPU load observation unavailable; preserve timing candidates') from error
    if result.returncode:
        raise RuntimeError('GPU load observation failed: '+result.stderr.strip()[:500])
    rows=result.stdout.strip().splitlines()
    try:
        if len(rows)!=1:raise ValueError('Expected one GPU')
        ordinal,utilization,free_memory=map(float,rows[0].split(','))
        if ordinal!=index or not all(math.isfinite(v) for v in (utilization,free_memory)):
            raise ValueError('Invalid GPU observation')
        if not 0<=utilization<=100 or free_memory<0:raise ValueError('Invalid GPU counters')
    except ValueError as error:
        raise RuntimeError('GPU utilization counters are unavailable or invalid') from error
    return {'gpu_index':index,'utilization_percent':utilization,'memory_free_mib':free_memory}


def wait_for_idle_gpu(log_path=None,index=0,max_utilization=20,consecutive=3,poll_seconds=2,
                      max_wait_seconds=None,query=None,sleeper=None,now=None,on_wait=None):
    """Require consecutive low-load samples before launching a timed subprocess.

    This preflight reduces known background contention. It does not guarantee
    exclusive GPU ownership during the subsequent benchmark or certify T4 cost.
    """
    if (not math.isfinite(max_utilization) or not 0<=max_utilization<=100
            or type(consecutive) is not int or consecutive<2
            or not math.isfinite(poll_seconds) or poll_seconds<=0):
        raise ValueError('Invalid GPU idle policy')
    if max_wait_seconds is not None and (not math.isfinite(max_wait_seconds) or max_wait_seconds<=0):
        raise ValueError('Invalid idle wait budget')
    query=query or (lambda:query_gpu_load(index));sleeper=sleeper or time.sleep;now=now or time.monotonic
    start=now();streak=[];count=0;last_notification=None
    path=Path(log_path) if log_path is not None else None
    if path is not None:
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError('Existing GPU observations preserved: '+str(path))
    while True:
        elapsed=now()-start
        if max_wait_seconds is not None and elapsed>=max_wait_seconds:
            raise TimeoutError('GPU remained busy; preserve candidates without latency-based downgrading')
        sample=query();utilization=sample['utilization_percent']
        if not math.isfinite(utilization) or not 0<=utilization<=100:
            raise RuntimeError('Invalid GPU utilization observation')
        sample={**sample,'elapsed_seconds':elapsed}
        count+=1
        if path is not None:
            with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(sample)+'\n')
        if utilization<=max_utilization:
            streak.append(sample)
        else:
            streak=[]
            if on_wait and (last_notification is None or elapsed-last_notification>=30):
                on_wait({'stage':'WAITING_FOR_LOW_GPU_LOAD',**sample,'maximum_percent':max_utilization})
                last_notification=elapsed
        if len(streak)>=consecutive:
            return {'passed':True,'gpu_index':index,'maximum_utilization_percent':max_utilization,
                    'consecutive_samples':consecutive,'poll_seconds':poll_seconds,'observations':count,
                    'wait_seconds':now()-start,'final_samples':streak[-consecutive:],
                    'observation_log':str(path) if path is not None else None,
                    'scope':'Low-load preflight only; exclusive GPU access and T4 performance are not certified'}
        sleeper(poll_seconds)
