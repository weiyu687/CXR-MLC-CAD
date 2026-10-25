"""Verify resource gating cannot silently time or reject candidates while busy."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from tools.gpu_idle import query_gpu_load,wait_for_idle_gpu


class Clock:
    def __init__(self):self.value=0;self.sleeps=[]
    def now(self):return self.value
    def sleep(self,seconds):self.sleeps.append(seconds);self.value+=seconds


def must_reject(function,kind):
    try:function()
    except kind:return
    raise AssertionError('Invalid observation or busy GPU accepted')


def main():
    calls=[]
    def runner(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout='0, 7, 2345\n',stderr='')
    assert query_gpu_load(runner=runner)=={'gpu_index':0,'utilization_percent':7.,'memory_free_mib':2345.}
    assert '--id=0' in calls[0] and '--format=csv,noheader,nounits' in calls[0]
    for value in ['0, N/A, 2','0, 101, 2','1, 7, 2','0, nan, 2','0, 7, -1','0, 7, 2\n1, 0, 3']:
        must_reject(lambda:query_gpu_load(runner=lambda *a,**k:SimpleNamespace(returncode=0,stdout=value,stderr='')),RuntimeError)
    must_reject(lambda:query_gpu_load(runner=lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='unavailable')),RuntimeError)
    for options in [{'max_utilization':101},{'consecutive':1},{'consecutive':2.5},
                    {'poll_seconds':float('nan')},{'max_wait_seconds':float('inf')}]:
        must_reject(lambda:wait_for_idle_gpu(**options),ValueError)
    with tempfile.TemporaryDirectory() as temporary:
        clock=Clock();values=iter([95,8,90,8,9,6]);notifications=[];path=Path(temporary)/'observations.jsonl'
        result=wait_for_idle_gpu(path,query=lambda:{'utilization_percent':next(values)},
                                 sleeper=clock.sleep,now=clock.now,on_wait=notifications.append)
        assert result['passed'] and result['observations']==6 and result['wait_seconds']==10
        assert [s['utilization_percent'] for s in result['final_samples']]==[8,9,6]
        assert len(clock.sleeps)==5 and notifications[0]['stage']=='WAITING_FOR_LOW_GPU_LOAD'
        assert len([json.loads(x) for x in path.read_text().splitlines()])==6
        must_reject(lambda:wait_for_idle_gpu(path),FileExistsError)
        clock=Clock()
        must_reject(lambda:wait_for_idle_gpu(query=lambda:{'utilization_percent':90},
                    max_wait_seconds=3,sleeper=clock.sleep,now=clock.now),TimeoutError)
        must_reject(lambda:wait_for_idle_gpu(query=lambda:{'utilization_percent':float('nan')},
                    sleeper=clock.sleep,now=clock.now),RuntimeError)
    print('PASS: low-load streak, busy reset, observation log, unsupported counters, timeout and no false idle acceptance')


if __name__=='__main__':main()
