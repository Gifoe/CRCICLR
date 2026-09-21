"""Memory-gated two-cell scheduler. Never retries terminal or failed jobs."""
import json
import os
import subprocess
import time
from pathlib import Path

CODE=Path(__file__).parent
RUNTIME=Path(r'D:\nips-temp\TotalP\P1\protected_arbitration_reliability_runtime')
ORDER=[(m,t,f) for m in ('EEGNet','EEGConformer') for t in ('OpenBMI_MI','OpenBMI_SSVEP') for f in range(5)]
PREFIX='PERSIST_EEG_PROTECTED_ARBITRATION_RELIABILITY_'

def ps(script):
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,check=True)
    return json.loads(result.stdout) if result.stdout.strip() else None

def name(cell):
    m,t,f=cell
    if cell==('EEGNet','OpenBMI_MI',0):return PREFIX+'PROBE_EEGNET_MI_F0_V1'
    if cell==('EEGNet','OpenBMI_MI',1):return PREFIX+'EEGNET_MI_F1_V1'
    return PREFIX+f'{m.upper()}_{t.upper()}_F{f}_V2'

def status():
    return ps("$ErrorActionPreference='Stop'; $jobs=@(Get-ScheduledTask | Where-Object {$_.TaskName -like '"+PREFIX+"*'} | Select-Object TaskName,State); $free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB; @{free_gb=$free; tasks=$jobs}|ConvertTo-Json -Depth 5 -Compress")

def launch(cell):
    m,t,f=cell;task=name(cell)
    script=f'''$ErrorActionPreference='Stop'
$name='{task}'
$previous=Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
if($previous -and $previous.State -eq 'Running'){{throw 'Already running'}}
$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{CODE / 'run_scheduled_queue.ps1'}" -Mode cell -Model {m} -Task {t} -Fold {f}' -WorkingDirectory '{CODE}'
$principal=New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::FromHours(24)) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $name
@{{launched=$name}}|ConvertTo-Json -Compress
'''
    return ps(script)

def write_state(value):
    RUNTIME.mkdir(parents=True,exist_ok=True)
    target=RUNTIME/'supervisor_status.json';tmp=target.with_suffix('.part')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,target)

def main():
    attempted=set();pending={};iterations=0
    previous=RUNTIME/'supervisor_status.json'
    if previous.exists():
        prior=json.loads(previous.read_text(encoding='utf-8'))
        attempted.update(tuple(c) for c in prior.get('attempted',[]))
    while True:
        snap=status();jobs={j['TaskName']:j['State'] for j in snap['tasks']}
        active=[c for c in ORDER if jobs.get(name(c))==4 or jobs.get(name(c))=='Running']
        attempted.update(active);complete=[];failures=[]
        for c in ORDER:
            path=RUNTIME/'cells'/c[0].lower()/c[1].lower()/f'fold{c[2]}_seed0.json'
            if path.exists():
                data=json.loads(path.read_text(encoding='utf-8'))
                if data.get('status')=='COMPLETE' and data.get('implementation_revision')=='protocol_repair_v2':complete.append(c)
                else:failures.append({'cell':c,'reason':data.get('reason','invalid terminal implementation/status')})
            elif c in attempted and c not in active and time.time()-pending.get(c,0)>60:
                failures.append({'cell':c,'reason':'scheduled task stopped without terminal JSON; inspect log before any retry'})
        state={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'completed':len(complete),'total':20,'running':active,'free_gb':snap['free_gb'],'failures':failures,'status':'RUNNING' if active else 'WAITING_FOR_RESOURCES','attempted':sorted(attempted)}
        if failures:
            state['status']='NEEDS_ENGINEERING_REVIEW';write_state(state);return
        if len(complete)==20 and not active:
            state['status']='READY_FOR_AGGREGATION';write_state(state);return
        if len(active)<2 and snap['free_gb']>=40:
            candidates=[c for c in ORDER if c not in complete and c not in active and c not in attempted]
            if candidates:
                # Complete one model/task block before moving to the next.
                first=candidates[0]
                if not active or active[0][:2]==first[:2]:
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
                    if int(gpu.stdout.strip().splitlines()[0])>=16000:
                        launch(first);attempted.add(first);pending[first]=time.time();state['just_launched']=first;state['attempted']=sorted(attempted)
        write_state(state);iterations+=1;time.sleep(30)

if __name__=='__main__':
    try:main()
    except Exception as error:
        path=RUNTIME/'supervisor_status.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        state.update(status='SUPERVISOR_ERROR',reason=f'{type(error).__name__}: {error}')
        write_state(state)
        raise
