#!/usr/bin/env python3
"""Finite isolated macOS worker. All work/cache/model state stays in --work.

No launch agents, containers, live configuration, service restarts or public STT.
Only owned direct child servers are stopped; no process-name kills.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

RUNTIME='e6d193e85e84e30f26e370c66973ce287b8a9d57'
VOX_REV='02eb0caeb9dafb554c17a72b93dbf40cd3736c31'


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--uv',type=Path,required=True)
    p.add_argument('--parakeet-model-dir',type=Path,required=True)
    p.add_argument('--prior-synthetic-dir',type=Path,help='optional earlier benchmark with manifest synthetic=true allowlist')
    a=p.parse_args(); work=a.work.resolve();source=a.source.resolve()
    if platform.system()!='Darwin' or platform.machine()!='arm64':p.error('Apple Silicon worker required')
    if not str(work).startswith('/Volumes/') or not str(source).startswith('/Volumes/'):
        p.error('external-volume placement required')
    if work.exists(): p.error('one-use work directory already exists; inspect its receipt instead of duplicating work')
    work.mkdir(parents=True,mode=0o700)
    env=dict(os.environ,PATH='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin')
    for key in ['PYTHONPATH','WIZARD_VOICE_CONFIG','WIZARD_STT_CONFIG','WIZARD_VOICE_RULES','WIZARD_STT_CORRECTIONS','WIZARD_VOICE_ENV','HERMES_ENV']:
        env.pop(key,None)
    for name,value in {'TMPDIR':'tmp','UV_CACHE_DIR':'uv-cache','UV_PYTHON_INSTALL_DIR':'python',
        'HF_HOME':'hf-cache','XDG_CACHE_HOME':'cache','PIP_CACHE_DIR':'pip-cache'}.items():
        path=work/value;path.mkdir();env[name]=str(path)
    env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    receipt={'status':'running','source_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        'machine':platform.machine(),'memory_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),
        'phases':[],'engines':{},'owned_processes':[],'tinfoil':'pending control-plane replay with existing read-only credential'}
    children=[]
    def save():
        (work/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    def run(name,argv,*,timeout=900,cwd=work,extra_env=None,check=True):
        started=time.monotonic();log=work/(name+'.log')
        print(name,flush=True)
        with log.open('w') as out:
            proc=subprocess.run([str(x) for x in argv],cwd=cwd,env=dict(env,**(extra_env or {})),stdout=out,stderr=subprocess.STDOUT,timeout=timeout,check=False)
        receipt['phases'].append({'name':name,'exit_code':proc.returncode,'seconds':time.monotonic()-started,'log':str(log)})
        save()
        if check and proc.returncode:raise RuntimeError(f'{name} failed; see phase log')
        return proc.returncode
    def get(url):
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url,timeout=3) as response:return json.load(response)
    def start(name,module,port,extra):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'refusing occupied candidate port {port}')
        log=(work/(name+'-server.log')).open('w')
        proc=subprocess.Popen([str(python),'-m','uvicorn',module,'--host','127.0.0.1','--port',str(port),'--no-access-log'],
            cwd=work,env=dict(env,**extra),stdout=log,stderr=subprocess.STDOUT)
        children.append((proc,log));receipt['owned_processes'].append({'pid':proc.pid,'name':name,'port':port});save()
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if proc.poll() is not None:raise RuntimeError(f'{name} exited before readiness')
            try:
                result=get(f'http://127.0.0.1:{port}/health')
                if result.get('status') in {'ready','ok'}:return proc,result
            except (OSError,ValueError):pass
            time.sleep(0.5)
        raise TimeoutError(f'{name} model readiness deadline')
    def stop_all():
        while children:
            proc,log=children.pop()
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=15)
            log.close()
        receipt['all_owned_children_reaped']=True;save()
    save()
    try:
        run('venv',[a.uv,'venv','--python','3.12',work/'venv'])
        python=work/'venv/bin/python'
        run('build',[a.uv,'build','--wheel','--out-dir',work/'dist',source])
        wheel=next((work/'dist').glob('*.whl'))
        receipt['wheel_sha256']=hashlib.sha256(wheel.read_bytes()).hexdigest();save()
        run('install',[a.uv,'pip','install','--python',python,str(wheel)+'[test,parakeet]',
            'voxmlx @ git+https://github.com/awni/voxmlx.git@'+RUNTIME],timeout=1200)
        run('freeze',[a.uv,'pip','freeze','--python',python])
        run('installed-proof',[python,'-c',
            'import providers,parakeet_adapter,voxtral_adapter,batch_audio,json;print(json.dumps({m.__name__:m.__file__ for m in [providers,parakeet_adapter,voxtral_adapter,batch_audio]},indent=2))'])
        run('suite',[python,'-m','pytest','-q',str(source/'tests')],timeout=600,cwd=source)
        run('synthetic',[python,source/'benchmarks/make_synthetic.py','--output',work/'corpus'],timeout=240)
        if a.prior_synthetic_dir:
            prior = json.loads((a.prior_synthetic_dir/'manifest.json').read_text())
            entries = prior.get('fixtures', prior.get('files', [])) if isinstance(prior,dict) else prior
            copied = []
            for name in ['synthetic.wav','silence.wav']:
                row = next((x for x in entries if x.get('file')==name),None)
                if not row or row.get('synthetic') is not True or not isinstance(row.get('reference'),str):
                    raise ValueError('earlier fixture is not explicitly synthetic with reference')
                target = work/'corpus'/('prior-'+name)
                shutil.copyfile(a.prior_synthetic_dir/name,target)
                if hashlib.sha256(target.read_bytes()).hexdigest() != row.get('wav_sha256'):
                    raise ValueError('earlier synthetic fixture hash changed')
                target.with_suffix('.txt').write_text(row['reference'])
                copied.append({'file':target.name,'audio_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'origin':str(a.prior_synthetic_dir/name)})
            receipt['earlier_nonprivate_fixtures']=copied;save()
        voxdir=work/'models/voxtral-mini-4b-realtime-6bit'
        download="from huggingface_hub import snapshot_download;import sys;print(snapshot_download('mlx-community/Voxtral-Mini-4B-Realtime-6bit',revision='"+VOX_REV+"',local_dir=sys.argv[1]))"
        if shutil.disk_usage(work).free < 15*2**30:raise RuntimeError('15 GiB external free-space floor')
        run('download-voxtral',[python,'-c',download,voxdir],timeout=1800)
        receipt['voxtral_model']={'repo':'mlx-community/Voxtral-Mini-4B-Realtime-6bit','revision':VOX_REV,'runtime_commit':RUNTIME};save()
        for engine,module,port,modelenv in [
            ('parakeet-local','parakeet_adapter:app',19865,{'WIZARD_PARAKEET_MODEL_DIR':str(a.parakeet_model_dir),'WIZARD_PARAKEET_THREADS':'4'}),
            ('voxtral-local','voxtral_adapter:app',19866,{'WIZARD_VOXTRAL_MODEL_DIR':str(voxdir)})]:
            try:
                proc,health=start(engine,module,port,dict(modelenv,HF_HUB_OFFLINE='1'))
                receipt['engines'][engine]={'status':'ready','health':health,'models':get(f'http://127.0.0.1:{port}/v1/models')};save()
                config=work/(engine+'.yaml')
                run('config-'+engine,[python,'-c',
                    'import yaml,sys;from pathlib import Path;r=yaml.safe_load(Path(sys.argv[1]).read_text());r["stt"]["active_profile"]=sys.argv[2];Path(sys.argv[3]).write_text(yaml.safe_dump(r))',
                    source/'examples/three-engines.yaml',engine,config])
                state=work/'state'/engine;state.mkdir(parents=True)
                broker,bh=start('broker-'+engine,'server:app',19864,{
                    'WIZARD_VOICE_CONFIG':str(config),'WIZARD_VOICE_DATA_DIR':str(state),
                    'WIZARD_VOICE_ENV':str(work/'absent.env'),'WIZARD_STT_CORRECTIONS':str(work/'absent.yaml'),
                    'WIZARD_VOICE_RETAIN_FAILED_AUDIO':'0','WIZARD_VOICE_PRIVATE_BUFFER_DIR':''})
                run('http-broker-'+engine,[python,source/'benchmarks/probe_broker.py','--url','http://127.0.0.1:19864',
                    '--corpus',work/'corpus','--output',work/('broker-'+engine+'.json')],timeout=1200,
                    extra_env={'WIZARD_BENCH_INSTALLED':'1'})
                rc=run('matrix-'+engine,[python,source/'benchmarks/three_engines.py','--config',config,'--corpus',work/'corpus',
                    '--output',work/'results','--engines',engine],timeout=1800,extra_env={'WIZARD_BENCH_INSTALLED':'1'},check=False)
                receipt['engines'][engine].update(status='measured',matrix_exit_code=rc,health_after=get(f'http://127.0.0.1:{port}/health'));save()
                # Stop ONLY the owned adapter, leaving the broker to prove fail-closed behavior over real HTTP.
                proc.terminate();proc.wait(timeout=15)
                run('http-outage-'+engine,[python,source/'benchmarks/probe_broker.py','--url','http://127.0.0.1:19864',
                    '--corpus',work/'corpus','--output',work/('outage-'+engine+'.json'),'--expect-unavailable'],timeout=60,
                    extra_env={'WIZARD_BENCH_INSTALLED':'1'})
            except Exception as exc:
                receipt['engines'].setdefault(engine,{}).update(status='failed',error_type=type(exc).__name__,error=str(exc));save()
            finally:stop_all()
        receipt['status']='completed-local-lanes'
        save()
        return 0 if all(x.get('status')=='measured' for x in receipt['engines'].values()) else 1
    except Exception as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc));save()
        raise
    finally:stop_all()


if __name__=='__main__':raise SystemExit(main())
