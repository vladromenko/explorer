"""Small automatic-learning UI service; old physical collection stays untouched."""
from contextlib import asynccontextmanager
from collections import Counter
import argparse
import json
import os
import subprocess
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

from fastapi import FastAPI,HTTPException,Request,Query
from fastapi.responses import FileResponse,HTMLResponse,JSONResponse,Response
from pydantic import BaseModel,Field
import uvicorn

from .automatic_learning import AutomaticLearning,legacy_post,legacy_get,PAUSED
from .skill_training import identity,list_trainings
from .learning import sha256
from .resource_budget import memory_budget

STATIC=Path(__file__).resolve().parent/'automatic_static'
DEFAULT_ROOT=Path('/Volumes/ExplorerSim/ExplorerSim')


class Learn(BaseModel):
    text:str=Field(min_length=1,max_length=4000)
    parent_id:str|None=Field(default=None,pattern='^[0-9a-f]{12}$')
    attempts:int=Field(default=100,ge=8,le=1000)
    hours:float=Field(default=24,gt=0,le=24)


def ensure_legacy_service():
    try:legacy_get('/api/session');return
    except OSError:pass
    support=Path.home()/'Library/Application Support/ExplorerSimulator'
    source=support/'source';script=source/'bin/start-simulator.command'
    if not script.is_file():raise ValueError('Исходная служба не найдена; сохранённые данные не изменены')
    with (support/'service.log').open('ab') as log,(support/'service-error.log').open('ab') as errors:
        process=subprocess.Popen(['/bin/bash',str(script)],cwd=source,stdout=log,stderr=errors,start_new_session=True)
    for _ in range(90):
        try:legacy_get('/api/session');return
        except OSError:
            if process.poll() is not None:break
            time.sleep(.5)
    raise ValueError('Не удалось запустить службу прежнего обучения; данные сохранены')


def summarized(row,root=DEFAULT_ROOT):
    reasons=Counter(r.get('reason') for r in row.get('episodes',[]) if r.get('status')=='complete' and r.get('success') is not True and r.get('reason'))
    workers=[]
    for worker in row.get('active_workers',[]):
        item=dict(worker)
        try:
            identity(worker['episode_id']);path=Path(root)/'skill-training'/identity(row['id'])/'episodes'/worker['episode_id']/'progress.json'
            progress=json.loads(path.read_text());item.update(phase=progress.get('phase','initializing'),frame_age_s=max(0.,time.time()-path.stat().st_mtime))
        except (OSError,ValueError,KeyError):item.update(phase='initializing',frame_age_s=None)
        workers.append(item)
    return dict(id=row['id'],state=row['state'],title='Носок → корзина' if row['task_id']=='sock_basket' else row['task_id'],
        completed=row['completed_attempts'],total=max(8,row['options']['attempts']//8*8),hours=row['options']['hours'],
        phase=row.get('phase_group'),reason=row.get('reason'),failure_causes=[dict(reason=text,count=count) for text,count in reasons.most_common(2)],active_workers=workers,
        validation_completed=sum(r.get('status')=='complete' and r.get('split') in ('validation','user_validation') for r in row.get('episodes',[])),
        validation_total=10*len(row.get('validation_seeds',[[],[]])),simulation_ready=row.get('simulation_ready',False),physical_ready=False)


def create_app(root=DEFAULT_ROOT,engine_factory=AutomaticLearning,start=True):
    @asynccontextmanager
    async def lifespan(app):
        engine=engine_factory(Path(root));app.state.engine=engine;app.state.token=secrets.token_urlsafe(32)
        if start:engine.start()
        try:yield
        finally:engine.close()
    app=FastAPI(lifespan=lifespan)
    @app.middleware('http')
    async def local(request,call_next):
        host=request.headers.get('host','');hostname=urlsplit('http://'+host).hostname
        if hostname not in ('127.0.0.1','localhost','::1'):return JSONResponse({'detail':'Только на этом Mac'},403)
        origin=request.headers.get('origin')
        if origin and origin!=str(request.base_url).rstrip('/'):return JSONResponse({'detail':'Запрос с другого сайта запрещён'},403)
        if request.headers.get('sec-fetch-site')=='cross-site':return JSONResponse({'detail':'Запрос с другого сайта запрещён'},403)
        if request.method not in ('GET','HEAD','OPTIONS'):
            if not secrets.compare_digest(request.headers.get('x-explorer-sim-token',''),request.app.state.token):
                return JSONResponse({'detail':'Обновите страницу'},403)
        try:response=await call_next(request)
        except (ValueError,KeyError,FileNotFoundError) as error:return JSONResponse({'detail':str(error)},400)
        response.headers['Cache-Control']='no-store';response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' blob:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        return response

    @app.get('/api/session')
    def session(request:Request):return dict(token=request.app.state.token,simulation_only=True)

    @app.get('/api/state')
    def state(request:Request):
        engine=request.app.state.engine;trainings=list_trainings(engine.root)
        legacy=next((row for row in trainings if row.get('scene_recipe') is None),None)
        previous=summarized(legacy,engine.root) if legacy else None
        if previous and previous['state'] in ('running','waiting_for_memory','validating','packaging'):
            try:legacy_get('/api/session')
            except OSError:previous.update(state='interrupted',active_workers=[],reason='Служба прежней очереди остановлена; продолжите сохранённое обучение')
        return dict(queue_pause=engine.pause_info(),requests=engine.list(),trainings=[summarized(row,engine.root) for row in trainings],
            legacy=previous,models=engine.model_status(),
            memory=memory_budget(),disk_connected=engine.root.parent.is_mount(),
            simulation_only=True,physical_ready=False)

    @app.post('/api/learn',status_code=202)
    def learn(body:Learn,request:Request):
        return request.app.state.engine.create(body.text,attempts=body.attempts,hours=body.hours,parent_id=body.parent_id)

    @app.post('/api/requests/{identifier}/{action}')
    def control(identifier:str,action:str,request:Request):
        engine=request.app.state.engine
        if action=='pause':return engine.pause(identifier)
        if action=='resume':return engine.resume(identifier)
        raise HTTPException(404)

    @app.post('/api/queue/resume')
    def queue_resume(request:Request):
        request.app.state.engine.resume_queue();return dict(ok=True)

    @app.post('/api/legacy/{identifier}/{action}')
    def legacy_control(identifier:str,action:str,request:Request):
        identity(identifier);engine=request.app.state.engine
        row=next((r for r in list_trainings(engine.root) if r['id']==identifier and r.get('scene_recipe') is None),None)
        if row is None:raise HTTPException(404)
        if action=='pause':
            snapshot=legacy_get('/api/state?compact=true')
            owners=[j for j in snapshot.get('jobs',[]) if j.get('status')=='running']
            if not owners or any((j.get('details') or {}).get('training_id')!=identifier for j in owners):
                raise HTTPException(409,'Другая очередь владеет вычислениями')
            engine.pause_queue(identifier)
            return legacy_post('/api/cancel',{})
        if action=='resume':
            if row['state'] not in PAUSED:
                try:legacy_get('/api/session');raise HTTPException(409,'Сохранённая очередь не на паузе')
                except OSError:pass
            if engine.simulator_busy():raise HTTPException(409,'Другая очередь уже работает')
            ensure_legacy_service()
            result=legacy_post('/api/jobs/skill_train',dict(skill_training_id=identifier,task_id=row['task_id'],
                task_ids=[row['task_id']],count=row['options']['attempts'],hours=row['options']['hours'],
                workers=row['options']['workers'],learning_mode='skill_search'))
            engine.resume_queue();return result
        raise HTTPException(404)

    @app.post('/api/disconnect')
    def disconnect(request:Request):
        import psutil
        engine=request.app.state.engine
        if engine.active_id:engine.pause(engine.active_id)
        else:engine.pause_queue()
        engine.stop.set()
        try:
            snapshot=legacy_get('/api/state?compact=true')
            owners=[j for j in snapshot.get('jobs',[]) if j.get('status')=='running']
            if owners:
                identifier=(owners[-1].get('details') or {}).get('training_id')
                engine.pause_queue(identifier);legacy_post('/api/cancel',{})
        except OSError:pass
        pids=[os.getpid()]
        for process in psutil.process_iter(['pid','cmdline']):
            args=process.info['cmdline'] or []
            if len(args)>=5 and args[-4:]==['-m','simulator.server','--port','8766']:pids.append(process.info['pid'])
        script=Path(__file__).resolve().parents[1]/'bin/shutdown-automatic-learning.py'
        support=Path.home()/'Library/Application Support/ExplorerSimulator'
        with (support/'disconnect.log').open('ab') as log:
            subprocess.Popen([os.sys.executable,str(script),*[str(pid) for pid in pids]],cwd=support,
                stdout=log,stderr=log,start_new_session=True)
        return dict(stopping=True,message='Сохраняю вычисления и отключаю диск. Дождитесь уведомления macOS перед отсоединением.')

    @app.get('/api/frames/{identifier}/{camera}')
    def frame(identifier:str,camera:str,request:Request,slot:int=Query(default=0,ge=0,le=7)):
        identity(identifier)
        if camera not in ('overview','wrist'):raise HTTPException(404)
        root=request.app.state.engine.root
        row=next((r for r in list_trainings(root) if r['id']==identifier),None)
        if row is None:raise HTTPException(404)
        workers=row.get('active_workers',[])
        worker=next((w for w in workers if w['slot']==slot),None)
        episode=worker['episode_id'] if worker else (next((r['episode_id'] for r in reversed(row.get('episodes',[]))),None) if not workers and slot==0 else None)
        if episode is None:raise HTTPException(404,'Кадр ещё готовится')
        identity(episode);folder=root/'skill-training'/identifier/'episodes'/episode
        if camera=='overview':path=folder/'overview-preview.jpg'
        else:
            frames=sorted((folder/'frames').glob('*.jpg'));path=frames[-1] if frames else None
        if path is None or not path.is_file():raise HTTPException(404,'Кадр ещё готовится')
        # Read once, avoiding stat/open races while a live file is replaced.
        return Response(path.read_bytes(),media_type='image/jpeg')

    @app.get('/api/scenes/{identifier}/{task_id}')
    def scene_preview(identifier:str,task_id:str,request:Request):
        identity(identifier);engine=request.app.state.engine;value=engine.get(identifier)
        if not any(t['task_id']==task_id for t in value['tasks']):raise HTTPException(404)
        path=engine.root/'automatic-scenes'/identifier/task_id/'preview.jpg'
        if not path.is_file():raise HTTPException(404)
        return Response(path.read_bytes(),media_type='image/jpeg')

    @app.get('/api/requests/{identifier}/package')
    def package(identifier:str,request:Request):
        engine=request.app.state.engine;value=engine.get(identifier);artifact=value.get('package')
        if value['state']!='simulation_qualified' or not artifact:raise HTTPException(409,'Проверки ещё не пройдены')
        path=Path(artifact['path']).resolve()
        if not path.is_relative_to(engine.root/'exports') or sha256(path)!=artifact['sha256']:raise HTTPException(409,'Контрольная сумма не совпала')
        return FileResponse(path,filename=path.name,media_type='application/gzip')

    @app.get('/app.js')
    def js():return FileResponse(STATIC/'app.js',media_type='text/javascript')
    @app.get('/style.css')
    def css():return FileResponse(STATIC/'style.css',media_type='text/css')
    @app.get('/')
    def index():return FileResponse(STATIC/'index.html')
    return app


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8767);parser.add_argument('--data',type=Path,default=DEFAULT_ROOT)
    options=parser.parse_args();uvicorn.run(create_app(options.data),host='127.0.0.1',port=options.port,access_log=False)

if __name__=='__main__':main()
