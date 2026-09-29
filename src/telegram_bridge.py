"""Allowlisted Telegram adapter to Explorer's existing deterministic API.

No shell, raw actuator API, commissioning changes, automatic stop release or
replay of queued movement after reconnect. Secrets never enter status/errors.
"""
import json
import fcntl
import queue
import secrets
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from lerobot_bridge import write_json
from telegram_pairing import Pairing

HELP='''Explorer: /status — питание, нагрузка, режим и датчики; /stop — остановить;
/objects — что видно; /where предмет — последние наблюдения;
/find sock — поиск на текущем кадре; /places — сохранённые места;
/go имя — поездка; /survey имя1,имя2 — осмотр маршрута;
/explore — исследовать доступные границы карты; /mobile — телефонная панель;
/lights режим — подсветка: auto, off, headlights, work, search, success, error, gradient;
/skills — функции и обучение;
/ask вопрос — локальный помощник.
Поездки требуют готового шасси и выбранного автономного режима в панели.'''

def format_status(value,graduation=None):
    power=value.get('power_telemetry',{});resources=power.get('resources',{})
    voltage=power.get('battery_voltage_v',value.get('battery'));gauge=value.get('battery_gauge',{})
    temperatures=resources.get('temperatures_c',{});temperature=max(temperatures.values()) if temperatures else None
    sensor=value.get('sensor_age',{});fresh=sum(type(v) in (int,float) and 0<=v<1 for v in sensor.values())
    lines=['Explorer: '+('STOP' if value.get('stop_latched') else 'движение разрешено')+' · '+str(value.get('mode','—'))]
    approximate=gauge.get('percent')
    lines.append('Питание: '+(('%.2f В'%voltage) if type(voltage) in (int,float) else 'нет данных')+
                 (' · V≈% '+str(round(approximate)) if type(approximate) in (int,float) else '')+' · '+str(power.get('state','UNKNOWN')))
    lines.append('Jetson: CPU '+(('%.0f%%'%resources['cpu_percent']) if type(resources.get('cpu_percent')) in (int,float) else '—')+
                 ' · GPU '+(('%.0f%%'%resources['gpu_percent']) if type(resources.get('gpu_percent')) in (int,float) else '—')+
                 ' · RAM свободно '+(('%.0f МБ'%resources['ram_available_mb']) if type(resources.get('ram_available_mb')) in (int,float) else '—')+
                 ' · t° '+(('%.1f°C'%temperature) if temperature is not None else '—'))
    lines.append('Датчики свежее 1 с: '+str(fresh)+'/'+str(len(sensor))+' · камера '+('свежая' if not value.get('perception',{}).get('stale',True) else 'нет свежего кадра'))
    controller=value.get('controller',{});lines.append('Контроллер: '+('на связи' if not controller.get('stale',True) else 'нет свежей связи')+' · причина: '+str(value.get('reason','—')))
    if graduation:
        accepted=set(graduation.get('accepted',[]));lines.append('Автономность: '+str(len(accepted))+'/'+str(len(graduation.get('items',[])))+' допусков · '+(', '.join(sorted(accepted)) or 'пока нет'))
        pending=next((x for x in graduation.get('items',[]) if x.get('state')!='accepted'),None)
        if pending:lines.append('Следующее: '+pending['name']+' — '+pending['next_action'])
    return '\n'.join(lines)

def format_experiment_catalog(experiments):
    lines=['Эксперименты выполняют анализ без движения. Нажатие запускает один observe-тест:']
    for item in experiments:
        lines.append(item['id']+' · '+item['name']+' — '+item['implementation'])
    return '\n'.join(lines)


def authorize(update,config,now):
    message=update.get('message',{})
    sender=message.get('from',{});chat=message.get('chat',{})
    if chat.get('type')!='private' or sender.get('is_bot'):return None
    if chat.get('id') not in config['allowed_chat_ids'] or sender.get('id') not in config['allowed_user_ids']:return None
    text=message.get('text','').strip()
    if not text or len(text)>1500:return None
    age=now-message.get('date',0)
    if not 0<=age<=30:return None
    return dict(chat_id=chat['id'],text=text,at=message['date'])


def command(text):
    name,_,argument=text.partition(' ');name=name.lower();argument=argument.strip()
    if name in ('/stop','стоп'):return 'control',dict(op='stop')
    if name=='/status':return 'status',None
    if name=='/objects':return 'objects',None
    if name=='/places':return 'places',None
    if name=='/experiments':return 'experiments',None
    if name=='/memory':return 'experiments/memory',None
    if name=='/learn':return 'teaching',None
    if name=='/results':return 'experiments/results',None
    if name=='/skills':return 'skills',None
    if name=='/explore':return 'missions',dict(kind='explore',x=None,y=None,yaw=0.)
    if name=='/camera':return 'camera',None
    if name=='/mobile':return 'mobile',None
    if name=='/lights':
        allowed=('auto','off','headlights','work','search','success','error','water','marquee','breathe','gradient','sparkle','battery')
        if not argument:return 'appearance',None
        if argument not in allowed:raise ValueError('Режимы: '+', '.join(allowed))
        return 'appearance',dict(mode=argument)
    if name=='/where' and argument:
        from urllib.parse import urlencode
        return 'memory?'+urlencode({'label':argument}),None
    if name=='/find' and argument:return 'objects/find',dict(label=argument)
    if name=='/go' and argument:return 'places/go',dict(name=argument)
    if name=='/survey' and argument:
        places=[p.strip() for p in argument.split(',') if p.strip()]
        if not 1<=len(places)<=12:raise ValueError('Нужно от 1 до 12 имён мест')
        return 'agents/survey',dict(places=places,narrate=False)
    if name=='/ask' and argument:return 'agent',dict(text=argument)
    if name in ('/start','/help'):return None,None
    if not name.startswith('/'):return 'agent',dict(text=text)
    raise ValueError('Неизвестная команда. /help — список')


class TelegramBridge:
    def __init__(self,root):
        self.root=Path(root);self.config=json.loads((self.root/'config/telegram.json').read_text())
        if not self.config.get('enabled'):raise ValueError('Telegram не включён')
        for field in ('allowed_chat_ids','allowed_user_ids'):
            if any(type(v) is not int or v<=0 for v in self.config.get(field,[])):
                raise ValueError('Нужны явные числовые разрешения Telegram')
        token_path=Path(self.config.get('token_file',self.root/'config/telegram-token'))
        if token_path.stat().st_mode&0o077:raise ValueError('Файл токена должен быть доступен только владельцу')
        self.token=token_path.read_text().strip()
        if ':' not in self.token:raise ValueError('Некорректный токен Telegram')
        self.api_token=(self.root/'config/access_token').read_text().strip()
        self.cursor_file=self.root/'data/telegram-cursor.json'
        self.offset=json.loads(self.cursor_file.read_text())['offset'] if self.cursor_file.exists() else 0
        self.started=time.time();self.tasks=queue.Queue(maxsize=8)
        self.pairing=Pairing(self.root);self.generation=0;self.callbacks={}
        self.poller_lock=(self.root/'data/telegram-poller.lock').open('a')
        fcntl.flock(self.poller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Only Telegram uses the explicit VPN proxy; robot API remains local.
        proxy=self.config.get('https_proxy')
        self.network=urllib.request.build_opener(urllib.request.ProxyHandler({'https':proxy} if proxy else {}))
        self.local=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def telegram(self,method,payload,timeout=30):
        request=urllib.request.Request('https://api.telegram.org/bot'+self.token+'/'+method,
            data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with self.network.open(request,timeout=timeout) as response:result=json.load(response)
        if not result.get('ok'):raise ValueError('Telegram API rejected request')
        return result['result']

    def api(self,path,payload):
        request=urllib.request.Request('http://127.0.0.1:8080/api/'+path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Authorization':'Bearer '+self.api_token,'Content-Type':'application/json'})
        try:
            with self.local.open(request,timeout=90 if path=='agent' else 5) as response:return json.load(response)
        except urllib.error.HTTPError as exc:
            try:detail=json.load(exc).get('detail','Команда отклонена')
            except ValueError:detail='Команда отклонена'
            raise ValueError(str(detail)) from None

    def reply(self,chat,text,markup=None):
        keyboard=markup or {'keyboard':[[{'text':x} for x in row] for row in
            [('Состояние','Камера'),('Телефон','Эксперименты'),('Память','Обучение'),('Результаты','Остановить')]],
            'resize_keyboard':True}
        self.telegram('sendMessage',dict(chat_id=chat,text=text[:3900],reply_markup=keyboard,
            link_preview_options={'is_disabled':True}),timeout=8)

    def callback_button(self,chat,experiment):
        key=secrets.token_urlsafe(12)
        self.callbacks={k:v for k,v in self.callbacks.items() if time.time()<v['expires']}
        if len(self.callbacks)>=100:self.callbacks.clear()
        self.callbacks[key]=dict(chat=chat,experiment=experiment['id'],expires=time.time()+120,generation=self.generation)
        return {'text':experiment['id']+' · '+experiment['name'][:42],'callback_data':key}

    def execute(self,item):
        if item.get('generation',self.generation)!=self.generation:return
        if time.time()-item['at']>30:
            self.reply(item['chat_id'],'Команда устарела в очереди и не выполнена.');return
        if item.get('experiment'):
            result=self.api('experiments/run',dict(experiment=item['experiment'],mode='observe',params={},
                request_id=item['request_id'],issued_at=item['at']))
            if item['generation']==self.generation:
                self.reply(item['chat_id'],result.get('result',{}).get('summary',result.get('state','Нет результата')))
            return
        path,payload=command(item['text'])
        if path is None:self.reply(item['chat_id'],HELP);return
        if path=='camera':
            from remote_access import status
            remote=status(self.root);token=self.api_token
            urls=['Домашняя сеть: http://explorer.local:8080/#'+token]
            if remote.get('local_url'):urls.append('По IP: '+remote['local_url'].replace('/mobile','/')+'#'+token)
            if remote.get('tailscale_url'):urls.append('Через Tailscale: '+remote['tailscale_url'].replace('/mobile','/')+'#'+token)
            self.reply(item['chat_id'],'Камера и полная панель:\n'+'\n'.join(urls));return
        if path=='mobile':
            from remote_access import status
            remote=status(self.root)
            urls=['Дома: http://explorer.local:8080/mobile#'+self.api_token]
            if remote.get('local_url'):urls.append('По локальному IP: '+remote['local_url']+'#'+self.api_token)
            if remote.get('tailscale_url'):urls.append('Вне дома через Tailscale: '+remote['tailscale_url']+'#'+self.api_token)
            text='Телефонное управление и запись обучения:\n'+'\n'.join(urls)+'\nПанель содержит шасси, руку, захват, камеру и запись полного показа.'
            self.reply(item['chat_id'],text);return
        if path=='experiments':
            result=self.api(path,None)
            self.reply(item['chat_id'],format_experiment_catalog(result['experiments']),
                {'inline_keyboard':[[self.callback_button(item['chat_id'],e)] for e in result['experiments']]})
            return
        if path=='appearance':
            result=self.api(path,payload)
            self.reply(item['chat_id'],'Передняя подсветка: '+result['rgb_mode']+'\nКоманда: /lights auto|off|headlights|work|search|success|error|gradient');return
        if path=='skills':
            self.reply(item['chat_id'],'Функции: карта и frontier exploration; поездки к сохранённым местам; поиск предметов GroundingDINO/YOLO; память «где видел»; ручное управление шасси и 6 суставами; запись полного показа подъехать→взять→перевезти→положить; офлайн LeRobot ACT после 10+ успешных показов; эксперименты /experiments. Автономный навык включается только после проверки модели.')
            return
        result=self.api('status' if path=='objects' else path,payload)
        if path=='status':
            try:graduation=self.api('autonomy/graduation',None)
            except (OSError,ValueError):graduation=None
            text=format_status(result,graduation)
        elif path=='objects':
            perception=result.get('perception',{})
            if perception.get('stale',True):text='Нет свежего изображения камеры.'
            else:
                text='Предположения детектора: '+('; '.join(str(o['label'])+' '+str(round(o['confidence']*100))+'%' for o in perception.get('objects',[])) or 'ничего уверенно не распознано')
        elif path=='places':text='Сохранённые места: '+(', '.join(p['name'] for p in result) or 'пока нет')
        elif path=='experiments/results':text='Последние результаты:\n'+'\n'.join(r['experiment']+' · '+r['state']+' · '+str(r.get('summary','')) for r in result[:6])
        elif path=='experiments/memory':text='Память предметов:\n'+'\n'.join(o['label']+' · '+o['id'][:8] for o in result['objects'][:15])
        elif path=='teaching':
            graduation=self.api('autonomy/graduation',None)
            text='Обучение доступно в телефонной панели (/mobile). Успешных показов руки: '+str(result['teaching'].get('successful',0))+'; полных мобильных: '+str(result['mobile'].get('successful',0))+'\n'+'\n'.join(x['name']+': '+('открыто' if x['state']=='accepted' else x['next_action']) for x in graduation['items'])
        elif path=='agent':text=result.get('answer','Нет ответа')
        elif path=='control':text='Команда STOP передана. При физической неисправности связи нужна кнопка питания.'
        elif path in ('places/go','agents/survey','missions'):
            text='Задание принято: '+str(result.get('id'))+'. Завершение ещё не подтверждено.'
        else:text=json.dumps(result,ensure_ascii=False)[:3800]
        self.reply(item['chat_id'],text)

    def worker(self):
        while True:
            item=self.tasks.get()
            try:self.execute(item)
            except ValueError as exc:
                try:self.reply(item['chat_id'],str(exc)[:1000])
                except (OSError,ValueError):pass
            except OSError:
                try:self.reply(item['chat_id'],'Связь недоступна. Команда автоматически не повторяется.')
                except (OSError,ValueError):pass
            except (KeyError,TypeError):
                try:self.reply(item['chat_id'],'Ответ сервиса не распознан. Повторного физического запуска не будет.')
                except (OSError,ValueError):pass
            finally:self.tasks.task_done()

    def run(self):
        me=self.telegram('getMe',{})
        if me['id']!=self.config.get('bot_id',8850343219):raise ValueError('Unexpected bot ID')
        if self.telegram('getWebhookInfo',{}).get('url'):raise ValueError('Existing webhook; no changes made')
        names={'/status':'Состояние','/camera':'Камера','/experiments':'Эксперименты','/memory':'Память',
               '/skills':'Функции и обучение','/explore':'Исследовать комнату','/mobile':'Телефонная панель',
               '/learn':'Обучение','/lights':'Передняя подсветка','/results':'Результаты','/stop':'Остановить'}
        self.telegram('setMyCommands',{'commands':[{'command':k[1:],'description':v} for k,v in names.items()]})
        labels={v:k for k,v in names.items()}
        labels['Телефон']='/mobile'
        threading.Thread(target=self.worker,daemon=True).start()
        while True:
            try:
                self.config=json.loads((self.root/'config/telegram.json').read_text())
                updates=self.telegram('getUpdates',dict(offset=self.offset,timeout=20,allowed_updates=['message','callback_query']))
                self.config=json.loads((self.root/'config/telegram.json').read_text())
                for update in updates:
                    if update['update_id']>=self.offset:
                        # Persist before dispatch: restart cannot repeat an action.
                        self.offset=update['update_id']+1
                        write_json(self.cursor_file,dict(offset=self.offset))
                        message=update.get('message',{})
                        sender=message.get('from',{});chat=message.get('chat',{})
                        content=message.get('text','')
                        if content.startswith('/pair ') and chat.get('type')=='private' and not sender.get('is_bot') and 0<=time.time()-message.get('date',0)<30:
                            accepted=self.pairing.claim(content[6:].strip(),sender.get('id'),chat.get('id'),sender.get('username',''))
                            if accepted:self.reply(chat['id'],'Заявка получена. Подтвердите свой аккаунт в локальной веб-панели Explorer.')
                        callback=update.get('callback_query')
                        if callback:
                            owner=callback.get('from',{}).get('id');chat_id=callback.get('message',{}).get('chat',{}).get('id')
                            if owner in self.config['allowed_user_ids'] and chat_id in self.config['allowed_chat_ids']:
                                action=self.callbacks.pop(callback.get('data'),None)
                                self.telegram('answerCallbackQuery',{'callback_query_id':callback['id']})
                                if action and action['chat']==chat_id and time.time()<action['expires'] and action['generation']==self.generation:
                                    item=dict(chat_id=chat_id,experiment=action['experiment'],at=time.time(),
                                        request_id='tg-'+str(update['update_id'])+'-'+action['experiment'],generation=self.generation)
                                    try:self.tasks.put_nowait(item)
                                    except queue.Full:self.reply(chat_id,'Очередь занята. Откройте эксперименты заново.')
                                else:self.reply(chat_id,'Кнопка истекла или уже использована. Откройте «Эксперименты» заново.')
                        if content in labels:message['text']=labels[content]
                        item=authorize(update,self.config,time.time())
                        if item and item['at']>=self.started:
                            if item['text'].lower() in ('/stop','стоп'):
                                self.generation+=1;self.callbacks.clear();self.execute(item)
                            else:
                                item['generation']=self.generation
                                try:self.tasks.put_nowait(item)
                                except queue.Full:self.reply(item['chat_id'],'Очередь занята. Повторите позже.')
                write_json(self.root/'data/telegram-status.json',dict(at=time.time(),connected=True,queue=self.tasks.qsize(),
                    username=me['username'],bot_id=me['id'],paired=bool(self.config['allowed_user_ids'])))
            except (OSError,ValueError,KeyError) as exc:
                write_json(self.root/'data/telegram-status.json',dict(at=time.time(),connected=False,error=type(exc).__name__))
                time.sleep(3)


if __name__=='__main__':
    try:TelegramBridge('/home/vlad/Explorer').run()
    except Exception as exc:raise SystemExit('Telegram service stopped: '+type(exc).__name__) from None
