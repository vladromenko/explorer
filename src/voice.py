"""On-demand local microphone and speaker. Recognition never executes a command."""
import json
import math
import os
import select
from pathlib import Path
import subprocess
import threading
import time
import urllib.request
import uuid
import wave
import numpy as np

ROOT=Path('/home/vlad/Explorer')

def audio_quality(path):
    with wave.open(str(path),'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getframerate()!=16000:
            raise ValueError('Неподдерживаемый формат записи')
        values=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)/32768
    if not len(values):raise ValueError('Пустая запись микрофона')
    rms=float(np.sqrt(np.mean(values**2)))
    return dict(duration_s=len(values)/16000,rms_dbfs=20*math.log10(max(rms,1e-9)),
                peak=float(np.max(np.abs(values))),clipped_fraction=float(np.mean(np.abs(values)>.999)))

class Voice:
    def __init__(self,root=ROOT):
        self.root=Path(root);self.config=json.loads((self.root/'config/voice.json').read_text())
        self.folder=self.root/'data/voice';self.folder.mkdir(exist_ok=True)
        self.lock=threading.Lock();self.cancel=threading.Event();self.process=None
        self.state=dict(phase='idle',transcript='',error=None,recording=False,automatic_motion=False)

    def status(self):
        return dict(self.state,busy=self.lock.locked(),continuous_listening=False,
                    speaker_model=(self.root/self.config['voice_model']).exists(),
                    microphone_name='C-Media USB',local_only=True)

    def power_check(self):
        power=json.loads((self.root/'data/power.json').read_text())
        if not 0<=time.time()-power['at']<4 or power['state'] in ('LOW_POWER','CRITICAL','UNKNOWN'):
            raise ValueError('Голос отключён до восстановления питания или телеметрии')

    def start(self,operation,text=''):
        if operation not in ('listen','speak','test'):raise ValueError('Неизвестная голосовая операция')
        if operation=='speak' and (not text.strip() or len(text)>1200):raise ValueError('Введите текст до 1200 символов')
        self.power_check()
        if not self.lock.acquire(blocking=False):raise ValueError('Завершите текущую запись или воспроизведение')
        self.cancel.clear();self.state.update(phase='starting',error=None,recording=False,transcript='')
        threading.Thread(target=self.run,args=(operation,text),daemon=True).start()
        return self.status()

    def command(self,args,input_text=None,timeout=30):
        if self.cancel.is_set():raise InterruptedError('Операция отменена')
        self.process=subprocess.Popen(args,stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                                      stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            out,err=self.process.communicate(input=input_text,timeout=timeout)
            if self.cancel.is_set():raise InterruptedError('Операция отменена')
            if self.process.returncode:raise ValueError(err[-700:] or 'Аудиоустройство недоступно')
            return out
        except subprocess.TimeoutExpired:
            self.process.kill();self.process.communicate();raise ValueError('Аудиооперация превысила время ожидания')
        finally:self.process=None

    def ensure_recognizer(self):
        self.power_check()
        subprocess.run(['systemctl','--user','start','explorer-speech.service'],check=True,timeout=5)
        deadline=time.monotonic()+25
        while time.monotonic()<deadline:
            if self.cancel.is_set():raise InterruptedError('Операция отменена')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8082/',timeout=1):return
            except OSError:time.sleep(.25)
        raise ValueError('Распознавание речи не запустилось')

    def transcribe(self,path):
        quality=audio_quality(path);self.state['microphone']=quality
        if quality['rms_dbfs']<-55:raise ValueError('Слишком тихо: проверьте микрофон и говорите ближе')
        if quality['clipped_fraction']>.15:raise ValueError('Микрофон перегружен: уменьшите усиление')
        self.ensure_recognizer();self.state.update(phase='recognizing',recording=False)
        boundary='Explorer'+uuid.uuid4().hex
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()+path.read_bytes()+b'\r\n'
        for key,value in {'response_format':'json','language':'ru','temperature':'0.0','temperature_inc':'0.0'}.items():
            body+=f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        body+=f'--{boundary}--\r\n'.encode()
        req=urllib.request.Request('http://127.0.0.1:8082/inference',data=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary})
        with urllib.request.urlopen(req,timeout=65) as r:result=json.load(r)
        if self.cancel.is_set():raise InterruptedError('Операция отменена')
        return result.get('text','').strip()

    def record(self,path):
        self.state.update(phase='recording',recording=True)
        self.process=subprocess.Popen(['parec','--raw','--device='+self.config['capture_source'],
                 '--format=s16le','--rate=16000','--channels=1','--latency-msec=50'],
                 stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        samples=bytearray();limit=self.config['record_seconds']*32000;end=time.monotonic()+self.config['record_seconds']+2
        try:
            while len(samples)<limit and time.monotonic()<end:
                if self.cancel.is_set():raise InterruptedError('Запись отменена')
                if self.process.poll() is not None:raise ValueError('Микрофон недоступен: проверьте подключение USB')
                ready,_,_=select.select([self.process.stdout],[],[],.1)
                if ready:
                    chunk=os.read(self.process.stdout.fileno(),min(4096,limit-len(samples)))
                    if not chunk:raise ValueError('Нет данных от микрофона')
                    samples.extend(chunk)
            if len(samples)<16000:raise ValueError('Микрофон не передал достаточно данных')
            with wave.open(str(path),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(samples)
        finally:
            self.process.terminate()
            try:self.process.communicate(timeout=2)
            except subprocess.TimeoutExpired:self.process.kill();self.process.communicate()
            self.process=None;self.state['recording']=False

    def speak(self,text,path):
        self.state['phase']='synthesizing'
        self.command([str(self.root/'.venv-voice/bin/python'),str(self.root/'bin/speak.py'),str(path)],input_text=text,timeout=40)
        self.power_check();self.state['phase']='speaking'
        self.command(['paplay','--device='+self.config['playback_sink'],str(path)],timeout=90)

    def run(self,operation,text):
        path=self.folder/('temporary-'+uuid.uuid4().hex+'.wav')
        try:
            if operation=='listen':
                self.record(path);self.state['transcript']=self.transcribe(path)
                if not self.state['transcript']:raise ValueError('Речь не распознана; попробуйте ещё раз')
            else:self.speak('Проверка звука. Я робот Эксплорер. Микрофон включается только по вашей команде.' if operation=='test' else text,path)
            self.state['phase']='ready' if operation=='listen' else 'idle'
        except (OSError,ValueError,InterruptedError,wave.Error,subprocess.SubprocessError) as exc:
            self.state.update(phase='error',error=str(exc))
        finally:
            path.unlink(missing_ok=True);self.state['recording']=False
            # Recognition is demand-loaded, then unloaded: no idle GPU occupancy.
            try:subprocess.run(['systemctl','--user','stop','explorer-speech.service'],timeout=6,check=False)
            finally:self.lock.release()

    def stop(self):
        self.cancel.set()
        process=self.process
        if process and process.poll() is None:process.terminate()
        subprocess.run(['systemctl','--user','--no-block','stop','explorer-speech.service'],timeout=3,check=False)
        return {'cancel_requested':True}
