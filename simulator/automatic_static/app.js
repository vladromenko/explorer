'use strict';
const $=id=>document.getElementById(id);
const labels={queued:'В очереди',planning:'Разбор задачи',waiting_for_current:'Ожидает текущую очередь',building_scene:'Создание сцены',training:'Обучается',running:'Обучается',validating:'Независимая проверка',packaging:'Проверка пакета',waiting_for_memory:'Ожидает память',pausing:'Останавливается',operator_paused:'Пауза',maintenance_paused:'Сохранено для обновления программы',memory_paused:'Пауза: память',budget_paused:'Бюджет завершён',interrupted:'Остановлено при перезапуске',needs_clarification:'Нужно уточнение',needs_diagnosis:'Остановлено без улучшения',validation_failed:'Проверка не пройдена',simulation_qualified:'Проверено в симуляции',failed:'Ошибка'};
const paused=new Set(['operator_paused','memory_paused','budget_paused','interrupted','maintenance_paused']);
const active=new Set(['pausing','queued','planning','waiting_for_current','building_scene','training','running','validating','packaging','waiting_for_memory']);
const phases={pick:'Захват и подъём',view:'Подъезд и обзор',place:'Размещение'};
const searchModes={geometry:'точка и сторона подхода',contact:'смыкание и контакт',hold_view:'подъём и проверка захвата',inspection:'обзор корзины',release:'отпускание',withdraw:'отвод и проверка размещения',view:'ракурс камеры',causal_repair:'сравнение исправления причины',repair_confirmation:'перепроверка исправления на новых сценах'};
const workerPhases={initializing:'Подготовка сцены',skill_find:'Поиск предмета',skill_find_destination:'Осмотр корзины или подноса',skill_approach:'Подъезд',skill_reobserve:'Повторное измерение',skill_pregrasp:'Подход захвата',skill_grasp:'Захват',skill_lift:'Подъём',skill_verify_hold:'Проверка удержания',skill_carry:'Перенос',skill_reorient_destination:'Проверка обзора над опорой',skill_support:'Подход к месту размещения',skill_release:'Отпускание',skill_withdraw:'Отвод руки',skill_verify_place:'Проверка размещения',skill_view_retreat:'Выбор другого ракурса'};
let token='',snapshot=null,target=null,parent=null,busy=false,signature='',disconnecting=false,historyOpen=false,frameTargets=[];
async function api(path,body,retry=true){const response=await fetch(path,{signal:AbortSignal.timeout(12000),...(body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Explorer-Sim-Token':token},body:JSON.stringify(body)})});if(response.status===403&&retry){token=(await api('/api/session',undefined,false)).token;return api(path,body,false);}if(!response.ok){const error=await response.json().catch(()=>({}));throw Error(error.detail||'Ошибка связи');}return response.json();}
function notice(message){$('notice').textContent=message;$('notice').hidden=false;setTimeout(()=>$('notice').hidden=true,7000);}
function node(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function priorAttempts(s,row){
 const seen=new Set(),trainingIds=new Set();let previous=s.requests.find(r=>r.id===row.parent_id),count=0;
 while(previous&&!seen.has(previous.id)&&previous.id!==row.id){seen.add(previous.id);for(const task of previous.tasks||[]){if(trainingIds.has(task.training_id))continue;trainingIds.add(task.training_id);count+=s.trainings.find(t=>t.id===task.training_id)?.completed||0;}previous=s.requests.find(r=>r.id===previous.parent_id);}
 return count;
}
function disposeCard(card){for(const img of card.querySelectorAll('img')){img.frameRequest?.abort();if(img.dataset.url)URL.revokeObjectURL(img.dataset.url);}card.remove();}
async function refreshFrames(){
 if(document.hidden||disconnecting)return;
 await Promise.all(frameTargets.map(async({img,training,slot,camera,episode})=>{
  if(!img.isConnected||img.frameRequest)return;
  const controller=new AbortController();img.frameRequest=controller;const timeout=setTimeout(()=>controller.abort(),12000);let url;
  try{const response=await fetch(`/api/frames/${training}/${camera}?slot=${slot}&t=${Date.now()}`,{signal:controller.signal});if(!response.ok)throw Error('Кадр готовится');url=URL.createObjectURL(await response.blob());const decoded=new Image();decoded.src=url;await decoded.decode();
   if(!img.isConnected||img.dataset.episode!==episode)return;
   const previous=img.dataset.url;img.src=url;img.dataset.url=url;url=null;img.hidden=false;img.parentElement.querySelector('.frame-wait').hidden=true;if(previous)URL.revokeObjectURL(previous);
  }catch{if(img.hidden)img.parentElement.querySelector('.frame-wait').hidden=false;}
  finally{clearTimeout(timeout);if(url)URL.revokeObjectURL(url);if(img.frameRequest===controller)delete img.frameRequest;}
 }));
}
function showCurrent(s){
 const own=s.requests.find(r=>active.has(r.state)&&r.tasks?.some(t=>t.progress?.active_workers?.length))||s.requests.find(r=>['training','validating','packaging'].includes(r.state));
 const legacy=s.legacy;
 const liveLegacy=legacy&&active.has(legacy.state);
 let row,kind,id,training;
 const pausedOwner=s.queue_pause?.owner_id,pausedRequest=s.requests.find(r=>r.id===pausedOwner);
 if(pausedRequest){row=pausedRequest;kind='requests';id=row.id;training=s.trainings.find(t=>row.tasks?.some(task=>task.training_id===t.id));}
 else if(s.queue_pause&&legacy?.id===pausedOwner){row=legacy;kind='legacy';id=row.id;training=legacy;}
 else if(liveLegacy){row=legacy;kind='legacy';id=legacy.id;training=legacy;}
 else if(own){row=own;kind='requests';id=own.id;training=s.trainings.find(t=>own.tasks.some(task=>task.training_id===t.id&&task.state!=='simulation_qualified'));}
 else if(legacy&&paused.has(legacy.state)){row=legacy;kind='legacy';id=legacy.id;training=legacy;}
 else{row=s.requests.find(r=>active.has(r.state))||s.requests.find(r=>paused.has(r.state)&&!r.superseded_by);if(row){kind='requests';id=row.id;training=s.trainings.find(t=>row.tasks?.some(task=>task.training_id===t.id));}else if(!s.requests.length&&legacy){row=legacy;training=legacy;kind='legacy';id=legacy.id;}}
 if(s.queue_pause&&(!row||row.state==='queued'||row.state==='waiting_for_current')){row={...row,state:'operator_paused',reason:'Очередь на паузе'};kind='queue';}
 target=row?{kind,id,state:row.state}:null;$('current-control').disabled=disconnecting||row?.state==='pausing';$('current-panel').hidden=!row;if(!row){frameTargets=[];for(const card of Array.from($('cameras').children))disposeCard(card);return;}
 $('current-title').textContent=row.title||row.plan?.title||'Ваша задача';$('current-status').textContent=training?.active_workers?.some(w=>w.unresponsive)?'Исполнитель не отвечает: симуляция не выдаёт кадры или прогресс.':row.reason||labels[row.state]||row.state;$('current-cause').textContent=!active.has(row.state)?(row.failure_causes||[]).map(c=>`${c.count} попыток: ${c.reason}`).join(' · '):'';
 const completed=training?.completed||0,total=training?.total||row.budget?.attempts||100;
 $('progress').max=total;$('progress').value=completed;$('attempts').textContent=training?.state==='validating'?`Проверено ${training.validation_completed} из ${training.validation_total} независимых сцен`:`В текущей версии: ${completed} из ${total} попыток`;
 if(training?.state==='validating'){$('progress').max=training.validation_total;$('progress').value=training.validation_completed;}
 $('workers').textContent=`Исполнителей: ${training?.active_workers?.length||0}`;$('phase').textContent=(phases[training?.phase]?'Подбор: '+phases[training.phase]:labels[row.state]||row.state)+(searchModes[training?.search_mode]?' · '+searchModes[training.search_mode]:'')+(training?.automatic_recoveries?' · смен стратегии: '+training.automatic_recoveries:'');
 let repair=$('repair-status');if(!repair){repair=node('p',undefined,'muted');repair.id='repair-status';$('phase').parentElement.after(repair);}
 const previousAttempts=kind==='requests'?priorAttempts(s,row):0;
 repair.textContent=training?[previousAttempts?`В предыдущих версиях сохранено ${previousAttempts} попыток; они не входят в счётчик этой версии`:'',training.warm_start_from?'Найденные ранее настройки перенесены для повторной проверки':'',(training.diagnosed_causes||[]).map(c=>c.title).join(' · '),training.repair_tests?`Проверок исправлений: ${training.repair_tests}; принятых улучшений: ${training.promoted_repairs}`:'',training.distinct_training_scenes?`Различных учебных сцен: ${training.distinct_training_scenes}; пройдено пар: ${training.training_scene_pairs}/${training.minimum_scene_pairs}`:'',training.reused_recordings?`Используется архив из ${training.reused_recordings} совместимых записей`:''].filter(Boolean).join('. '):'';

 const capacity=training?.parallelism;if(capacity){const explanation=capacity.state==='measuring'?`Измеряется параллелизм: ${capacity.workers} исполнителей; эти повторы не засчитываются в обучение и проверку`:capacity.state==='completed'?`Замер: ускорение ${capacity.speedup.toFixed(2)}×; выбранных исполнителей: ${training.configured_workers}; минимальный запас памяти ${(capacity.minimum_memory_fraction*100).toFixed(0)}%`:'Больше исполнителей будет проверено, когда хватит памяти';repair.textContent+='. '+explanation;}
 $('current-control').hidden=!(active.has(row.state)||paused.has(row.state));$('current-control').textContent=paused.has(row.state)?'Продолжить':'Пауза';
 if(training){
  const views=training.active_workers?.length?training.active_workers:[{slot:0}];
  frameTargets=[];
  for(const card of Array.from($('cameras').children))if(!views.some(w=>card.id===`worker-card-${w.slot}`))disposeCard(card);
  for(const worker of views){const key=JSON.stringify([training.id,worker.episode_id]);let card=$(`worker-card-${worker.slot}`);
   if(card&&card.dataset.episode!==key){disposeCard(card);card=null;}
   if(!card){card=node('div',undefined,'worker');card.id=`worker-card-${worker.slot}`;card.dataset.episode=key;const images=node('div',undefined,'cameras');card.append(node('h3',`Исполнитель ${worker.slot+1}`));for(const camera of ['overview','wrist']){const figure=node('figure'),img=node('img'),waiting=node('div','Подготавливаю кадр…','frame-wait');img.id=`view-${worker.slot}-${camera}`;img.dataset.episode=key;img.hidden=true;img.alt=camera==='overview'?'Общий вид попытки':'Камера на руке';figure.append(img,waiting,node('figcaption',camera==='overview'?'Физическая симуляция · доступные кадры до 2/с':'Что видит робот'));images.append(figure);}card.append(images);$('cameras').append(card);}
   card.style.order=String(worker.slot);
   for(const camera of views.length>1?['overview']:['overview','wrist'])frameTargets.push({img:$(`view-${worker.slot}-${camera}`),training:training.id,slot:worker.slot,camera,episode:key});
  }
  for(const worker of views){const heading=$(`worker-card-${worker.slot}`).querySelector('h3');heading.textContent=`Исполнитель ${worker.slot+1} · ${worker.unresponsive?'Не отвечает':worker.frame_age_s>12?'Планирование или распознавание':workerPhases[worker.phase]||'Расчёт следующего действия'}`;}
  void refreshFrames();
 }else{frameTargets=[];for(const card of Array.from($('cameras').children))disposeCard(card);}
}
function showRequests(s){const next=JSON.stringify(s.requests);if(next===signature)return;signature=next;const root=$('requests');root.replaceChildren();if(!s.requests.length){root.append(node('p','Новые просьбы появятся здесь. Текущее обучение продолжает работу.','muted'));return;}
 const archived=s.requests.filter(row=>row.superseded_by),history=node('details');history.id='request-history';history.open=historyOpen;history.addEventListener('toggle',()=>{historyOpen=history.open;});history.append(node('summary',`История версий (${archived.length})`),node('p','Прежние записи сохранены для диагностики и подсказок. Они не засчитываются как проверка текущей версии.','muted'));
 for(const row of s.requests){const card=node('article',undefined,'task'),heading=node('div',undefined,'task-title');heading.append(node('h3',row.plan?.title||row.text),node('span',row.superseded_by?'Предыдущая версия':labels[row.state]||row.state,'state'));card.append(heading);if(row.superseded_by){card.append(node('p','Прежние попытки сохранены. Задача продолжена с исправленным исполнителем.','muted'));history.append(card);continue;}card.append(node('p',row.reason||''));
  if(row.plan){for(const text of row.plan.missing||[])card.append(node('p',text,'blocked'));const steps=node('ol');for(const task of row.tasks||[]){const li=node('li',task.recipe.title+' · '+(labels[task.state]||task.state));steps.append(li);if(task.scene){const image=node('img');image.src=`/api/scenes/${row.id}/${task.task_id}`;image.alt='Созданная сцена: '+task.recipe.title;card.append(image);}}card.append(steps);for(const task of row.tasks||[])for(const cause of task.result?.failure_causes||[])card.append(node('p',`${cause.count} попыток: ${cause.reason}`,'muted'));}
  if(row.state==='needs_clarification'||row.state==='needs_diagnosis'||row.state==='validation_failed'){const button=node('button','Уточнить задачу');button.addEventListener('click',()=>{parent=row.id;$('request').value='';$('request').focus();$('form-status').textContent='Уточнение к задаче: '+(row.plan?.title||row.text);$('learn').textContent='Отправить уточнение';});card.append(button);}
  if(paused.has(row.state)||row.can_resume_after_update){const button=node('button',row.can_resume_after_update?'Продолжить с исправлением':'Продолжить');button.addEventListener('click',async()=>{try{await api(`/api/requests/${row.id}/resume`,{});await poll();}catch(e){notice(e.message);}});card.append(button);}
  if(row.state==='simulation_qualified'&&row.package?.verified){const download=node('a','Скачать проверенный результат','download');download.href=`/api/requests/${row.id}/package`;card.append(download,node('p','Проверено в симуляции. Подключение и наблюдаемая проверка настоящего робота — следующий отдельный этап.','muted'));}
  root.append(card);
 }
 if(archived.length)root.append(history);
}
async function poll(){if(busy)return;busy=true;try{snapshot=await api('/api/state');$('connection').textContent='На этом Mac · изменения сохраняются';showCurrent(snapshot);showRequests(snapshot);$('models').textContent=snapshot.models.ready?'Модели установлены':'Недостающие модели установятся автоматически';const m=snapshot.memory;$('resources').textContent=`Свободно по оценке системы: ${(m.available_bytes/2**30).toFixed(1)} ГБ · ${snapshot.disk_connected?'диск подключён':'подключите диск ExplorerSim'}`;}catch(e){$('connection').textContent='Переподключаюсь…';}finally{busy=false;}}
$('learn-form').addEventListener('submit',async event=>{event.preventDefault();if(disconnecting)return;const text=$('request').value.trim();if(!text)return;$('learn').disabled=true;try{await api('/api/learn',{text,parent_id:parent,attempts:1000,hours:24});parent=null;$('learn').textContent='Обучить';$('form-status').textContent='Задача сохранена. Текущее обучение не прерывается.';await poll();}catch(e){notice(e.message);}finally{$('learn').disabled=false;}});
$('current-control').addEventListener('click',async()=>{if(!target)return;const selected={...target};$('current-control').disabled=true;try{await api(selected.kind==='queue'?'/api/queue/resume':`/api/${selected.kind}/${selected.id}/${paused.has(selected.state)?'resume':'pause'}`,{});await poll();}catch(e){notice(e.message);}finally{$('current-control').disabled=false;}});
(async()=>{try{token=(await api('/api/session')).token;await poll();setInterval(poll,1500);setInterval(refreshFrames,500);}catch(e){notice(e.message);}})();

$('disconnect').addEventListener('click',async()=>{disconnecting=true;$('disconnect').disabled=true;try{const result=await api('/api/disconnect',{});$('form-status').textContent=result.message;$('learn').disabled=true;$('current-control').disabled=true;}catch(e){disconnecting=false;notice(e.message);$('disconnect').disabled=false;}});
