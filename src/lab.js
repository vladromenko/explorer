/* Local UI for the existing authenticated API. No external scripts or models. */
(() => {
  let selected='E01', last=null, replayId=null, pendingOwner=null, catalog=[],capture=null,firstCorner=null,roi=null;
  const el=id=>document.getElementById(id);
  const request=async(path,body)=>{const r=await api(path,body);return r.json();};
  const text=(parent,value)=>{const p=document.createElement('p');p.textContent=value;parent.append(p);return p;};
  const button=(parent,label,fn)=>{const b=document.createElement('button');b.textContent=label;b.onclick=fn;parent.append(b);return b;};
  const showError=e=>{el('lab-message').textContent=e.message;};
  async function history(){
    const runs=await request('experiments/results');el('lab-history').replaceChildren();
    if(!runs.length)text(el('lab-history'),'Первый запуск ещё не выполнен.');
    for(const run of runs.slice(0,10)){const row=document.createElement('div');row.className='row';
      const state={completed:'анализ завершён',failed:'нужны условия',cancelled:'отменён'}[run.state]||run.state;
      button(row,run.experiment+' · '+new Date(run.at*1000).toLocaleTimeString()+' · '+state,async()=>{
        try{const r=await request('experiments/results/'+run.id);replayId=run.id;select(run.experiment);render(r);}catch(e){showError(e);}
      });el('lab-history').append(row);}
  }
  function select(id){selected=id;const entry=catalog.find(e=>e.id===id);el('lab-title').textContent=entry.id+' · '+entry.name;
    el('lab-description').textContent=entry.implementation+' · Источник идеи: '+entry.inspiration;
    for(const b of el('lab-cards').querySelectorAll('button'))b.setAttribute('aria-pressed',String(b.dataset.id===id));}
  function render(run){last=run;replayId=run.id;el('lab-export').disabled=false;
    el('lab-message').textContent=(run.result?.summary||run.state)+' · '+(run.elapsed_ms??'?')+' мс';
    el('lab-result').replaceChildren();const r=run.result||{};
    if(r.arm){for(const [kind,label] of [['commanded','Команда'],['measured','Измерение']]){
      const s=r.arm[kind];text(el('lab-result'),label+': '+(s.values?s.values.join('°, ')+'°':'нет достоверных значений')+(s.age_s!=null?' · возраст '+s.age_s.toFixed(2)+' с':''));}
      text(el('lab-result'),r.arm.next_step);}
    if(r.blocked_by?.length)text(el('lab-result'),'Не подтверждено: '+r.blocked_by.join(', '));
    if(r.queue)for(const item of r.queue)text(el('lab-result'),item.request);
    if(r.steps)text(el('lab-result'),'Порядок навыков: '+r.steps.map(x=>x.skill).join(' → '));
    if(r.objects)text(el('lab-result'),'Объектов: '+r.objects.length);
    if(r.error_m!=null)text(el('lab-result'),'Ошибка прогноза: '+(r.error_m*100).toFixed(2)+' см');
    if(r.required)text(el('lab-result'),'Для расчёта нужны: '+r.required.join(', '));
    el('lab-json').textContent=JSON.stringify(run,null,2);
  }
  function payload(){const params=JSON.parse(el('lab-params').value||'{}');params.query=el('lab-query').value;
    if(capture&&roi&&['E09','E10'].includes(selected)){params.capture_id=capture.id;params.bbox=roi;}
    if(el('lab-mode').value==='replay'){if(!replayId)throw Error('Сначала выберите прошлый запуск');params.run_id=replayId;}
    const requestId=Array.from(crypto.getRandomValues(new Uint8Array(16)),b=>b.toString(16).padStart(2,'0')).join('');
    return {experiment:selected,mode:el('lab-mode').value,params,request_id:requestId,issued_at:Date.now()/1000};}
  async function memory(){const r=await request('experiments/memory');el('memory-list').replaceChildren();
    for(const o of r.objects){const row=document.createElement('div');row.className='row';text(row,o.label+' · '+o.id.slice(0,8)+' · '+(o.active?'активен':'удалён из активного списка'));
      if(o.active)button(row,'Исправить: предмета здесь нет',async()=>{const reason=prompt('Укажите подтверждение или причину исправления');
        if(reason)try{await request('experiments/memory',{operation:'remove',object_id:o.id,evidence:{reason}});await memory();}catch(e){el('memory-message').textContent=e.message;}});
      el('memory-list').append(row);}}
  async function telegram(){try{const r=await request('telegram/setup');const s=r.status;pendingOwner=r.pairing.pending;
    el('telegram-confirm').disabled=!pendingOwner;
    el('telegram-status').textContent=pendingOwner?'Проверьте аккаунт: @'+pendingOwner.username+' · ID '+pendingOwner.user_id:
      s.paired?'Владелец привязан · @'+s.username:s.connected?'Бот @'+s.username+' подключён, ждёт привязки':r.token_configured?'Бот запускается или проверяет связь':'Токен ещё не настроен';
  }catch(e){el('telegram-status').textContent=e.message;}}
  async function mount(){try{
    const markup=await(await api('lab.html')).text();const holder=document.createElement('div');holder.innerHTML=markup;
    document.querySelector('main').append(holder.firstElementChild);
    const css=document.createElement('style');css.textContent='#lab{display:none;grid-column:1/-1}body[data-view="lab"] #lab{display:block}body[data-view="lab"] #teach,body[data-view="lab"] .camera-panel{display:none}.lab-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px}.lab-grid button{text-align:left;font-size:14px}.lab-grid button[aria-pressed="true"]{border-color:#7ce8b5;background:#295243}#lab input{min-width:0;max-width:95%}#lab pre{max-height:500px;overflow:auto}';document.head.append(css);
    button(document.querySelector('nav'),'Эксперименты',()=>{release();document.body.dataset.view='lab';history().catch(showError);memory().catch(showError);telegram();});
    catalog=(await request('experiments')).experiments;
    for(const e of catalog){const b=button(el('lab-cards'),e.id+' · '+e.name,()=>select(e.id));b.dataset.id=e.id;
      const badge=document.createElement('small');badge.style.display='block';badge.textContent='Анализ доступен · физический допуск отсутствует';b.append(badge);}
    select('E01');
    const profile=await request('resources/profile');el('resource-profile').value=profile.mode;
    el('resource-message').textContent=profile.limits;
    el('resource-apply').onclick=async()=>{try{const r=await request('resources/profile',{mode:el('resource-profile').value});el('resource-message').textContent='Профиль: '+r.label;}catch(e){el('resource-message').textContent=e.message;}};
    el('lab-capture').onclick=async()=>{try{capture=await request('experiments/capture',{});firstCorner=null;roi=null;
      const response=await api('experiments/capture/'+capture.id);const url=URL.createObjectURL(await response.blob());const img=new Image();
      img.onload=()=>{const canvas=el('lab-roi');canvas.width=capture.width;canvas.height=capture.height;canvas.getContext('2d').drawImage(img,0,0);URL.revokeObjectURL(url);};img.src=url;
      el('lab-roi-message').textContent='Нажмите левый верхний и правый нижний углы предмета.';
    }catch(e){el('lab-roi-message').textContent=e.message;}};
    el('lab-roi').onclick=event=>{if(!capture)return;const canvas=el('lab-roi');const rect=canvas.getBoundingClientRect();
      const point=[Math.round((event.clientX-rect.left)*canvas.width/rect.width),Math.round((event.clientY-rect.top)*canvas.height/rect.height)];
      if(!firstCorner){firstCorner=point;el('lab-roi-message').textContent='Теперь противоположный угол.';}
      else{roi=[Math.min(firstCorner[0],point[0]),Math.min(firstCorner[1],point[1]),Math.max(firstCorner[0],point[0]),Math.max(firstCorner[1],point[1])];
        const ctx=canvas.getContext('2d');ctx.strokeStyle='#7ce8b5';ctx.lineWidth=3;ctx.strokeRect(roi[0],roi[1],roi[2]-roi[0],roi[3]-roi[1]);
        firstCorner=null;el('lab-roi-message').textContent='Область выбрана. Откройте E10 и нажмите «Запустить без движения».';}};
    el('lab-preflight').onclick=async()=>{try{const r=await request('experiments/preflight',payload());el('lab-message').textContent=r.ready?'Готово: анализ без доступа к моторам':'Недоступно';}catch(e){showError(e);}};
    el('lab-run').onclick=async()=>{el('lab-run').disabled=true;el('lab-message').textContent='Выполняю…';try{render(await request('experiments/run',payload()));await history();}catch(e){showError(e);}finally{el('lab-run').disabled=false;}};
    el('lab-cancel').onclick=async()=>{try{await request('experiments/cancel',{});el('lab-message').textContent='Анализ отменён. Команды движения этот режим не отправляет.';}catch(e){showError(e);}};
    el('lab-export').onclick=()=>{if(last){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(last,null,2)],{type:'application/json'}));a.download='Explorer-'+last.experiment+'-'+last.id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}};
    el('memory-add').onclick=async()=>{try{await request('experiments/memory',{operation:'add',label:el('memory-label').value,evidence:{reason:'Явное добавление владельцем'}});await memory();el('memory-message').textContent='Сохранён отдельный предмет.';}catch(e){el('memory-message').textContent=e.message;}};
    el('experience-save').onclick=async()=>{try{await request('experiments/label',{task:el('experience-task').value,condition:el('experience-condition').value,outcome:el('experience-outcome').value,reason:el('experience-reason').value});el('experience-message').textContent='Оценка сохранена. Откройте E18, чтобы увидеть очередь исправлений.';}catch(e){el('experience-message').textContent=e.message;}};
    el('telegram-pair').onclick=async()=>{try{const r=await request('telegram/pairing',{});el('telegram-code').textContent=r.command;await telegram();}catch(e){el('telegram-status').textContent=e.message;}};
    el('telegram-confirm').onclick=async()=>{if(pendingOwner)try{await request('telegram/confirm',{user_id:pendingOwner.user_id});el('telegram-code').textContent='';await telegram();}catch(e){el('telegram-status').textContent=e.message;}};
    setInterval(()=>{if(document.body.dataset.view==='lab')telegram();},2500);
  }catch(e){if(!el('lab'))setTimeout(mount,3000);}}
  mount();
})();
