/* Observed full mission. Loading this panel never starts or resumes motion. */
(()=>{
  let session=null,candidate=null,timer=null;
  const node=document.createElement('div');node.className='card';
  node.innerHTML='<h2>Носок → корзина: полный сценарий</h2><p>Найти оба предмета → подъехать → взять → перевезти → положить → проверить. Используется отдельно проверенный пакет полного сценария.</p><p id="roomCleanupState" class="muted">Проверяю наличие пакета…</p><label><input type="checkbox" id="roomCleanupObserver"> Я рядом и наблюдаю, пространство свободно</label><div class="row"><button id="roomCleanupStart" disabled>Начать наблюдаемую проверку</button><button id="roomCleanupStop" class="danger">СТОП</button></div>';
  document.getElementById('training')?.prepend(node);
  const state=node.querySelector('#roomCleanupState'),start=node.querySelector('#roomCleanupStart');
  async function status(){
    try{
      const data=await(await api('room-cleanup')).json();candidate=data.candidates[0]||null;
      if(data.execution.busy){state.textContent='Выполняется: '+data.execution.phase;start.disabled=true;return;}
      if(!candidate){state.textContent='Проверенного пакета полного сценария пока нет. Обучение и проверка выполняются на Mac.';start.disabled=true;return;}
      const check=await(await api('room-cleanup/'+candidate+'/preflight')).json();
      state.textContent=check.ready?'Пакет проверен в симуляции. Следующий шаг — наблюдаемая попытка на роботе.':check.blocked_by.join(' · ');
      start.disabled=!check.ready;
    }catch(e){state.textContent=e.message;start.disabled=true;}
  }
  async function stop(){
    session=null;clearInterval(timer);timer=null;
    try{await api('room-cleanup/stop',{});}catch(e){state.textContent=e.message;}
    await status();
  }
  async function lease(){
    if(!session)return;
    if(document.hidden||!node.querySelector('#roomCleanupObserver').checked){await stop();return;}
    try{const result=await(await api('room-cleanup/lease',{session,held:true})).json();if(!result.busy)await stop();}
    catch(e){await stop();state.textContent=e.message;}
  }
  start.onclick=async()=>{
    if(!candidate||!node.querySelector('#roomCleanupObserver').checked){state.textContent='Для запуска нужен наблюдающий оператор.';return;}
    start.disabled=true;
    try{const result=await(await api('room-cleanup/start',{candidate,observing:true})).json();session=result.session;timer=setInterval(lease,500);}
    catch(e){state.textContent=e.message;}
    await status();
  };
  node.querySelector('#roomCleanupStop').onclick=stop;
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&session)stop();});
  node.querySelector('#roomCleanupObserver').onchange=()=>{if(session)lease();};
  status();setInterval(()=>{if(!document.hidden)status();},5000);
})();
