'use strict';
const $ = id => document.getElementById(id);
let depthView = false;
let token = '', session = null, latest = null, selectedJoint = 0, arm = Array(6).fill(0), keys = new Set(), held = {arm: Array(6).fill(0), base: [0, 0, 0]}, recording = false, busy = false, gamepadButtons = [];
const jointNames = ['Поворот основания', 'Подъём плеча', 'Локоть', 'Наклон кисти', 'Поворот кисти', 'Захват'];
const jobNames = {skills: 'Подготовка коротких навыков', skill_trial: 'Проверка этапов',compare:'Сравнение моделей на одинаковых сценах',cycle:'Доработка до готовности',audit:'Настройка обратной связи',trials: 'Попытки', train: 'Обучение ACT', validate: 'Проверка', export: 'Пакет переноса', night: 'Автоматическое обучение', deploy: 'Загрузка на Jetson', assistant: 'Ваша учебная задача'};
const statusNames = {running: 'в работе', complete: 'завершено', failed: 'ошибка', exporting: 'подготовка данных', training: 'обучение', validating: 'проверка', validated_simulation: 'проверено по записям', trained_simulation_unvalidated: 'обучено; успешность задачи не подтверждена', diagnosed:'диагностика завершена; результат задачи проверяется отдельно',prepared:'данные подготовлены; исполнение не принято',cancelled: 'остановлено', queued: 'ожидает', paused: 'приостановлено', interrupted: 'прервано; можно продолжить', corrupt: 'файл повреждён'};
const phases = {skill_find:'Ищет предмет по глубине',skill_inspect_destination:'Осматривает корзину',skill_approach:'Подъезжает к измеренной точке',skill_reobserve:'Измеряет цель заново',skill_pregrasp:'Подводит раскрытый захват',skill_grasp:'Захватывает предмет',skill_lift:'Поднимает предмет',skill_verify_hold:'Проверяет удержание по двум кадрам',skill_find_destination:'Ищет корзину',skill_carry:'Переносит подтверждённый предмет',skill_support:'Проверяет точку отпускания',skill_release:'Раскрывает захват',skill_withdraw:'Отводит руку',skill_verify_place:'Проверяет размещение',approach: 'Подведите захват к предмету', reach: 'Подведите захват к предмету', grasp: 'Замкните захват', lift: 'Поднимите предмет', transport: 'Перенесите к корзине', place: 'Отпустите в корзину', release_retreat: 'Отводит раскрытый захват', release_unhook: 'Поворачивает кисть, чтобы освободить ткань', inspect_floor: 'Осматривает пол', success: 'Объект доставлен', navigate_source: 'Едет в соседнюю комнату', navigate_destination: 'Перевозит предмет через дверь', idle: 'Готов к вашему показу', align_base: 'Подъезжает к объекту', lower: 'Опускает захват', close: 'Замыкает захват', policy: 'Обученная политика пробует выполнить задачу'};
function notify(text, error = false) { $('notice').textContent = text; $('notice').classList.toggle('error', error); $('notice').classList.remove('hidden'); clearTimeout(notify.timer); notify.timer = setTimeout(() => $('notice').classList.add('hidden'), 4500); }
async function api(path, body, refreshSession = true) {
  const options = body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Explorer-Sim-Token': token}, body: JSON.stringify(body)};
  const response = await fetch(path, options);
  if (response.status === 403 && body !== undefined && refreshSession) {
    const renewed = await api('/api/session'); token = renewed.token;
    return api(path, body, false);
  }
  if (!response.ok) { let detail; try { detail = (await response.json()).detail; } catch (_) { detail = response.statusText; } throw new Error(typeof detail === 'string' ? detail : 'Проверьте параметры задания'); }
  return response.json();
}
function button(id, action) { $(id).addEventListener('click', async () => { const element = $(id); element.disabled = true; try { await action(); } catch (error) { notify(error.message, true); } finally { element.disabled = false; } }); }
function makeJoints() {
  jointNames.forEach((name, index) => {
    const row = document.createElement('div'); row.className = 'joint-row'; row.tabIndex = 0;
    const label = document.createElement('span'); label.textContent = name; row.append(label);
    const value = document.createElement('output'); value.id = `joint-${index}`; value.textContent = '—'; row.append(value);
    [-1, 1].forEach(direction => { const key = document.createElement('button'); key.textContent = direction < 0 ? '−' : '+'; key.setAttribute('aria-label', `${name}: ${direction < 0 ? 'уменьшить' : 'увеличить'}`); holdButton(key, () => { selectedJoint = index; held.arm[index] = direction; selectJoint(); }, () => { held.arm[index] = 0; }); row.append(key); });
    row.addEventListener('click', () => { selectedJoint = index; selectJoint(); }); row.addEventListener('keydown', event => { if (event.code === 'Enter') { selectedJoint = index; selectJoint(); } }); $('joints').append(row);
  }); selectJoint();
}
function selectJoint() { [...$('joints').children].forEach((row, index) => row.classList.toggle('selected', index === selectedJoint)); }
function holdButton(element, start, end) { element.addEventListener('pointerdown', event => { event.preventDefault(); element.setPointerCapture(event.pointerId); start(); }); ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(name => element.addEventListener(name, end)); }
function zeroControls(cancelTask = false) { keys.clear(); held = {arm: Array(6).fill(0), base: [0, 0, 0]}; if (token) api(cancelTask === true ? '/api/stop' : '/api/stop-manual', {}).catch(() => {}); }
function deadzone(number) { return Math.abs(number || 0) < .15 ? 0 : number; }
function padControl() {
  const pad = navigator.getGamepads ? [...navigator.getGamepads()].find(Boolean) : null;
  $('gamepad').textContent = pad ? 'Геймпад подключён' : 'Геймпад не подключён';
  if (!pad) { gamepadButtons = []; return {arm: 0, base: [0, 0, 0], grip: 0}; }
  const buttons = pad.buttons.map(item => item.pressed);
  if (buttons[4] && !gamepadButtons[4]) selectedJoint = (selectedJoint + 5) % 6;
  if (buttons[5] && !gamepadButtons[5]) selectedJoint = (selectedJoint + 1) % 6;
  gamepadButtons = buttons; selectJoint();
  return {arm: -deadzone(pad.axes[3]), base: [-deadzone(pad.axes[1]) * .2, -deadzone(pad.axes[0]) * .2, ((pad.buttons[6]?.value || 0) - (pad.buttons[7]?.value || 0)) * .6], grip: Number(buttons[0]) - Number(buttons[1])};
}
let controlInFlight = false, inputEnabled = true;
async function sendControls() {
  if (!token || controlInFlight || !latest || document.hidden || !inputEnabled) return;
  const activeInput = ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName);
  const is = key => !activeInput && keys.has(key) ? 1 : 0;
  const pad = padControl(), speed = Number($('speed').value);
  const velocity = held.arm.map(value => value * speed);
  velocity[selectedJoint] += (is('ArrowUp') - is('ArrowDown') + pad.arm) * speed;
  velocity[5] += (is('BracketRight') - is('BracketLeft') + pad.grip) * speed;
  const base = [(is('KeyW') - is('KeyS')) * .2, (is('KeyA') - is('KeyD')) * .2, (is('KeyQ') - is('KeyE')) * .6].map((value, index) => value + held.base[index] + pad.base[index]);
  controlInFlight = true;
  try { await api('/api/control', {arm_velocity: velocity, base}); } catch (_) {} finally { controlInFlight = false; }
}
function showResult(result) {
  if (!result || typeof result !== 'object') return '';
  const pieces = [];
  const successes = result.physics_verified_successes ?? result.successes;
  const total = result.episodes ?? result.count ?? result.attempts;
  const validation = result.validation;
  if (validation?.mae !== undefined) pieces.push(`Ошибка по независимым записям: ${Number(validation.mae).toFixed(3)}`);
  if (typeof successes === 'number') pieces.push(`Доставок в симуляции: ${successes}${typeof total === 'number' ? ' / ' + total : ''}`);
  if (typeof result.validation_mae === 'number') pieces.push(`Ошибка по записям: ${result.validation_mae.toFixed(3)}`);
  if (result.state || result.status) pieces.push(statusNames[result.state || result.status] || result.state || result.status);
  if (result.physics_validated === false) pieces.push('Динамика настоящего робота ещё не идентифицирована');
  if (result.scene_coverage) pieces.push(`Высоты корзин: ${result.scene_coverage.basket_heights_m.map(h=>Math.round(h*100)+' см').join(', ') || 'нет данных'}`);
  if (result.message) pieces.push(result.message);
  return pieces.join(' · ');
}
function renderStatus(snapshot) {
  latest = snapshot; renderLearning(snapshot.learning_status); renderSkills(snapshot); $('depth-view').disabled = !snapshot.depth_available; const state = snapshot.state || {}, task = state.task || {}, command = state.command || state.command9_deg || [];
  arm = command.slice(0, 6); jointNames.forEach((_, index) => { $(`joint-${index}`).textContent = Number.isFinite(arm[index]) ? `${arm[index].toFixed(0)}°` : '—'; });
  const skillScene=String(task.phase || '').startsWith('skill_');
  const skillRunning=(snapshot.jobs || []).some(job=>job.kind==='skill_trial' && job.status==='running');
  $('phase').textContent = snapshot.error ? snapshot.error : skillScene && !skillRunning ? 'Проверка завершена · результат сохранён выше' : skillRunning ? phases[task.phase] || 'Проверка этапов' : snapshot.paused ? 'Сцена на паузе' : task.success ? 'Объект доставлен · подтверждено моделью' : phases[task.phase] || 'Управляйте рукой и платформой';
  $('contact').textContent = task.grasp_contact===true ? 'Есть' : task.grasp_contact===false ? 'Нет' : 'Не измерен'; $('lifted').textContent = task.lifted===true ? 'Да' : task.lifted===false ? 'Нет' : 'Не измерено'; $('placed').textContent = task.placed===true ? 'Да' : task.placed===false ? 'Нет' : 'Не измерено'; $('simtime').textContent = `${Number(state.sim_time || 0).toFixed(1)} с`;
  $('pause').textContent = snapshot.paused ? 'Продолжить' : 'Пауза'; recording = snapshot.recording; $('record').textContent = recording ? 'Закончить запись' : 'Начать запись'; $('record-status').textContent = recording ? `● ${snapshot.record_frames} кадров` : ''; $('annotation').disabled = !recording;
  const jobs = snapshot.jobs || []; busy = jobs.some(job => ['running', 'exporting', 'training', 'validating'].includes(job.status));
  ['cycle-start', 'cycle-resume', 'trials', 'train', 'show-policy', 'validate', 'export', 'night-start', 'night-resume', 'deploy', 'request-learn', 'request-plan', 'request-reply', 'request-partial', 'request-resume', 'request-deploy', 'queue-task'].forEach(id => { $(id).disabled = busy; }); $('cancel-job').classList.toggle('hidden', !busy); $('reset').disabled = busy; $('task-select').disabled = busy || recording; $('floor-view').disabled = busy;
  renderGoals(snapshot.learning_goals || []);
  $('request-cancel').classList.toggle('hidden',!busy);
  const activeJob=jobs.at(-1);$('request-progress').textContent=busy?`${jobNames[activeJob.kind]}: ${activeJob.message || 'В работе'}`:activeJob?.status==='failed'?`Не завершено: ${activeJob.message}`:'';
  document.querySelectorAll('.night-tasks input, #night-attempts, #night-steps, #night-hours, #night-varied').forEach(input => { input.disabled = busy; });
  renderNight(snapshot.night_batches || []); const current = jobs.at(-1); $('job-state').firstElementChild.textContent = current ? `${jobNames[current.kind]} · ${current.message || statusNames[current.status] || current.status}` : 'Готов к работе'; $('job-state').querySelector('progress').value = current?.progress || 0;
  if (jobs.length) { $('jobs').replaceChildren(); jobs.slice().reverse().forEach(job => { const row = document.createElement('div'); row.className = 'job'; const title = document.createElement('strong'); title.textContent = jobNames[job.kind]; const status = document.createElement('span'); status.textContent = statusNames[job.status] || job.status; const detail = document.createElement('small'); detail.style.display = 'block'; detail.textContent = showResult(job.result) || job.message || ''; row.append(title, status, detail); $('jobs').append(row); }); }
  $('logs').replaceChildren(); (snapshot.logs || []).slice().reverse().forEach(log => { const row = document.createElement('div'), stamp = document.createElement('time'); stamp.textContent = log.time; row.append(stamp, document.createTextNode(log.message)); $('logs').append(row); });
  const model = snapshot.model; $('model-status').textContent = model ? `Модель ${model.id} · ${model.task_validated_simulation ? 'навык проверен на независимых сценах' : 'полная проверка навыка ещё не пройдена'} · доставок ${model.successes}/${model.episodes}` : 'Сначала обучите модель. «Показать навык» запускает её на одной новой сцене.'; $('download').classList.toggle('hidden', !snapshot.package_available); $('storage').textContent = `Хранилище: ${snapshot.data_root}`; $('connection').textContent = 'Симуляция работает';
}
function renderLearning(info) {
  if (!info) return;
  const names={needs_diagnosis:'Нужен разбор причин',running:'Обучение продолжается',needs_more_training:'Нужна доработка',ready_for_transfer:'Готово к переносу',paused:'Обучение на паузе',interrupted:'Можно продолжить',failed:'Нужна проверка ошибки',corrupt:'Повреждена запись состояния'};
  const stages={compare:'Сравниваем новую и опорную модель на одинаковых сценах',diagnosis:'Повтор остановлен: улучшение не подтверждено',audit:'Сравниваем частоту зрительной обратной связи',trials:'Собираем новые экспертные показы',train:'Обучаем модель ACT',validate:'Проверяем самостоятельное выполнение',export:'Собираем проверенный пакет',finished:'Последний цикл завершён',ready:'Пакет готов'};
  $('cycle-title').textContent=info.title;
  $('cycle-state').textContent=names[info.state] || info.state;
  const teacher=info.phase==='trials' && info.teacher_profile==='planned_release_v4'?' · проверяем захват и отпускание ткани':info.phase==='trials' && info.teacher_profile==='planned_grasp_v3'?' · проверяем удержание носка':'';
  $('cycle-phase').textContent=(stages[info.phase] || info.phase)+' · итерация '+info.iteration+teacher+(info.message?' · '+info.message:'');
  $('cycle-attempts').textContent=info.demonstrations.attempts.toLocaleString('ru-RU');
  $('cycle-examples').textContent=info.demonstrations.successful.toLocaleString('ru-RU');
  $('cycle-check').textContent=info.episodes?`${info.successes} / ${info.episodes}`:'Ещё не выполнена';
  $('cycle-steps').textContent=info.training_budget?`${Number(info.steps_completed || 0).toLocaleString('ru-RU')} / ${info.training_budget.effective_steps.toLocaleString('ru-RU')}`:'—';
  const eta=info.eta_s?` До завершения обучения ориентировочно ${Math.ceil(info.eta_s/60)} мин; затем проверка.`:'';
  $('cycle-progress').value=info.phase==='train' && info.training_budget?100*(info.steps_completed || 0)/info.training_budget.effective_steps:info.progress || 0;
  const next=info.simulation_ready?'Модель прошла две отдельные проверки. Подключите Jetson для загрузки и физической проверки.':info.state==='running'?'После обучения цикл сравнит новую модель с прежней. Повтор допускается при подтверждённом улучшении или один раз после проверенного исправления показов.':info.state==='paused' || info.state==='interrupted'?'Нажмите «Продолжить сохранённое обучение». Записи и готовые этапы сохраняются.':'Готовность ещё не подтверждена. «Довести выбранный навык до готовности» запускает автоматическую доработку с сохранением результатов.';
  $('cycle-next').textContent=info.state==='needs_diagnosis'?(info.message || 'Новая модель не улучшила результат. Автоматический повтор остановлен для диагностики.')+(info.comparison?` Опорная модель: ${info.comparison.reference_successes}/${info.comparison.episodes}; новая: ${info.comparison.candidate_successes}/${info.comparison.episodes} на тех же сценах.`:''):next+eta;
  $('cycle-pause').classList.toggle('hidden',info.state!=='running');
  $('cycle-resume').classList.toggle('hidden',!info.cycle_id || !['paused','interrupted'].includes(info.state));
  $('cycle-download').classList.toggle('hidden',!info.simulation_ready); if(info.cycle_id)$('cycle-download').href='/api/cycles/'+encodeURIComponent(info.cycle_id)+'/package';
  $('cycle-storage').textContent=`Хранилище: ${latest.data_root} · модель ${info.model_id || 'ещё не создана'} · цикл ${info.cycle_id || info.plan_id || 'ещё не запущен'}`;
  $('cycle-preview-note').textContent=info.phase==='train'?'Обучение нейросети идёт без движения. Здесь кадр из сохранённого примера.':info.phase==='audit'?'Настройки сравниваются в отдельных процессах. Здесь последняя записанная сцена; счётчик сравнения показан выше.':'Камера текущей попытки либо последней сохранённой сцены.';
}
button('cycle-start',async()=>{zeroControls();await checkNight();const tasks=selectedTasks();if(!tasks.length)throw new Error('Выберите хотя бы одну задачу');const goalId=currentGoal?.plan?.fully_supported && JSON.stringify([...currentGoal.plan.task_ids].sort())===JSON.stringify([...tasks].sort())?currentGoal.id:null;await api('/api/jobs/cycle',{task_ids:tasks,goal_id:goalId,steps:Number($('night-steps').value),count:Number($('night-attempts').value),hours:Number($('night-hours').value),workers:4,scene_profile:$('night-varied').checked?'sock_variants_v1':'baseline'});notify('Автоматическая доработка запущена');await poll();});
button('cycle-resume',async()=>{await api('/api/jobs/cycle',{cycle_id:latest.learning_status.cycle_id});await poll();});
button('cycle-pause',async()=>{await api('/api/cancel',{});notify('Останавливаю вычисления. Продолжение начнётся с последнего контрольного сохранения.');await poll();});

let workerViewsBusy = false;
async function refreshWorkerViews() {
  if (document.hidden || workerViewsBusy) return;
  workerViewsBusy = true;
  try {
    const response = await fetch('/static/learning-live/state.json?t='+Date.now(), {cache:'no-store'});
    if (!response.ok) throw new Error('views unavailable');
    const state = await response.json(), stale = Date.now()/1000-state.updated_at>10;
    const text = {trials:'До четырёх отдельных сцен: исполнители собирают новые показы параллельно.',audit:'До четырёх отдельных сцен: сравниваем настройки зрительной обратной связи.',validate:'Самостоятельная проверка идёт в одной сцене. Остальные исполнители ждут.',train:'Сейчас одна модель ACT учится на всех накопленных показах. Роботы не двигаются; ниже последние кадры четырёх разных сохранённых попыток.',idle:'Нет активных сцен. Ниже последние сохранённые примеры.'};
    $('workers-phase').textContent=stale?'Связь с камерами потеряна. Последние кадры не являются прямой трансляцией.':text[state.phase] || 'Сохранённые сцены';
    const count=Math.max(4,...state.views.map(view=>view.slot+1));
    for(let slot=0;slot<count;slot++) {
      let card=$('worker-'+slot);
      if(!card) {card=document.createElement('div');card.id='worker-'+slot;card.className='worker-card';$('worker-grid').append(card);}
      const view=state.views.find(view=>view.slot===slot), camera=$('worker-camera').value;
      if(view) {
        const url=view.cameras[camera] || view.cameras.wrist || view.cameras.overview;
        let img=card.querySelector('img');
        if(!img) {card.replaceChildren();img=document.createElement('img');img.alt='Камера исполнителя '+(slot+1);card.append(img);const label=document.createElement('div');label.className='worker-label';card.append(label);}
        if(!img.dataset.loading && img.dataset.frame!==String(view.captured_at)+url) {img.dataset.loading='1';img.dataset.frame=String(view.captured_at)+url;img.onload=()=>delete img.dataset.loading;img.onerror=()=>{delete img.dataset.loading;delete img.dataset.frame;};img.src=url+'?t='+view.captured_at;}
        const label=card.querySelector('.worker-label');label.replaceChildren();const title=document.createElement('strong');title.textContent='Исполнитель '+(slot+1);const note=document.createElement('span');note.textContent=view.label+(!view.live?' · запись':stale||view.stale?' · кадр устарел':' · в работе')+(camera==='overview'&&!view.cameras.overview?' · камера на руке':'');label.append(title,note);
      } else {
        card.replaceChildren();const empty=document.createElement('div');empty.className='worker-empty';empty.textContent='Исполнитель '+(slot+1)+' · '+(state.phase==='train'?'не двигается во время обучения ACT':'ожидает следующую сцену');card.append(empty);
      }
    }
  } catch (_) {$('workers-phase').textContent='Камеры исполнителей временно недоступны. Состояние обучения показано выше.';}
  finally {workerViewsBusy=false;}
}

function renderNight(plans) {
  const root=$('night-results'); root.replaceChildren();
  const plan=plans[0]; $('night-resume').classList.toggle('hidden', !plan || !['paused','interrupted'].includes(plan.state));
  if (!plan) return;
  const heading=document.createElement('p'); heading.textContent=`Очередь ${plan.id}: ${statusNames[plan.state] || plan.state} · ${plan.reason || ''}`; root.append(heading);
  if(plan.options?.scene_profile==='sock_variants_v1' && plan.tasks.some(task=>['sock_basket','cloth_basket'].includes(task.task_id))){const details=document.createElement('p');details.textContent=`Разные носки и корзины, включая круглую 25 × 30 см · ${plan.options.count} начальных попыток · ${plan.options.steps} шагов обучения · до ${plan.hours} ч`;root.append(details);}
  plan.tasks.forEach(task=>{const row=document.createElement('p'); const report=task.stages.validate; const stateLabel=task.state==='complete'?(task.simulation_qualified?'проверено в симуляции':'расчёт завершён; навык ещё не прошёл проверку'):(statusNames[task.state] || task.state); row.textContent=`${({sock_basket:'Носок → корзина',cookie_delivery:'Печенье → поднос',toy_box:'Кубик → коробка',cloth_basket:'Ткань → бельё'})[task.task_id] || task.task_id}: ${stateLabel} · ${report ? 'Доставок '+report.successes+'/'+report.episodes : task.error || 'Ожидает проверки'}`; root.append(row);});
}
let currentGoal = null, goalSignature = '', selectedGoalId = null, goalHistorySignature = '';
const readableGoalTitle = goal => ({sock_basket:'Носок → корзина',cookie_delivery:'Печенье в упаковке → поднос',toy_box:'Кубик → коробка',cloth_basket:'Ткань → бельё'})[goal.plan?.title] || goal.plan?.title || 'Повреждённая запись';
function renderGoals(goals) {
  const history=$('request-history');
  const historySignature=JSON.stringify(goals.map(goal=>[goal.id,goal.plan?.title]));
  if(historySignature!==goalHistorySignature){
    goalHistorySignature=historySignature;history.replaceChildren();
    goals.forEach(goal=>{const option=document.createElement('option');option.value=goal.id;option.textContent=readableGoalTitle(goal)+' · '+new Date(goal.created*1000).toLocaleString('ru-RU');history.append(option);});
  }
  $('request-history-label').classList.toggle('hidden',goals.length<2);
  const goal=goals.find(item=>item.id===selectedGoalId) || goals[0]; currentGoal=goal;
  if(goal)history.value=goal.id;
  const signature=JSON.stringify(goal); if(signature===goalSignature)return; goalSignature=signature;
  if(!goal)return;
  const root=$('request-result'); root.replaceChildren();
  const line=(text,tag='p',className='')=>{const node=document.createElement(tag);node.textContent=text;if(className)node.className=className;root.append(node);return node;};
  if(!goal.plan){line(goal.error || 'Просьба не прочитана');['request-reply','request-partial','request-resume','request-deploy','request-download'].forEach(id=>$(id).classList.add('hidden'));return;}
  const plan=goal.plan;
  line(readableGoalTitle(goal),'h2');line('Вы попросили: '+plan.request);
  line(goal.learning_mode==='skills'?'План отдельных этапов. Подготовка не подтверждает самостоятельное исполнение.':plan.explanation);line('Понимание текста: локальная '+plan.language_model.model+' · данные не отправлялись в облако','small');
  if(plan.missing.length || plan.clarification){line('Что пока не входит в готовый сценарий','h3');plan.missing.forEach(text=>line(text,'p','gap'));if(plan.clarification)line(plan.clarification,'p','gap');}
  plan.tasks.forEach(task=>{const box=document.createElement('div');box.className='result-task';const title=document.createElement('h3');title.textContent=task.title;box.append(title);const info=document.createElement('p');info.textContent=task.learns;box.append(info);const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Порядок действий: найти → подъехать → взять → перевезти → отпустить';details.append(summary);const list=document.createElement('ol');task.steps.forEach(step=>{const li=document.createElement('li');li.textContent=step.title;list.append(li);});details.append(list);box.append(details);root.append(box);});
  const resultStates={prepared:'Показы разделены по этапам; экспериментальные модели подготовлены. Самостоятельное исполнение ещё не принято.',planned:'План готов. Можно начать обучение.',needs_decision:'Нужен ваш выбор: переформулируйте просьбу или обучите только доступную часть.',running:'Обучение идёт. Попытки и результаты сохраняются на внешнем диске.',paused:'Очередь приостановлена. Продолжение сохранит готовые этапы.',interrupted:'Приложение перезапускалось. Продолжите сохранённую задачу.',simulation_checked:'Все выбранные задачи прошли проверку в симуляции. Подключите Jetson, загрузите модели и выполните наблюдаемую проверку.',partial_simulation_checked:'Доступная часть прошла симуляцию. Исходная просьба ещё не выполнена полностью; недостающие этапы перечислены выше.',needs_more_training:'Расчёт закончен, но модель ещё не выполняет задачу устойчиво. Не используйте её как готовый автономный навык.',failed:'Задание завершилось ошибкой.',cancelled:'Задание остановлено.'};
  line(resultStates[goal.state] || 'Состояние: '+goal.state,'p',goal.state==='simulation_checked'?'ready':'gap');
  if(goal.error)line(goal.error);
  (goal.results || []).forEach(result=>line(`${result.title}: ${result.successes ?? '—'}/${result.episodes ?? '—'} успешных новых сцен${result.error?' · '+result.error:''}`));
  line(goal.learning_mode==='skills'?'Новый исполнитель проверяется отдельно. Передача на настоящий робот пока недоступна.':'Архив прежнего ACT: результаты и модели сохранены. Новая подготовка продолжит работу с отдельными этапами.');
  const resumable=['paused','interrupted'].includes(goal.state);
  $('request-resume').classList.toggle('hidden',!resumable);
  $('request-reply').classList.toggle('hidden',!plan.clarification && !plan.missing.length);
  $('request-partial').classList.toggle('hidden',!plan.task_ids.length || Boolean(goal.night_plan_id) || resumable);
  $('request-partial').textContent=plan.fully_supported?'Подготовить навыки по плану':'Подготовить доступную часть';
  const downloadable=Boolean(goal.archive);
  $('request-deploy').classList.toggle('hidden',!downloadable);$('request-download').classList.toggle('hidden',!downloadable);
  $('request-download').href='/api/goals/'+encodeURIComponent(goal.id)+'/package';
}
async function languageRequest(autoTrain,reply=false){zeroControls();const text=$('learning-request').value.trim();if(!text)throw new Error('Напишите, чему научить робота');const parent=reply?currentGoal.id:null;await api('/api/jobs/assistant',{request_text:text,goal_id:parent,auto_train:autoTrain,learning_mode:'skills',scene_profile:'sock_variants_v1',count:Number($('request-attempts').value),steps:Number($('request-steps').value),hours:Number($('request-hours').value),batch_size:16,seed:Math.floor(Math.random()*1000000)});selectedGoalId=null;goalSignature='';notify('Помощник разбирает просьбу на этом Mac');await poll();}
$('request-history').addEventListener('change',()=>{selectedGoalId=$('request-history').value;goalSignature='';renderGoals(latest?.learning_goals || []);if(currentGoal?.plan)$('learning-request').value=currentGoal.plan.request;});
button('request-learn',()=>languageRequest(true));button('request-plan',()=>languageRequest(false));
button('request-reply',()=>languageRequest(true,true));
button('request-cancel',async()=>{await api('/api/cancel',{});notify('Обучение останавливается; готовые этапы сохранятся');});
button('request-partial',async()=>{if(!currentGoal)throw new Error('Сначала составьте план');await api('/api/jobs/assistant',{goal_id:currentGoal.id,learning_mode:'skills',allow_partial:true});await poll();});
button('request-resume',async()=>{await api('/api/jobs/assistant',{goal_id:currentGoal.id,learning_mode:'skills',allow_partial:!currentGoal.plan.fully_supported});await poll();});
button('request-deploy',async()=>{await api('/api/jobs/deploy',{goal_id:currentGoal.id});notify('Передача моделей этой просьбы началась; робот не двигается');await poll();});
document.querySelectorAll('[data-request]').forEach(button=>button.addEventListener('click',()=>{$('learning-request').value=button.dataset.request;$('learning-request').focus();}));
async function checkNight(){const result=await api('/api/night-readiness');$('night-ready').textContent=result.ready?`Готов: ${result.device}, свободно ${result.free_gib} ГБ, модели установлены`:'Не готов: '+result.blocked_by.join('; ');if(!result.ready)throw new Error(result.blocked_by.join('; '));return result;}
button('night-check',checkNight);
const taskChoices=[...document.querySelectorAll('.night-tasks input')];
function selectedTasks(){return taskChoices.filter(input=>input.checked).map(input=>input.value);}
function showTaskSelection(){const titles=taskChoices.filter(input=>input.checked).map(input=>input.closest('label').querySelector('strong').textContent);$('night-selection').textContent=titles.length?'Выбрано: '+titles.join(' · ')+'. Задачи выполняются последовательно.':'Выберите хотя бы одну задачу.';}
function saveTaskSelection(){showTaskSelection();try{localStorage.setItem('explorer-sim-selected-tasks',JSON.stringify(selectedTasks()));}catch(_){} }
try{const saved=JSON.parse(localStorage.getItem('explorer-sim-selected-tasks'));if(Array.isArray(saved)&&saved.every(id=>taskChoices.some(input=>input.value===id)))taskChoices.forEach(input=>{input.checked=saved.includes(input.value);});}catch(_){}
taskChoices.forEach(input=>input.addEventListener('change',saveTaskSelection));showTaskSelection();
button('queue-task',()=>{const choice=taskChoices.find(input=>input.value===$('task-select').value);if(choice)choice.checked=true;saveTaskSelection();$('automatic-learning').scrollIntoView({behavior:'smooth',block:'start'});});
button('night-start',async()=>{const tasks=selectedTasks();if(!tasks.length)throw new Error('Выберите хотя бы одну задачу для обучения');zeroControls();await checkNight();await api('/api/jobs/skills',{task_ids:tasks,workers:4,scene_profile:$('night-varied').checked?'sock_variants_v1':'baseline',count:Number($('night-attempts').value),steps:Number($('night-steps').value),hours:Number($('night-hours').value),batch_size:16,seed:Math.floor(Math.random()*1000000)});notify('Подготовка отдельных навыков запущена');await poll();});
button('night-resume',async()=>{const plan=(latest.night_batches || [])[0];await api('/api/jobs/night',{plan_id:plan.id});await poll();});
button('deploy',async()=>{const info=latest.learning_status;if(info.cycle_id && !info.simulation_ready)throw new Error('Текущий цикл ещё не прошёл независимые проверки');await api('/api/jobs/deploy',info.cycle_id?{cycle_id:info.cycle_id}:{});notify('Загрузка началась; движений не будет');await poll();});
$('task-select').addEventListener('change',async()=>{zeroControls();await api('/api/reset',{seed:100,difficulty:$('difficulty').value,task_id:$('task-select').value});episodeSignature='';await poll();await updateEpisodes();});
let episodeSignature = '';
async function updateEpisodes() {
  try {
    const episodes = await api('/api/episodes'); const signature = JSON.stringify(episodes);
    if (signature === episodeSignature) return;
    const known = new Map([...document.querySelectorAll('#episodes input')].map(input => [input.value, input.checked]));
    episodeSignature = signature; $('episode-count').textContent = `(${episodes.length})`;
    if (!episodes.length) return;
    $('episodes').replaceChildren();
    episodes.filter(episode => episode.task_id === $('task-select').value).forEach(episode => { const label = document.createElement('label'), input = document.createElement('input'); input.type = 'checkbox'; input.value = episode.id; input.checked = known.has(episode.id) ? known.get(episode.id) : episode.physics_verified_success === true; const title = document.createElement('span'); title.textContent = `${episode.id} · ${episode.frames} кадров · ${episode.physics_verified_success ? 'доставка подтверждена моделью' : 'без подтверждённой доставки'}`; label.append(input, title); $('episodes').append(label); });
  } catch (_) {}
}
let pollInFlight = false;
async function poll() { if (pollInFlight) return; pollInFlight = true; try { renderStatus(await api('/api/state?compact=true')); } catch (error) { $('connection').textContent = 'Переподключаюсь…'; $('cycle-next').textContent = 'Ожидаю службу симулятора. Сохранённые результаты остаются на диске. '+error.message; } finally { pollInFlight = false; } }
function refreshImages() { if (document.hidden) return; if ($('learning-preview').closest('details').open && !$('learning-preview').dataset.loading) { $('learning-preview').dataset.loading='1'; $('learning-preview').src='/api/learning-preview.jpg?t='+Date.now(); } for (const camera of ['overview', 'wrist']) { const image = $(camera); if (!image.dataset.loading) { image.dataset.loading = '1'; image.src = `/${camera === 'wrist' && depthView ? 'depth' : camera}.jpg?t=${Date.now()}`; } } }
function jobs(kind, validationCount = 10) { return api(`/api/jobs/${kind}`, {count: kind === 'validate' ? validationCount : Number($('attempts').value), steps: Number($('steps').value), batch_size: Number($('batch').value), task_id: $('task-select').value, episode_ids: kind === 'train' ? [...document.querySelectorAll('#episodes input:checked')].map(input => input.value) : null, seed: kind === 'validate' ? 10000 : 100}).then(() => { notify('Задание началось'); poll(); }); }
$('rgb-view').addEventListener('click', () => { depthView = false; $('rgb-view').classList.add('selected'); $('depth-view').classList.remove('selected'); $('depth-note').classList.add('hidden'); refreshImages(); });
$('depth-view').addEventListener('click', () => { depthView = true; $('depth-view').classList.add('selected'); $('rgb-view').classList.remove('selected'); $('depth-note').classList.remove('hidden'); refreshImages(); });
button('floor-view', async () => { zeroControls(); await api('/api/floor-view', {}); notify('Рука переходит к обзору пола'); });
button('cancel-job', async () => { await api('/api/cancel', {}); notify('Задание останавливается'); });
button('reset', async () => { zeroControls(); await api('/api/reset', {seed: Math.floor(Math.random() * 1000000), difficulty: $('difficulty').value, task_id: $('task-select').value}); await poll(); });
button('pause', async () => { zeroControls(); await api('/api/pause', {}); await poll(); });
button('stop', async () => { zeroControls(true); notify('Движение остановлено'); });
button('record', async () => { if (recording) await api('/api/record/stop', {annotation: $('annotation').value}); else await api('/api/record/start', {}); await poll(); });
['trials', 'train', 'validate', 'export'].forEach(kind => button(kind, () => jobs(kind)));
button('show-policy', () => jobs('validate', 1));
$('tab-arm').addEventListener('click', () => { $('arm-controls').classList.remove('hidden'); $('base-controls').classList.add('hidden'); $('tab-arm').classList.add('selected'); $('tab-base').classList.remove('selected'); });
$('tab-base').addEventListener('click', () => { $('base-controls').classList.remove('hidden'); $('arm-controls').classList.add('hidden'); $('tab-base').classList.add('selected'); $('tab-arm').classList.remove('selected'); });
$('speed').addEventListener('input', () => { $('speed-value').textContent = `${$('speed').value} °/с`; });
document.querySelectorAll('[data-drive]').forEach(element => { const direction = element.dataset.drive.split(',').map(Number); holdButton(element, () => { held.base = direction.map((value, index) => value * (index === 2 ? .6 : .2)); }, () => { held.base = [0, 0, 0]; }); });
window.addEventListener('keydown', event => { if (['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName)) return; if (event.code === 'Space') { event.preventDefault(); zeroControls(true); return; } if (['ArrowUp', 'ArrowDown', 'BracketLeft', 'BracketRight', 'KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE'].includes(event.code)) { event.preventDefault(); keys.add(event.code); } if (/^Digit[1-6]$/.test(event.code)) { selectedJoint = Number(event.code.slice(-1)) - 1; selectJoint(); } });
window.addEventListener('keyup', event => keys.delete(event.code)); window.addEventListener('blur', () => { inputEnabled = false; zeroControls(); }); window.addEventListener('focus', () => { inputEnabled = true; }); window.addEventListener('pagehide', () => zeroControls()); document.addEventListener('visibilitychange', () => { if (document.hidden) zeroControls(); });
for (const camera of ['overview', 'wrist', 'learning-preview']) { $(camera).addEventListener('load', () => delete $(camera).dataset.loading); $(camera).addEventListener('error', () => delete $(camera).dataset.loading); }
makeJoints();
$('worker-camera').addEventListener('change',refreshWorkerViews);
refreshWorkerViews();setInterval(refreshWorkerViews,1000);
(async () => { try { session = await api('/api/session'); token = session.token; await poll(); refreshImages(); setInterval(sendControls, 50); setInterval(poll, 1000); setInterval(refreshImages, 200); updateEpisodes(); setInterval(updateEpisodes, 10000); } catch (error) { notify(`Не удалось запустить: ${error.message}`, true); } })();

function renderSkills(snapshot) {
  const rows=snapshot.skill_workshops || [];
  const skillNames={find:'Поиск предмета',approach:'Подъезд',reobserve:'Новое измерение',pregrasp:'Подход руки',grasp:'Захват',lift:'Подъём',verify_hold:'Проверка удержания',carry:'Поиск корзины и перенос',support:'Размещение над корзиной',release:'Отпускание',withdraw:'Отвод руки',verify_place:'Проверка размещения'};
  const active=(snapshot.jobs || []).find(j=>j.status==='running' && ['skills','skill_trial'].includes(j.kind));
  const list=$('skill-results'); list.replaceChildren();
  for(const row of rows.slice(0,6)) {
    const item=document.createElement('li');
    const names={preparing:'подготовка данных',prepared:'данные и короткие модели подготовлены; исполнение ещё не принято',diagnosed:'проверка завершена',paused:'пауза',failed:'ошибка',running:'проверка этапов'};
    item.textContent=`${({sock_basket:'Носок → корзина',cloth_basket:'Ткань → бельё',toy_box:'Кубик → коробка',cookie_delivery:'Печенье → поднос'})[row.task_id] || row.task_id}: ${names[row.state] || row.state}${row.phase?' · '+(skillNames[row.phase] || phases[row.phase] || row.phase):''}${row.reason?' · '+row.reason:''}`;
    list.append(item);
  }
  if(active) $('skill-progress').textContent=active.message || 'Выполняется проверка отдельного этапа';
  else if(rows.length) $('skill-progress').textContent='Последний результат сохранён. Успех команды, удержания и всей задачи проверяется отдельно.';
  const evidence=$('skill-evidence');evidence.replaceChildren();
  const diagnostic=rows.find(row=>row.execution);
  if(diagnostic && !active){
    const title=document.createElement('strong');title.textContent=diagnostic.success?'Эта сцена выполнена; общей готовности ещё нет':'Проверка остановилась на этапе: '+(skillNames[diagnostic.execution.stage] || diagnostic.execution.stage || 'не определён');evidence.append(title);
    const note=document.createElement('p');note.textContent='Удержание: '+(diagnostic.execution.held_verified?'подтверждено':'не подтверждено')+' · Размещение: '+(diagnostic.execution.placed_verified?'подтверждено':'не подтверждено')+' · Вся задача: '+(diagnostic.success?'выполнена в этой сцене':'не выполнена');evidence.append(note);
    if(diagnostic.reason){const reason=document.createElement('p');reason.textContent=diagnostic.reason;evidence.append(reason);}
    const timeline=document.createElement('ol');timeline.className='skill-timeline';
    for(const event of diagnostic.execution.events || []){const li=document.createElement('li');li.textContent=(skillNames[event.stage] || event.stage)+' · '+({success:'пройден',unknown:'не подтверждён',failure:'ошибка',retry:'повторное измерение'}[event.result.outcome] || event.result.outcome);timeline.append(li);}evidence.append(timeline);
  }
  const inProgress=(snapshot.jobs || []).some(j=>j.status==='running');
  ['skill-run','skill-prepare','skill-resume'].forEach(id=>$(id).disabled=inProgress);
  $('skill-pause').classList.toggle('hidden',!inProgress);
  $('skill-resume').classList.toggle('hidden',!rows.some(r=>['paused','failed','preparing'].includes(r.state) && r.selected_episodes));
}
button('skill-run',async()=>{zeroControls();await api('/api/jobs/skill_trial',{task_id:$('task-select').value,seed:Math.floor(Math.random()*1000000000),scene_profile:'sock_variants_v1'});await poll();});
button('skill-prepare',async()=>{const tasks=selectedTasks();if(!tasks.length)throw new Error('Выберите хотя бы одну задачу');await api('/api/jobs/skills',{task_ids:tasks});await poll();});
button('skill-pause',async()=>{await api('/api/cancel',{});notify('Останавливаю вычисления; записи и завершённые этапы сохраняются');await poll();});
button('skill-resume',async()=>{const row=(latest.skill_workshops || []).find(r=>['paused','failed','preparing'].includes(r.state) && r.selected_episodes);if(!row)throw new Error('Нет сохранённой подготовки');await api('/api/jobs/skills',{task_ids:[row.task_id],workshop_id:row.id});await poll();});
