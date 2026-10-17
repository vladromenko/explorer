// UI regression: a paused historical search must not mask live neural training.
// All POST requests are intercepted: the live learner is never controlled.
const assert=require('node:assert/strict');
const {chromium}=require('playwright');
const base=process.env.EXPLORER_UI_URL || 'http://127.0.0.1:8766';
(async()=>{
 const snapshot=await (await fetch(base+'/api/state?compact=true')).json();
 const info={...snapshot.learning_status,state:'running',phase:'train',steps_completed:8500,
  training_budget:{effective_steps:119047},eta_s:78000,cycle_id:'a'.repeat(12),
  title:'Носок → корзина',iteration:1,message:'Обучение ACT',demonstrations:{attempts:1125,successful:701}};
 snapshot.learning_status=info;
 snapshot.jobs=[{id:'b'.repeat(12),kind:'train',status:'running',message:'Обучение ACT',progress:6}];
 snapshot.skill_trainings=[{id:'c'.repeat(12),state:'operator_paused',completed_attempts:88,
  options:{attempts:1000},active_workers:[],phase_group:'place',reason:'Работники остановлены; прогресс сохранён'}];
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 try{
  const page=await browser.newPage({viewport:{width:1200,height:900}}),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.route(base+'/**',route=>{
   const request=route.request(),url=new URL(request.url());
   if(request.method()==='POST')return route.fulfill({contentType:'application/json',body:'{}'});
   if(url.pathname==='/api/state')return route.fulfill({contentType:'application/json',body:JSON.stringify(snapshot)});
   return route.continue();
  });
  await page.goto(base+'/#learning-dashboard');
  await page.waitForFunction(()=>document.getElementById('cycle-steps').textContent.includes('119'));
  assert.equal(await page.locator('#learning-dashboard').isVisible(),true);
  assert.equal(await page.locator('main > :first-child').getAttribute('id'),'learning-dashboard');
  assert.equal(await page.locator('#cycle-state').innerText(),'Обучение продолжается');
  assert.match(await page.locator('#cycle-steps').innerText(),/8\s*500/);
  assert.equal(await page.locator('#skill-history').getAttribute('open'),null);
  assert.equal(await page.locator('#skill-learning-summary').isVisible(),false);
  assert.equal(await page.locator('#cycle-pause').isVisible(),true);
  assert.equal(await page.locator('#cycle-resume').isVisible(),false);
  info.steps_completed=8501;
  await page.evaluate(value=>renderStatus(value),snapshot);
  assert.match(await page.locator('#cycle-steps').innerText(),/8\s*501/);
  await page.locator('#skill-history > summary').click();
  assert.match(await page.locator('#skill-learning-summary').innerText(),/архив прежнего/);
  assert.equal(await page.locator('#skill-learning-resume').isVisible(),false);
  await page.locator('#skill-history > summary').click();
  await page.screenshot({path:'/tmp/explorer-neural-progress.png'});
  info.state='paused';snapshot.jobs=[];
  await page.evaluate(value=>renderStatus(value),snapshot);
  assert.equal(await page.locator('#cycle-state').innerText(),'Обучение на паузе');
  assert.equal(await page.locator('#cycle-resume').isVisible(),true);
  assert.equal(await page.locator('#cycle-pause').isVisible(),false);
  assert.deepEqual(errors,[]);
  // Read the actual running service too; no fixture or reload of the learner.
  const live=await browser.newPage({viewport:{width:1200,height:900}});
  await live.route(base+'/**',route=>route.request().method()==='POST'
   ? route.fulfill({contentType:'application/json',body:'{}'}) : route.continue());
  await live.goto(base+'/#learning-dashboard');
  await live.waitForFunction(()=>document.getElementById('cycle-state').textContent==='Обучение продолжается');
  assert.equal(await live.locator('main > :first-child').getAttribute('id'),'learning-dashboard');
  assert.equal(await live.locator('#cycle-steps').isVisible(),true);
  assert.equal(await live.locator('#skill-learning-summary').isVisible(),false);
  console.log(JSON.stringify({liveState:await live.locator('#cycle-state').innerText(),liveSteps:await live.locator('#cycle-steps').innerText()}));
  console.log(JSON.stringify({passed:true,checks:'live ACT visible first; historical pause collapsed; exact steps update; real cycle pause distinguishable; no JS errors; real POSTs blocked'}));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
