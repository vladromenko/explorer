// End-to-end UI contracts; all mutations are intercepted, including disconnect.
const assert=require('node:assert/strict');
const {chromium}=require('playwright');
const base=process.env.EXPLORER_AUTOMATIC_URL||'http://127.0.0.1:8767';
(async()=>{
 let snapshot=await(await fetch(base+'/api/state')).json();
 const calls=[],errors=[];
 const browser=await chromium.launch({headless:true,executablePath:process.env.EXPLORER_CHROME||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1150}});page.on('pageerror',error=>errors.push(error.message));
  await page.route(base+'/**',async route=>{
   const request=route.request(),url=new URL(request.url());
   if(request.method()==='POST'){calls.push({path:url.pathname,body:request.postDataJSON()});return route.fulfill({contentType:'application/json',body:JSON.stringify({id:'a'.repeat(12),state:'queued',message:'Сохранено',stopping:true})});}
   if(url.pathname==='/app.js'&&process.env.EXPLORER_AUTOMATIC_SCRIPT)return route.fulfill({contentType:'application/javascript',body:require('node:fs').readFileSync(process.env.EXPLORER_AUTOMATIC_SCRIPT,'utf8')});
   if(url.pathname==='/api/state')return route.fulfill({contentType:'application/json',body:JSON.stringify(snapshot)});
   return route.continue();
  });
  await page.goto(base+'/');await page.waitForFunction(()=>document.getElementById('models').textContent.includes('установлены'));
  assert.equal(await page.locator('textarea').count(),1);
  assert.equal(await page.locator('#cycle-start,#record,#request-steps,#reset,#task-select').count(),0);
  assert.equal(await page.locator('#learn').innerText(),'Обучить');
  const initialCount=calls.length;
  await page.locator('#request').fill('Научись убирать носок в корзину высотой 30 см и диаметром 25 см');
  await page.locator('#learn').click();
  await page.waitForFunction(()=>document.getElementById('form-status').textContent.includes('сохранена'));
  let request=calls.slice(initialCount).find(r=>r.path==='/api/learn');assert.equal(request.body.attempts,1000);assert.equal(request.body.hours,24);assert.equal(request.body.parent_id,null);
  const id='b'.repeat(12);
  snapshot={...snapshot,requests:[{id,text:'Печенье в руку',state:'needs_clarification',reason:'Передача в руку пока не поддерживается',plan:{title:'Печенье в руку',missing:['Нет проверенной передачи в ладонь']},tasks:[]}]};
  await page.evaluate(s=>showRequests(s),snapshot);
  await page.getByRole('button',{name:'Уточнить задачу'}).click();
  await page.locator('#request').fill('Вместо руки положи на устойчивый поднос');await page.locator('#learn').click();
  for(let i=0;i<30&&calls.filter(r=>r.path==='/api/learn').length<2;i++)await page.waitForTimeout(20);
  assert.equal(calls.filter(r=>r.path==='/api/learn').at(-1).body.parent_id,id);
  const legacy={id:'c'.repeat(12),title:'Носок → корзина',state:'operator_paused',completed:16,total:96,active_workers:[],reason:'Пауза'};
  snapshot={...snapshot,legacy,requests:[],trainings:[legacy]};
  await page.evaluate(s=>showCurrent(s),snapshot);
  assert.equal(await page.locator('#current-control').innerText(),'Продолжить');await page.locator('#current-control').click();
  assert(calls.some(r=>r.path==='/api/legacy/'+legacy.id+'/resume'));
  const own={id,text:'Убрать носок',state:'training',reason:'Обучаю',plan:{title:'Носок → корзина'},tasks:[{task_id:'sock_basket',training_id:'d'.repeat(12),state:'running',recipe:{title:'Носок → корзина'},progress:{active_workers:[{},{}]}}]};
  const training={id:'d'.repeat(12),state:'running',completed:8,total:96,phase:'pick',active_workers:[{slot:0,episode_id:'e'.repeat(12)},{slot:1,episode_id:'f'.repeat(12)}]};
  snapshot={...snapshot,legacy:null,requests:[own],trainings:[training]};await page.evaluate(s=>showCurrent(s),snapshot);
  assert.equal(await page.locator('.worker').count(),2);assert((await page.locator('#workers').innerText()).includes('2'));
  const archived={...own,id:'0'.repeat(12),state:'maintenance_paused',superseded_by:own.id};
  snapshot={...snapshot,requests:[own,archived]};await page.evaluate(s=>showRequests(s),snapshot);
  assert.equal(await page.locator('#requests > article').count(),1);
  assert.equal(await page.locator('#request-history').evaluate(e=>e.open),false);
  assert.equal(await page.locator('#request-history article').count(),1);
  await page.locator('#request-history summary').click();
  await page.waitForFunction(()=>historyOpen===true);
  snapshot={...snapshot,requests:[{...own,reason:'Новая проверка'},archived]};await page.evaluate(s=>showRequests(s),snapshot);
  assert.equal(await page.locator('#request-history').evaluate(e=>e.open),true);
  await page.locator('#current-control').click();assert(calls.some(r=>r.path==='/api/requests/'+id+'/pause'));
  await page.locator('#disconnect').click();assert(calls.some(r=>r.path==='/api/disconnect'));
  await page.waitForFunction(()=>document.getElementById('learn').disabled);
  assert.equal(await page.locator('#learn').isDisabled(),true);
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,realMutations:0,interceptedMutations:calls.length,checks:'single form, no expert controls, queued learning, clarification parent, same legacy resume, two real worker slots, pause, explicit disconnect, no JavaScript errors'}));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
