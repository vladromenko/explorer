// Browser contract test: every POST is intercepted; no real jobs or controls.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('playwright');
const base=process.env.EXPLORER_UI_URL || 'http://127.0.0.1:8766';
const repo=path.resolve(__dirname,'../..');
(async()=>{
 const original=await (await fetch(base+'/api/state?compact=true')).json();
 // This contract exercises archived controller-search controls, without a
 // concurrent neural cycle taking ownership of the primary panel.
 original.learning_status=null;
 const goal={id:'a'.repeat(12),created:1,state:'planned',learning_mode:'skill_search',
  plan:{title:'Носок в корзину',request:'Научись убирать носки в корзину',fully_supported:true,
   task_ids:['sock_basket'],tasks:[{task_id:'sock_basket',title:'Носок → корзина',learns:'Параметры действий',steps:[]}],
   missing:[],clarification:'',language_model:{model:'browser-test'}},results:[]};
 const learner={id:'b'.repeat(12),state:'budget_paused',completed_attempts:16,options:{attempts:100},active_workers:[],phase_group:'place'};
 const calls=[],errors=[],jobs=[];
 const browser=await chromium.launch({headless:true,executablePath:process.env.EXPLORER_CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 try{
  const page=await browser.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.route(base+'/**',async route=>{
   const req=route.request(),url=new URL(req.url());
   const json=value=>route.fulfill({contentType:'application/json',body:JSON.stringify(value)});
   if(req.method()==='POST'){
    calls.push({path:url.pathname,body:req.postDataJSON()});return json({id:'c'.repeat(12),status:'running',ok:true});
   }
   if(url.pathname==='/')return route.fulfill({contentType:'text/html',body:fs.readFileSync(path.join(repo,'simulator/static/index.html'))});
   if(url.pathname==='/static/app.js')return route.fulfill({contentType:'text/javascript',body:fs.readFileSync(path.join(repo,'simulator/static/app.js'))});
   if(url.pathname==='/api/state')return json({...original,jobs,learning_goals:[goal],skill_trainings:[learner]});
   if(url.pathname==='/api/night-readiness')return json({ready:true,device:'test',free_gib:12});
   if(url.pathname==='/static/learning-live/state.json')return json({updated_at:Date.now()/1000,phase:'idle',views:[]});
   return route.continue();
  });
  await page.goto(base+'/#skill-workshop');
  await page.locator('#skill-history').evaluate(node=>{node.open=true;});
  await page.waitForFunction(()=>document.getElementById('request-result').textContent.includes('Носок в корзину'));
  async function click(id,endpoint){
   const before=calls.length;await page.locator('#'+id).click();
   await page.waitForFunction(()=>true);
   for(let i=0;i<50;i++){const row=calls.slice(before).find(r=>r.path===endpoint);if(row)return row;await page.waitForTimeout(20);}
   throw Error('Missing intercepted request '+id);
  }
  await page.locator('#learning-request').fill('Научись убирать носки в корзину');
  let req=await click('request-plan','/api/jobs/assistant');assert.equal(req.body.auto_train,false);assert.equal(req.body.learning_mode,'skill_search');
  req=await click('request-learn','/api/jobs/assistant');assert.equal(req.body.auto_train,true);assert.equal(req.body.workers,4);
  for(const state of ['operator_paused','memory_paused','budget_paused','interrupted']){
   Object.assign(goal,{state,results:[{task_id:'sock_basket',skill_training_id:learner.id,simulation_ready:false}]});
   await page.evaluate(g=>{goalSignature='';renderGoals([g]);},goal);
   assert.equal(await page.locator('#request-resume').isVisible(),true,state);
   assert.equal((await page.locator('#request-result').innerText()).includes('undefined'),false);
   req=await click('request-resume','/api/jobs/assistant');assert.equal(req.body.goal_id,goal.id);assert.equal(req.body.learning_mode,'skill_search');
  }
  req=await click('skill-learning-resume','/api/jobs/skill_train');assert.equal(req.body.skill_training_id,learner.id);assert.equal(req.body.count,116);
  req=await click('skill-learn','/api/jobs/skill_train');assert.equal(req.body.task_id,'sock_basket');assert.equal(req.body.count,100);
  await page.evaluate(()=>{document.querySelectorAll('.night-tasks input').forEach(x=>{x.checked=x.value==='sock_basket';});showTaskSelection();});
  req=await click('night-start','/api/jobs/skill_train');assert.deepEqual(req.body.task_ids,['sock_basket']);
  jobs.push({id:'c'.repeat(12),kind:'skill_train',status:'running',message:'Test running',progress:0});
  await page.evaluate(snapshot=>renderSkills(snapshot),{...original,jobs,skill_trainings:[learner]});
  await click('skill-pause','/api/cancel');
  assert.equal(await page.locator('#worker-grid').isVisible(),true);
  assert.equal(await page.evaluate(()=>{const ids=[...document.querySelectorAll('[id]')].map(x=>x.id);return ids.length===new Set(ids).size;}),true);
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,checks:'plan, language launch, four pause resumes, same learner resume, direct launch, selected-task launch, pause, visible worker grid, unique IDs, no JS errors',interceptedPosts:calls.length,realPosts:0}));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
