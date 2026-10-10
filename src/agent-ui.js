"use strict";
function requestUUID() {
  const bytes=new Uint8Array(16);crypto.getRandomValues(bytes);
  bytes[6]=(bytes[6]&15)|64;bytes[8]=(bytes[8]&63)|128;
  const hex=Array.from(bytes,(value)=>value.toString(16).padStart(2,"0")).join("");
  return [hex.slice(0,8),hex.slice(8,12),hex.slice(12,16),hex.slice(16,20),hex.slice(20)].join("-");
}

const el = (id) => document.getElementById(id);
let token = (sessionStorage.getItem("explorer_token") || location.hash.slice(1)).trim();
let catalog = [], templates = [], plan = {steps: []}, current = null, showTask = false, draftRequestId = requestUUID(), imageUrl = null, cameraPending = false, voicePending = false;
let evidenceKey = "", evidenceUrls = [];
let displayedFrame = null, selection = null, target = {phase: "idle"}, targetPending = false, frameLoading = false, trackPending = false;
let trackRequestId = requestUUID(), targetRevision = 0;
let floorMode = false, floorPending = false, observationMode = false;
if (token) sessionStorage.setItem("explorer_token", token);
if (location.hash) history.replaceState(null, "", location.pathname);

function message(text) { el("feedback").textContent = text; }
function login(error = "") { el("loginError").textContent = error; if (!el("login").open) el("login").showModal(); }
async function request(path, body) {
  if (!token) { login(); throw Error("Введите код доступа"); }
  const response = await fetch(path, {method: body === undefined ? "GET" : "POST", cache: "no-store",
    headers: {Authorization: "Bearer " + token, ...(body === undefined ? {} : {"Content-Type": "application/json"})},
    body: body === undefined ? undefined : JSON.stringify(body)});
  if (response.status === 401) { token = ""; sessionStorage.removeItem("explorer_token"); login("Код доступа не принят"); throw Error("Требуется вход"); }
  if (!response.ok) { const error = await response.text(); throw Error(error.slice(0, 500)); }
  return response;
}
async function api(path, body) { return (await request("/api/skill-agent/" + path, body)).json(); }
async function renderEvidence(task) {
  const evidence = (task.result?.steps || []).map((step) => step.evidence).filter(Boolean);
  const endpoints = evidence.map((entry) => entry.image_endpoint).filter((path) => typeof path === "string" && path.startsWith("/api/skill-agent/tasks/" + task.id + "/images/"));
  const exports=evidence.flatMap((entry) => entry.exports || []);
  const key = task.id + JSON.stringify([endpoints,exports]);
  if (key === evidenceKey) return;
  evidenceKey = key; evidenceUrls.forEach((url) => URL.revokeObjectURL(url)); evidenceUrls = []; el("evidenceImages").replaceChildren();
  for (const report of exports) for (const kind of ["json","csv"]) {
    const endpoint=report.artifacts?.[kind];
    if (typeof endpoint==="string" && /^\/api\/scene-reports\/[a-f0-9]{32}\/report\.(json|csv)$/.test(endpoint)) {
      const button=document.createElement("button");button.textContent="Скачать отчёт "+kind.toUpperCase();
      button.addEventListener("click",async () => { try { const response=await request(endpoint),url=URL.createObjectURL(await response.blob());
        const link=document.createElement("a");link.href=url;link.download="explorer-report-"+report.id+"."+kind;link.click();setTimeout(() => URL.revokeObjectURL(url),1000);
      } catch (error) { message(error.message); } });el("evidenceImages").append(button);
    }
  }
  for (const endpoint of endpoints) {
    try { const response = await request(endpoint); const url = URL.createObjectURL(await response.blob()); evidenceUrls.push(url);
      const image = document.createElement("img"); image.src = url; image.alt = "Сохранённое наблюдение этой задачи"; image.style.width = "100%"; image.style.borderRadius = "8px"; el("evidenceImages").append(image);
    } catch (error) { message("Сохранённое изображение: " + error.message); }
  }
}
function renderPlan(value = plan, events = [], syncEditor = true) {
  el("plan").replaceChildren();
  value.steps.forEach((step, index) => {
    const item = document.createElement("li"); item.className = "step";
    const skill = catalog.find((entry) => entry.name === step.skill);
    item.textContent = (index + 1) + ". " + (skill?.description || step.skill);
    const args = document.createElement("code"); args.textContent = JSON.stringify(step.args || {}); item.append(args);
    const latest = events.filter((event) => event.details?.step === index).at(-1);
    if (latest) { const state = document.createElement("small"); state.textContent = latest.phase + (latest.details?.state ? " · " + latest.details.state : ""); item.append(state); }
    el("plan").append(item);
  });
  if (syncEditor) el("planJson").value = JSON.stringify(value, null, 2);
}
function renderFields() {
  const skill = catalog.find((entry) => entry.name === el("skill").value);
  if (!skill) return;
  el("description").textContent = skill.description;
  el("requirements").textContent = "Ресурсы: " + (skill.resources.join(", ") || "чтение") + ". Условия: " + (skill.requirements.join(", ") || "без движения") + ". Единицы: " + skill.units + "; система: " + skill.frame;
  el("arguments").replaceChildren();
  Object.entries(skill.inputSchema.properties).forEach(([name, schema]) => {
    const label = document.createElement("label"); label.className = "field";
    label.textContent = name + (skill.inputSchema.required.includes(name) ? " *" : "");
    const input = schema.enum ? document.createElement("select") : document.createElement("input");
    input.dataset.argument = name;
    if (schema.enum) schema.enum.forEach((value) => { const option = document.createElement("option"); option.value = value; option.textContent = value; input.append(option); });
    else { input.type = schema.type === "boolean" ? "checkbox" : schema.type === "integer" || schema.type === "number" ? "number" : "text";
      if (schema.minimum !== undefined) input.min = schema.minimum;
      if (schema.maximum !== undefined) input.max = schema.maximum;
      if (schema.type === "array") input.placeholder = "JSON: [\"место 1\", \"место 2\"]";
      if (name === "goal_deg") input.placeholder = "[90,90,90,90,90,160]";
    }
    label.append(input); el("arguments").append(label);
  });
}
async function loadCatalog() {
  const data = await api("catalog"); catalog = data.skills; templates = data.templates;
  el("skill").replaceChildren(); catalog.forEach((skill) => { const option = document.createElement("option"); option.value = skill.name; option.textContent = skill.description; el("skill").append(option); });
  el("templates").replaceChildren();
  templates.forEach((template, index) => { const option = document.createElement("option"); option.value = index; option.textContent = template.name + " · v" + template.version; el("templates").append(option); });
  renderFields();
}
el("loginForm").addEventListener("submit", async (event) => {
  event.preventDefault(); token = el("code").value.trim(); sessionStorage.setItem("explorer_token", token);
  el("login").close(); el("code").value = "";
  try { await loadCatalog(); message("Подключено"); } catch (error) { message(error.message); }
});
el("skill").addEventListener("change", renderFields);
el("chooseSkill").addEventListener("click", () => {
  try {
    const skill = catalog.find((entry) => entry.name === el("skill").value), args = {};
    el("arguments").querySelectorAll("[data-argument]").forEach((input) => {
      const name = input.dataset.argument, schema = skill.inputSchema.properties[name];
      if (schema.type === "boolean") args[name] = input.checked;
      else if (input.value !== "") args[name] = schema.type === "array" || schema.type === "object" ? JSON.parse(input.value) : (schema.type === "number" || schema.type === "integer") ? Number(input.value) : input.value;
    });
    if (plan.steps.length >= 12) throw Error("Максимум 12 шагов");
    plan.steps.push({skill: skill.name, args}); showTask = false; draftRequestId = requestUUID(); renderPlan(); message("Навык добавлен. Аргументы ещё проверит робот при запуске.");
  } catch (error) { message(error.message); }
});
el("prepare").addEventListener("click", async () => {
  el("prepare").disabled = true;
  try { const result = await api("plan", {text: el("prompt").value});
    if (result.plan) { plan = result.plan; showTask = false; draftRequestId = requestUUID(); renderPlan(); message(result.explanation); }
    else message(result.clarification || (result.stop_requested ? "STOP запрошен" : "План не сформирован"));
  } catch (error) { message(error.message); } finally { el("prepare").disabled = false; }
});
el("run").addEventListener("click", async () => {
  el("run").disabled = true;
  try { const task = await api("tasks", {request_id: draftRequestId, plan, observing: el("observing").checked}); current = task.id; showTask = true; message("Принято. Следите за результатом задачи."); }
  catch (error) { message(error.message); } finally { el("run").disabled = false; }
});
el("cancel").addEventListener("click", async () => { try { if (current) await api("tasks/" + current + "/cancel", {}); message("Отмена запрошена"); } catch (error) { message(error.message); } });
el("stop").addEventListener("click", async () => { try { await request("/api/control", {op: "stop"}); message("STOP запрошен"); } catch (error) { message(error.message); } });
el("clearPlan").addEventListener("click", () => { plan = {steps: []}; showTask = false; draftRequestId = requestUUID(); renderPlan(); });
el("loadTemplate").addEventListener("click", () => { const template = templates[Number(el("templates").value)]; if (template) { plan = structuredClone(template.plan); showTask = false; draftRequestId = requestUUID(); el("templateName").value = template.name; renderPlan(); } });
el("saveTemplate").addEventListener("click", async () => { try { const value = JSON.parse(el("planJson").value); await api("templates", {name: el("templateName").value, plan: value}); plan = value; showTask = false; draftRequestId = requestUUID(); await loadCatalog(); renderPlan(); message("Сохранена новая версия; старые задания не изменены."); } catch (error) { message(error.message); } });
el("fullscreen").addEventListener("click", () => el("cameraViewport").requestFullscreen?.());
el("listen").addEventListener("click", async () => {
  if (voicePending) return; voicePending = true;
  try { await request("/api/voice", {operation: "listen"}); message("Говорите в микрофон робота…");
    for (let index = 0; index < 100; index++) { await new Promise((resolve) => setTimeout(resolve, 1000)); const result = await (await request("/api/voice")).json();
      if (!result.busy) { if (result.error) throw Error(result.error); el("prompt").value = result.transcript || ""; message("Речь записана. Проверьте текст и нажмите «Показать план»."); break; }
    }
  } catch (error) { message(error.message); } finally { voicePending = false; }
});
el("voiceStop").addEventListener("click", async () => { try { await request("/api/voice/stop", {}); } catch (error) { message(error.message); } });
el("openapiLink").addEventListener("click", async (event) => { event.preventDefault(); try { const result = await api("openapi"); const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type: "application/json"})); const link = document.createElement("a"); link.href = url; link.download = "explorer-skill-agent-openapi.json"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); } catch (error) { message(error.message); } });

async function refresh() {
  try { if (token) { const status = await api("status"); el("history").replaceChildren();
    status.tasks.forEach((task) => { const button = document.createElement("button"); button.textContent = new Date(task.created * 1000).toLocaleTimeString() + " · " + task.state + " · " + task.plan.steps.map((step) => step.skill).join(" → "); button.addEventListener("click", () => { current = task.id; showTask = true; }); el("history").append(button); });
    if (current) { const task = await api("tasks/" + current); el("taskState").textContent = task.state; if (showTask) renderPlan(task.plan, task.events, false); el("result").textContent = JSON.stringify(task.result, null, 2); el("events").textContent = JSON.stringify(task.events, null, 2); await renderEvidence(task); }
  } } catch (error) { message(error.message); } finally { setTimeout(refresh, 1000); }
}
function imageGeometry(frame = displayedFrame) {
  if (!frame) return null;
  const rect = el("camera").getBoundingClientRect(), outer = el("cameraViewport").getBoundingClientRect();
  const scale = Math.min(rect.width / frame.width, rect.height / frame.height);
  return {left: rect.left + (rect.width - frame.width * scale) / 2, top: rect.top + (rect.height - frame.height * scale) / 2,
    scale, outer};
}
function imagePoint(event, frame, clamp = false) {
  const geometry = imageGeometry(frame);
  if (!geometry || !geometry.scale) return null;
  const x = (event.clientX - geometry.left) / geometry.scale, y = (event.clientY - geometry.top) / geometry.scale;
  if (!clamp && (x < 0 || y < 0 || x > frame.width || y > frame.height)) return null;
  return [Math.max(0, Math.min(frame.width, x)), Math.max(0, Math.min(frame.height, y))];
}
function drawRoi(box, frame = displayedFrame, lost = false) {
  const geometry = imageGeometry(frame), overlay = el("roiBox");
  if (!box || !geometry) { overlay.style.display = "none"; return; }
  overlay.style.display = "block";
  overlay.classList.toggle("lost", lost);
  overlay.style.left = geometry.left - geometry.outer.left + box[0] * geometry.scale + "px";
  overlay.style.top = geometry.top - geometry.outer.top + box[1] * geometry.scale + "px";
  overlay.style.width = (box[2] - box[0]) * geometry.scale + "px";
  overlay.style.height = (box[3] - box[1]) * geometry.scale + "px";
}
function selectionBox(start, end) {
  return [Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[0], end[0]), Math.max(start[1], end[1])];
}
function renderTarget() {
  const live = target.target_id && (target.phase === "selected" || target.phase === "tracking");
  el("trackTarget").disabled = !live || trackPending;
  el("targetState").textContent = target.phase === "lost" ? "Предмет потерян. Выделите его заново. " + (target.reason || "")
    : live ? "Предмет выбран · " + target.phase + (target.association_fraction !== undefined ? " · совпадение " + Math.round(target.association_fraction * 100) + "%" : "")
    : "Предмет не выбран";
  if (!selection && !floorMode && !observationMode) drawRoi(target.bbox, displayedFrame, target.phase === "lost");
}
el("cameraViewport").addEventListener("pointerdown", (event) => {
  if (floorMode || observationMode) return;
  if (event.button !== 0 || selection || frameLoading || !displayedFrame) return;
  if (performance.now() - displayedFrame.receivedAt >= 450) { el("targetState").textContent = "Дождитесь свежего кадра и выделите предмет заново"; return; }
  const start = imagePoint(event, displayedFrame);
  if (!start) return;
  event.preventDefault();
  selection = {frame: {...displayedFrame}, start, end: start, pointer: event.pointerId};
  targetRevision++;
  el("cameraViewport").setPointerCapture(event.pointerId);
  drawRoi(selectionBox(start, start), selection.frame);
  el("targetState").textContent = "Кадр зафиксирован — быстро обведите предмет";
});
el("cameraViewport").addEventListener("pointermove", (event) => {
  if (selection && selection.pointer === event.pointerId && !selection.sending) {
    selection.end = imagePoint(event, selection.frame, true);
    drawRoi(selectionBox(selection.start, selection.end), selection.frame);
  }
});
el("cameraViewport").addEventListener("pointerup", async (event) => {
  if (floorMode && !floorPending && !frameLoading && event.button === 0 && displayedFrame) {
    const point = imagePoint(event, displayedFrame);
    if (!point) return;
    if (!el("observing").checked) { message("Подтвердите наблюдение за роботом справа перед поездкой."); return; }
    if (performance.now() - displayedFrame.receivedAt > 2500) { message("Кадр пола устарел. Нажмите «Поехать к точке пола» ещё раз."); return; }
    floorPending = true;
    const floorPlan = {steps: [{skill: "navigate_selected_floor", args: {frame_id: displayedFrame.id, u: point[0], v: point[1]}, timeout_s: 180}]};
    try { const task = await api("tasks", {request_id: displayedFrame.requestId, plan: floorPlan, observing: true, ttl_s: 190}); current = task.id; showTask = true; floorMode = false; message("Проверка выбранного пола и маршрута принята. Следите за результатом справа."); }
    catch (error) { message(error.message); } finally { floorPending = false; }
    return;
  }
  if (!selection || selection.pointer !== event.pointerId || selection.sending) return;
  event.preventDefault();
  const frozen = selection;
  let selectionError = "";
  frozen.end = imagePoint(event, frozen.frame, true);
  const bbox = selectionBox(frozen.start, frozen.end);
  frozen.sending = true;
  if (el("cameraViewport").hasPointerCapture(event.pointerId)) el("cameraViewport").releasePointerCapture(event.pointerId);
  try {
    if (performance.now() - frozen.frame.receivedAt >= 450) throw Error("Кадр устарел — выделите предмет заново на свежем кадре");
    if (Math.min(bbox[2] - bbox[0], bbox[3] - bbox[1]) < 12) throw Error("Обведите область не меньше 12 пикселей, включая края предмета");
    target = await (await request("/api/vision/select", {frame_id: frozen.frame.id, bbox})).json();
    trackRequestId = requestUUID();
    message("Предмет выбран без движения. Для движения взгляда используйте отдельную кнопку.");
  } catch (error) {
    selectionError = /expired|stale|fresh frame/i.test(error.message) ? "Кадр устарел — выделите предмет заново" : error.message;
    message(selectionError);
  } finally { selection = null; renderTarget(); if (selectionError) el("targetState").textContent = selectionError; }
});
el("cameraViewport").addEventListener("pointercancel", (event) => {
  if (selection?.pointer === event.pointerId && !selection.sending) { selection = null; renderTarget(); }
});
window.addEventListener("resize", renderTarget);
document.addEventListener("fullscreenchange", renderTarget);
el("trackTarget").addEventListener("click", async () => {
  if (trackPending || !target.target_id || target.phase === "lost") return;
  if (!el("observing").checked) { message("Подтвердите, что наблюдаете за роботом, в блоке выполнения справа."); return; }
  trackPending = true; renderTarget();
  const trackPlan = {steps: [{skill: "track_target", args: {target_id: target.target_id, duration_s: 10}, timeout_s: 15}]};
  try {
    const task = await api("tasks", {request_id: trackRequestId, plan: trackPlan, observing: el("observing").checked});
    current = task.id; showTask = true; message("Сопровождение принято исполнителем. Результат и условия — справа.");
  } catch (error) { message(error.message); } finally { trackPending = false; renderTarget(); }
});
el("cancelTarget").addEventListener("click", async () => {
  try { targetRevision++; target = await (await request("/api/vision/cancel", {})).json(); renderTarget(); message("Выбор снят, сопровождение взгляда остановлено"); }
  catch (error) { message(error.message); }
});
let cloudSample = null, cloudYaw = 0, cloudPitch = 0, cloudDrag = null;
function drawCloud() {
  const canvas = el("cloudCanvas"), ctx = canvas.getContext("2d");
  ctx.fillStyle = "#020617"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!cloudSample) return;
  const xyz = cloudSample.xyz, rgb = cloudSample.rgb;
  const valid = xyz.map((p, i) => ({p, i})).filter(({p}) => p.length === 3 && p.every(Number.isFinite));
  if (!valid.length) { ctx.fillStyle = "#cbd5e1"; ctx.fillText("Нет действительных точек глубины", 20, 30); return; }
  const center = [0, 1, 2].map(axis => valid.reduce((sum, {p}) => sum + p[axis], 0) / valid.length);
  const radius = Math.max(.05, ...valid.map(({p}) => Math.hypot(...p.map((v, axis) => v - center[axis]))));
  const scale = Math.min(canvas.width, canvas.height) * .43 / radius;
  const cy = Math.cos(cloudYaw), sy = Math.sin(cloudYaw), cp = Math.cos(cloudPitch), sp = Math.sin(cloudPitch);
  const points = valid.map(({p, i}) => {
    const [x, y, z] = p.map((v, axis) => v - center[axis]);
    const rx = cy * x + sy * z, rz = -sy * x + cy * z;
    return {x: canvas.width / 2 + rx * scale, y: canvas.height / 2 + (cp * y - sp * rz) * scale,
      z: sp * y + cp * rz, color: rgb[i]};
  }).sort((a, b) => b.z - a.z);
  for (const point of points) {
    const color = point.color;
    ctx.fillStyle = Array.isArray(color) && color.length === 3 && color.every(Number.isFinite)
      ? "rgb(" + color.map(v => Math.max(0, Math.min(255, Math.round(v)))).join(",") + ")" : "#a5b4fc";
    ctx.beginPath(); ctx.arc(point.x, point.y, 3, 0, Math.PI * 2); ctx.fill();
  }
  ctx.fillStyle = "#cbd5e1"; ctx.font = "14px system-ui";
  ctx.fillText(valid.length + " точек · размер области " + (2 * radius).toFixed(2) + " м", 16, 24);
}
el("cloudCanvas").addEventListener("pointerdown", event => {
  cloudDrag = {x: event.clientX, y: event.clientY}; el("cloudCanvas").setPointerCapture(event.pointerId);
});
el("cloudCanvas").addEventListener("pointermove", event => {
  if (!cloudDrag) return;
  cloudYaw += (event.clientX - cloudDrag.x) * .01;
  cloudPitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, cloudPitch + (event.clientY - cloudDrag.y) * .01));
  cloudDrag = {x: event.clientX, y: event.clientY}; drawCloud();
});
for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) el("cloudCanvas").addEventListener(name, () => { cloudDrag = null; });
for (const [button, endpoint] of [["visionColors", "colors"], ["visionTags", "tags"], ["visionDepth", "cloud?max_points=500"]]) {
  el(button).addEventListener("click", async () => {
    el(button).disabled = true;
    el("cloudView").hidden = true;
    try {
      const result = await (await request("/api/vision/" + endpoint)).json();
      if (button === "visionDepth" && result.units === "metres" && Array.isArray(result.xyz) && Array.isArray(result.rgb)) {
        cloudSample = result; cloudYaw = 0; cloudPitch = 0; el("cloudView").hidden = false; drawCloud();
        el("cloudCaption").textContent = "Снимок " + new Date(result.image_stamp * 1000).toLocaleTimeString()
          + " · " + result.frame + " · координаты камеры в метрах. Потяните изображение для вращения. Это снимок поверхностей, не глобальная карта.";
      }
      el("visionResult").textContent = JSON.stringify(result, null, 2); el("visionDetails").open = button !== "visionDepth";
    }
    catch (error) { el("visionResult").textContent = error.message; el("visionDetails").open = true; }
    finally { el(button).disabled = false; }
  });
}
async function refreshTarget() {
  if (token && !targetPending && !document.hidden && !selection && !floorMode && !observationMode) {
    targetPending = true;
    const revision = targetRevision;
    try { const result = await (await request("/api/vision/target")).json(); if (!selection && revision === targetRevision) { target = result; renderTarget(); } }
    catch (error) { if (!selection && revision === targetRevision) { target = {...target, phase: "unavailable"}; el("targetState").textContent = "Наблюдение недоступно: " + error.message; el("trackTarget").disabled = true; } }
    finally { targetPending = false; }
  }
  setTimeout(refreshTarget, 250);
}
async function camera() {
  const started = performance.now();
  if (token && !cameraPending && !document.hidden && !selection && !floorMode && !observationMode) { cameraPending = true;
    let next = null;
    try {
      const response = await request("/api/vision/frame"), identifier = response.headers.get("X-Target-Frame-ID"), receivedAt = performance.now();
      if (!/^[a-f0-9]{32}$/i.test(identifier || "")) throw Error("Камера не вернула идентификатор кадра");
      next = URL.createObjectURL(await response.blob());
      const preview = new Image();
      await new Promise((resolve, reject) => { preview.onload = resolve; preview.onerror = () => reject(Error("Не удалось показать JPEG")); preview.src = next; });
      if (!selection && !floorMode && !observationMode) {
        const previous = imageUrl, loadedUrl = next;
        frameLoading = true;
        await new Promise((resolve, reject) => {
          el("camera").onload = () => { displayedFrame = {id: identifier, width: preview.naturalWidth, height: preview.naturalHeight, receivedAt}; imageUrl = loadedUrl; if (previous) URL.revokeObjectURL(previous); renderTarget(); resolve(); };
          el("camera").onerror = () => reject(Error("Не удалось обновить изображение"));
          el("camera").src = loadedUrl;
        });
        next = null;
        el("cameraState").textContent = "Свежий кадр · " + new Date().toLocaleTimeString();
      }
    } catch (error) { el("cameraState").textContent = error.message; }
    finally { if (next) URL.revokeObjectURL(next); frameLoading = false; cameraPending = false; }
  }
  setTimeout(camera, Math.max(10, 180 - (performance.now() - started)));
}
el("floorFrame").addEventListener("click", async () => {
  if (floorPending || frameLoading) return;
  floorMode = true; observationMode = false;
  while (cameraPending) await new Promise((resolve) => setTimeout(resolve, 20));
  frameLoading = true;
  try {
    const response = await request("/api/navigation/target-frame");
    const identifier = response.headers.get("X-Target-Frame-ID"), receivedAt = performance.now();
    if (!/^[a-f0-9]{32}$/i.test(identifier || "")) throw Error("Кадр не содержит идентификатор");
    const url = URL.createObjectURL(await response.blob()), preview = new Image();
    await new Promise((resolve,reject) => { preview.onload = resolve; preview.onerror = reject; preview.src = url; });
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageUrl = url; el("camera").src = url;
    displayedFrame = {id: identifier, width: preview.naturalWidth, height: preview.naturalHeight, receivedAt, requestId: requestUUID()};
    drawRoi(null); el("cameraState").textContent = "Кадр пола зафиксирован. Выберите точку в течение 2,5 с.";
    message("Нажмите на свободный ровный пол. Это запустит ограниченную поездку после проверки.");
  } catch (error) { floorMode = false; message(error.message); }
  finally { frameLoading = false; }
});
el("liveCamera").addEventListener("click", () => { floorMode = false; observationMode = false; clearHumanOverlay(); message("Живой обзор: выделение предмета само не двигает робот."); });
function clearHumanOverlay() {
  const canvas=el("humanOverlay"); canvas.getContext("2d").clearRect(0,0,canvas.width,canvas.height);
}
function showVisionResult(result) { el("visionResult").textContent=JSON.stringify(result,null,2); el("visionDetails").open=true; }
async function showObservation(result) {
  observationMode=true; floorMode=false;
  while (cameraPending) await new Promise((resolve) => setTimeout(resolve,20));
  const response=await request(result.image_endpoint), next=URL.createObjectURL(await response.blob());
  const image=new Image();
  try { await new Promise((resolve,reject) => { image.onload=resolve;image.onerror=reject;image.src=next; }); }
  catch (error) { URL.revokeObjectURL(next);throw error; }
  if (imageUrl) URL.revokeObjectURL(imageUrl); imageUrl=next; el("camera").src=next;
  displayedFrame={id:result.frame_id,width:image.naturalWidth,height:image.naturalHeight,receivedAt:performance.now()};
  drawRoi(null); const canvas=el("humanOverlay"), geometry=imageGeometry();
  canvas.width=el("cameraViewport").clientWidth;canvas.height=el("cameraViewport").clientHeight;
  const ctx=canvas.getContext("2d");ctx.fillStyle="#fbbf24";
  for (const rows of Object.values(result.mode_results || {})) for (const row of rows) {
    for (const point of row.landmarks || []) { if (point.length>=2 && point.slice(0,2).every(Number.isFinite)) {
      ctx.beginPath();ctx.arc(geometry.left-geometry.outer.left+point[0]*geometry.scale,geometry.top-geometry.outer.top+point[1]*geometry.scale,3,0,Math.PI*2);ctx.fill();
    } }
  }
  el("cameraState").textContent="Кадр наблюдения зафиксирован. «Живой обзор» возобновляет видео.";
}
for (const [button,mode] of [["humanHand","hand"],["humanPose","pose"],["humanFace","face"]]) {
  el(button).addEventListener("click",async () => {
    el(button).disabled=true;
    try { const result=await (await request("/api/vision/humans",{modes:[mode]})).json();showVisionResult(result);await showObservation(result); }
    catch (error) { message(error.message); } finally { el(button).disabled=false; }
  });
}
el("zoneBegin").addEventListener("click",async () => {
  try { if (!target.bbox || target.phase==="lost") throw Error("Сначала выделите неподвижную область на живом кадре.");
    showVisionResult(await (await request("/api/vision/zone/begin",{roi:target.bbox.map(Math.round)})).json());
  } catch (error) { message(error.message); }
});
for (const [button,operation] of [["zoneCompare","compare"],["zoneCancel","cancel"]]) el(button).addEventListener("click",async () => {
  try { showVisionResult(await (await request("/api/vision/zone/"+operation,{})).json()); } catch (error) { message(error.message); }
});
el("savePose").addEventListener("click",async () => {
  try { showVisionResult(await (await request("/api/arm/library/pose",{name:el("poseName").value})).json());message("Командная поза сохранена. Движения не выполнялись."); } catch (error) { message(error.message); }
});
el("saveScene").addEventListener("click",async () => {
  try { showVisionResult(await (await request("/api/arm/library/scene",{name:el("sceneName").value,obstacles:JSON.parse(el("sceneBoxes").value)})).json()); } catch (error) { message(error.message); }
});
el("warmPlanner").addEventListener("click",async () => {
  try { showVisionResult(await (await request("/api/arm/planner/warmup",{})).json());message("Подготовка планировщика запрошена без движения."); } catch (error) { message(error.message); }
});
if (!token) login(); else loadCatalog().catch((error) => message(error.message));
renderPlan(); refresh(); refreshTarget(); camera();
