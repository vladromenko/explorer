"use strict";
let experimentCatalog = [];
let selectedExperimentRun = null;
const experimentCaptures = new Map();
const experimentBusy = new Set();
const experimentGroupNames = ["Все", "Камера", "Робот", "Память", "Карта", "Задачи", "Обучение", "Записанные данные"];

function experimentRequestId() {
    return "experiment-" + Array.from(crypto.getRandomValues(new Uint8Array(16)), value => value.toString(16).padStart(2, "0")).join("");
}
function experimentNode(tag, text = "", className = "") {
    const node = document.createElement(tag); node.textContent = text; node.className = className; return node;
}
function experimentMessage(text, failed = false) {
    $("experimentState").textContent = text;
    $("experimentState").className = "statusline " + (failed ? "bad" : "ok");
}
const experimentInputHelp = {
    events: "JSON-объект с events: список событий, поля stage, verified, source (operator/measured_sensor), outcome. Используйте записанную историю, максимум 100 событий.",
    motion: "JSON-объект с before и after: {x,y,yaw}, velocity: [vx,vy,wz], duration_s (до 10), observation_source. Метры, радианы, секунды; позы должны относиться к одной карте. Команда только анализируется.",
    contact: "JSON-объект с samples: список до 100 записей. Поля wrist_rgb, gripper_state, local_gravity, contact_label, label_source. Источник контактной метки должен быть calibrated_sensor или validated_tactile_dataset; непроверенные метки будут отклонены.",
    hold: "JSON-объект с before и after: минимум три наблюдения в каждом. Нужны at, object_id, confidence, object_xyz, tcp_xyz, base_xyyaw, floor_clearance_m, frame=base_footprint, depth_validated, identity_association_verified и проверенный источник положения камеры. Без достоверной геометрии результат — неизвестно."
};
async function loadExperiments() {
    try {
        const data = await (await api("experiments")).json();
        experimentCatalog = data.experiments || [];
        if (!$("experimentGroup").options.length) {
            for (const group of experimentGroupNames) {
                const option = experimentNode("option", group); option.value = group; $("experimentGroup").appendChild(option);
            }
        }
        renderExperimentCards(); await loadExperimentHistory();
    } catch (error) { experimentMessage(error.message, true); }
}
function renderExperimentCards() {
    const group = $("experimentGroup").value;
    experimentCaptures.clear();
    $("experimentList").replaceChildren();
    for (const item of experimentCatalog.filter(item => group === "Все" || item.group === group)) {
        const card = experimentNode("article", "", "experiment");
        card.append(experimentNode("small", item.group), experimentNode("h3", item.id + " · " + item.name),
            experimentNode("p", item.purpose), experimentNode("small", "Основа: " + item.inspiration + ". Реализация: инженерная проверка; отдельная исследовательская модель не заявлена."));
        if (["query", "task", "roi"].includes(item.input_kind)) {
            const input = experimentNode("input"); input.id = "experimentQuery" + item.id;
            input.placeholder = item.input_kind === "task" ? "Например: принеси носок в корзину" : "Название предмета (необязательно)";
            input.maxLength = 500; input.style.width = "100%"; input.style.marginTop = "12px";
            card.appendChild(input);
        }
        if (experimentInputHelp[item.input_kind]) {
            const details = experimentNode("details"); details.open = true;
            details.append(experimentNode("summary", "Загрузить записанные данные"), experimentNode("p", experimentInputHelp[item.input_kind], "muted"));
            const editor = experimentNode("textarea"); editor.id = "experimentParams" + item.id;
            editor.rows = 5; editor.placeholder = "Вставьте JSON с вашими реальными данными"; editor.style.width = "100%";
            const file = experimentNode("input"); file.type = "file"; file.accept = ".json,application/json";
            file.onchange = async () => {
                try {
                    const selected = file.files[0];
                    if (selected && selected.size <= 12000) editor.value = await selected.text();
                    else experimentMessage("Файл параметров должен быть не больше 12 КБ", true);
                } catch (error) { experimentMessage(error.message, true); }
            };
            details.append(editor, file); card.appendChild(details);
        }
        if (item.input_kind === "roi") {
            const captureButton = experimentNode("button", "Снять кадр и выделить предмет");
            captureButton.onclick = () => captureExperiment(item.id);
            const canvas = experimentNode("canvas", ""); canvas.id = "experimentCapture" + item.id;
            canvas.hidden = true; canvas.style.width = "100%"; canvas.style.touchAction = "none"; canvas.style.marginTop = "10px";
            const state = experimentNode("p", "Можно проверить распознанную цель по названию или выделить её на одном снимке.", "muted"); state.id = "experimentCaptureState" + item.id;
            card.append(captureButton, canvas, state);
        }
        const row = experimentNode("div", "", "row");
        const button = experimentNode("button", item.action_label, "primary"); button.id = "experimentRun" + item.id;
        button.onclick = () => runExperiment(item.id); row.appendChild(button);
        if (item.id === "E02") {
            const find = experimentNode("button", "Поиск по названию (GroundingDINO)"); find.onclick = () => searchExperimentObject(item.id); row.appendChild(find);
        }
        card.appendChild(row); $("experimentList").appendChild(card);
    }
}
async function captureExperiment(id) {
    try {
        const capture = await (await api("experiments/capture", {})).json();
        const blob = await (await api("experiments/capture/" + capture.id)).blob();
        const url = URL.createObjectURL(blob); const image = new Image();
        try { image.src = url; await image.decode(); } finally { URL.revokeObjectURL(url); }
        const canvas = $("experimentCapture" + id); canvas.hidden = false;
        canvas.width = capture.width; canvas.height = capture.height;
        const record = {capture, image, bbox: null}; experimentCaptures.set(id, record);
        const draw = () => {
            const ctx = canvas.getContext("2d"); ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
            if (record.bbox) {
                const [x1, y1, x2, y2] = record.bbox; ctx.strokeStyle = "#ffcf66"; ctx.lineWidth = 3; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            }
        };
        const position = event => {
            const rect = canvas.getBoundingClientRect();
            return [Math.max(0, Math.min(canvas.width, Math.round((event.clientX - rect.left) * canvas.width / rect.width))),
                    Math.max(0, Math.min(canvas.height, Math.round((event.clientY - rect.top) * canvas.height / rect.height)))];
        };
        let begin = null;
        canvas.onpointerdown = event => { begin = position(event); canvas.setPointerCapture(event.pointerId); event.preventDefault(); };
        canvas.onpointermove = event => {
            if (begin) {
                const end = position(event); record.bbox = [Math.min(begin[0], end[0]), Math.min(begin[1], end[1]), Math.max(begin[0], end[0]), Math.max(begin[1], end[1])]; draw();
            }
        };
        canvas.onpointerup = event => {
            if (begin) {
                canvas.onpointermove(event); begin = null;
                $("experimentCaptureState" + id).textContent = "Область: " + record.bbox.join(", ") + " пикс. Анализируется этот сохранённый снимок, без движения.";
            }
        };
        canvas.onpointercancel = () => { begin = null; record.bbox = null; draw(); };
        draw(); $("experimentCaptureState" + id).textContent = "Выделите предмет, проведя прямоугольник мышью или пальцем.";
    } catch (error) { experimentMessage("Снимок: " + error.message, true); }
}
function experimentParams(id) {
    const editor = $("experimentParams" + id);
    if (editor && !editor.value.trim()) throw Error("Вставьте записанные данные JSON или загрузите файл — формат указан в карточке");
    const params = editor ? JSON.parse(editor.value) : {};
    if (!params || Array.isArray(params) || typeof params !== "object") throw Error("Нужен JSON-объект параметров");
    const input = $("experimentQuery" + id);
    if (input) params.query = input.value.trim();
    const record = experimentCaptures.get(id);
    if (record) {
        if (!record.bbox || record.bbox[2] <= record.bbox[0] || record.bbox[3] <= record.bbox[1]) throw Error("Выделите прямоугольную область предмета");
        params.capture_id = record.capture.id; params.bbox = record.bbox;
    }
    return params;
}
async function runExperiment(id, replayId = null) {
    if (experimentBusy.has(id)) return;
    experimentBusy.add(id);
    const button = $("experimentRun" + id); if (button) button.disabled = true;
    try {
        const body = {experiment: id, mode: replayId ? "replay" : "observe",
            params: replayId ? {run_id: replayId} : experimentParams(id), request_id: experimentRequestId(), issued_at: Date.now() / 1000};
        experimentMessage(id + ": проверяю входные данные…");
        const check = await (await api("experiments/preflight", body)).json();
        if (!check.ready) throw Error((check.blocked_by || []).join("; "));
        const result = await (await api("experiments/run", body)).json();
        renderExperimentResult(result); await loadExperimentHistory();
        experimentMessage(id + (result.state === "completed" ? ": результат сохранён" : ": " + result.result.summary), result.state !== "completed");
    } catch (error) { experimentMessage(id + ": " + error.message, true); }
    finally { experimentBusy.delete(id); if (button) button.disabled = false; }
}
const experimentResultNames = {arm:"Углы и достоверность",objects:"Объекты",alternative:"Альтернативный поиск",evidence:"Основание результата",ranked:"Ракурсы",steps:"Порядок навыков",blocked_by:"Что требуется",next_step:"Следующий шаг",candidates:"Варианты захвата",prediction:"Прогноз",observed:"Наблюдение",confirmed_events:"Подтверждённые события",invalid:"Некорректные записи",episodes:"Сохранённые показы",learning:"Задания обучения",policy_preview:"Проверка модели",mobile_policy:"Исполнение мобильной модели",queue:"Следующие показы",mobile_demonstration_suggestions:"Что улучшить в мобильных показах",retrieval:"Найденные ракурсы",recent_baseline:"Последние ракурсы"};
function renderExperimentResult(run) {
    selectedExperimentRun = run;
    const container = $("experimentResult"); container.replaceChildren(); container.hidden = false;
    container.append(experimentNode("h2", run.experiment + " · " + (run.mode === "replay" ? "Повтор по сохранённым входам" : "Результат")),
        experimentNode("p", run.result.summary || run.state),
        experimentNode("small", new Date(run.at * 1000).toLocaleString() + " · " + run.elapsed_ms + " мс · " + run.state + " · команды моторам не отправлялись"));
    for (const [key, label] of Object.entries(experimentResultNames)) {
        if (run.result[key] !== undefined && run.result[key] !== null) {
            const details = experimentNode("details"); details.open = ["blocked_by", "steps", "objects", "episodes", "candidates"].includes(key);
            details.append(experimentNode("summary", label), experimentNode("pre", JSON.stringify(run.result[key], null, 2), "experiment-data"));
            container.appendChild(details);
        }
    }
    const details = experimentNode("details"); details.append(experimentNode("summary", "Все поля и сохранённые входные данные"), experimentNode("pre", JSON.stringify(run, null, 2), "experiment-data"));
    const row = experimentNode("div", "", "row");
    const replay = experimentNode("button", "Повторить анализ этого запуска"); replay.onclick = () => runExperiment(run.experiment, run.id);
    const download = experimentNode("button", "Скачать JSON"); download.onclick = downloadExperimentResult;
    row.append(replay, download); container.append(row, details);
    container.scrollIntoView({behavior:"smooth",block:"start"});
}
function downloadExperimentResult() {
    if (!selectedExperimentRun) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(selectedExperimentRun, null, 2)], {type:"application/json"}));
    const link = document.createElement("a"); link.href = url; link.download = selectedExperimentRun.experiment + "-" + selectedExperimentRun.id + ".json"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function loadExperimentHistory() {
    try {
        const history = await (await api("experiments/results")).json(); $("experimentHistory").replaceChildren();
        if (!history.length) $("experimentHistory").textContent = "Запусков пока нет";
        for (const item of history) {
            const button = experimentNode("button", item.experiment + " · " + new Date(item.at * 1000).toLocaleString() + " · " + item.state + " — " + (item.summary || ""), "history-entry");
            button.onclick = async () => {
                try { renderExperimentResult(await (await api("experiments/results/" + item.id)).json()); }
                catch (error) { experimentMessage(error.message, true); }
            };
            $("experimentHistory").appendChild(button);
        }
    } catch (error) { experimentMessage("История: " + error.message, true); }
}
async function searchExperimentObject(id) {
    try {
        const label = $("experimentQuery" + id).value.trim();
        if (label.length < 2) throw Error("Введите название предмета, например sock");
        await api("objects/find", {label});
        $("experimentSearchResult").hidden = false;
        $("experimentSearchState").textContent = "Модель ищет: " + label + ". Результат появится здесь.";
        pollExperimentSearch();
    } catch (error) { experimentMessage(error.message, true); }
}
let experimentSearchBusy = false;
async function pollExperimentSearch() {
    if (experimentSearchBusy) return;
    experimentSearchBusy = true;
    try {
        const state = await (await api("objects/find")).json();
        $("experimentSearchState").textContent = "Поиск: " + state.phase + (state.error ? " · " + state.error : "") + "\n" + JSON.stringify(state.result || state.objects || state.detections || [], null, 2);
        if (state.phase === "ready") {
            const blob = await (await api("objects/find/image")).blob(); const url = URL.createObjectURL(blob);
            $("experimentSearchImage").onload = () => URL.revokeObjectURL(url); $("experimentSearchImage").src = url;
        } else if (state.busy || ["loading", "searching", "running"].includes(state.phase)) setTimeout(pollExperimentSearch, 1500);
    } catch (error) { $("experimentSearchState").textContent = error.message; }
    finally { experimentSearchBusy = false; }
}
