"use strict";

let mappingBusy = false;
let lidarBusy = false;
let lastLidarView = null;
let mapObjectUrl = null;
let savedMapNames = null;

function mappingMessage(message, failed = false) {
    $("mappingActionState").textContent = message;
    $("mappingActionState").className = "statusline " + (failed ? "bad" : "ok");
}

async function refreshMapping(force = false) {
    if (mappingBusy || !token || (!force && (document.body.dataset.view !== "mapping" || !$("mapAutoRefresh").checked))) return;
    mappingBusy = true;
    try {
        const results = await Promise.allSettled([
            api("map").then(response => response.blob()),
            api("mapping/status").then(response => response.json())
        ]);
        if (results[0].status === "fulfilled") {
            const next = URL.createObjectURL(results[0].value);
            const previous = mapObjectUrl;
            mapObjectUrl = next;
            $("mapImage").onload = () => { if (previous) URL.revokeObjectURL(previous); };
            $("mapImage").src = next;
        } else {
            $("mapState").textContent = "Изображение карты недоступно: " + results[0].reason.message;
        }
        if (results[1].status === "fulfilled") {
            const data = results[1].value;
            const metadata = data.map;
            const pose = data.pose;
            const mapAge = Math.max(0, data.at - (metadata.at || 0));
            $("mapFreshness").textContent = metadata.stale ? "Нет свежей карты" :
                "Карта обновлена " + mapAge.toFixed(1) + " с назад · " + metadata.width + " × " + metadata.height +
                " клеток · " + (metadata.resolution * 100).toFixed(0) + " см/клетка";
            $("mapFreshness").className = "statusline " + (metadata.stale ? "bad" : "ok");
            $("mapState").textContent = pose ?
                "Explorer: x=" + pose.x.toFixed(2) + " м · y=" + pose.y.toFixed(2) + " м · курс " +
                (pose.yaw * 180 / Math.PI).toFixed(1) + "° · " +
                (pose.localization_verified ? "повторная локализация принята" : "текущая оценка lidar SLAM") :
                "Карта показана; положение робота сейчас недоступно: " + data.pose_error;
            const names = JSON.stringify(data.saved_maps.map(item => item.name));
            if (savedMapNames !== names) {
                const selected = $("savedMapSelect").value;
                $("savedMapSelect").replaceChildren();
                const empty = document.createElement("option");
                empty.value = ""; empty.textContent = "Выберите сохранённую карту";
                $("savedMapSelect").appendChild(empty);
                for (const item of data.saved_maps) {
                    const option = document.createElement("option");
                    option.value = item.name; option.textContent = item.name;
                    $("savedMapSelect").appendChild(option);
                }
                $("savedMapSelect").value = selected;
                savedMapNames = names;
            }
            $("mappingServices").textContent = Object.entries(data.services || {}).map(([name, value]) => name + ": " + value).join(" · ");
        } else {
            $("mapState").textContent = "Состояние карты недоступно: " + results[1].reason.message;
        }
    } finally { mappingBusy = false; }
}

function drawLidar(data = lastLidarView) {
    if (!data) return;
    lastLidarView = data;
    const canvas = $("lidarCanvas");
    const size = Math.max(280, Math.min(900, canvas.clientWidth || 600));
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = size * dpr; canvas.height = size * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "#080b19"; ctx.fillRect(0, 0, size, size);
    const radius = Number($("lidarRange").value);
    const factor = (size * .46) / radius;
    const center = size / 2;
    ctx.strokeStyle = "#303955"; ctx.fillStyle = "#aeb5d4"; ctx.font = "12px system-ui";
    for (let meters = 1; meters <= radius; meters++) {
        ctx.beginPath(); ctx.arc(center, center, meters * factor, 0, Math.PI * 2); ctx.stroke();
        ctx.fillText(meters + " м", center + 5, center - meters * factor + 14);
    }
    ctx.beginPath(); ctx.moveTo(center, 0); ctx.lineTo(center, size); ctx.moveTo(0, center); ctx.lineTo(size, center); ctx.stroke();
    const colors = ["#8296ff", "#da9cff"];
    data.sensors.forEach((sensor, index) => {
        ctx.fillStyle = sensor.fresh ? colors[index] : "#59617c";
        for (const [x, y] of sensor.points_m || []) {
            if (Math.hypot(x, y) <= radius) ctx.fillRect(center - y * factor - 1, center - x * factor - 1, 2.5, 2.5);
        }
        if (sensor.sensor_origin_m) {
            const [x, y] = sensor.sensor_origin_m;
            ctx.beginPath(); ctx.arc(center - y * factor, center - x * factor, 4, 0, Math.PI * 2); ctx.fill();
        }
    });
    ctx.strokeStyle = "white"; ctx.lineWidth = 2;
    ctx.strokeRect(center - .11 * factor, center - .17 * factor, .22 * factor, .34 * factor);
    ctx.fillStyle = "#ffcf66";
    ctx.beginPath(); ctx.moveTo(center, center - .27 * factor - 12); ctx.lineTo(center - 7, center - .27 * factor); ctx.lineTo(center + 7, center - .27 * factor); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "white"; ctx.fillText("ВПЕРЁД", center + 12, 20);
    $("lidarMetrics").replaceChildren();
    data.sensors.forEach((sensor, index) => {
        const row = document.createElement("div"); row.className = "metric";
        const title = document.createElement("strong"); title.style.color = colors[index];
        title.textContent = sensor.name + " · " + (sensor.fresh && sensor.transform_valid ? "обновляется" : sensor.reason || "нет данных");
        const info = document.createElement("div"); info.className = "muted";
        info.textContent = sensor.valid_returns == null ? "Ожидание сканов" :
            sensor.valid_returns + " измерений · " + (sensor.rate_hz || 0).toFixed(1) + " Гц · возраст " + sensor.source_age_s + " с · " + sensor.frame;
        row.append(title, info); $("lidarMetrics").appendChild(row);
    });
    $("liveLidarState").textContent = data.all_fresh ? "Оба скана свежие, положение датчиков взято из TF робота" : "Проверьте статус датчиков выше; серые точки относятся к старому скану";
}

async function refreshLidar(force = false) {
    if (lidarBusy || !token || (!force && (document.body.dataset.view !== "mapping" || !$("mapAutoRefresh").checked))) return;
    lidarBusy = true;
    try { drawLidar(await (await api("lidar")).json()); }
    catch (error) { $("liveLidarState").textContent = "Нет обновления лидаров: " + error.message; }
    finally { lidarBusy = false; }
}

async function refreshMapNow() { await Promise.all([refreshMapping(true), refreshLidar(true)]); }
async function recoverMapping() {
    try {
        const result = await (await api("mapping/recover", {})).json();
        mappingMessage(result.started.length ? "Запущены: " + result.started.join(", ") + ". Ожидаю новые данные." : "Все службы работают. Обновляю отображение.");
        await refreshMapNow();
    } catch (error) { mappingMessage(error.message, true); }
}
async function saveMap() {
    try { await api("maps/save", {name: $("mapName").value.trim()}); mappingMessage("Карта сохранена"); await refreshMapping(true); }
    catch (error) { mappingMessage("Сохранение: " + error.message, true); }
}
async function loadSavedMap() {
    if (!$("savedMapSelect").value) return mappingMessage("Сначала выберите сохранённую карту", true);
    try {
        await api("maps/load", {name: $("savedMapSelect").value});
        mappingMessage("Карта загружена. Сопоставьте текущее положение робота с окружением перед автономной поездкой.");
        await refreshMapping(true);
    } catch (error) { mappingMessage("Загрузка: " + error.message, true); }
}

setInterval(() => refreshMapping(), 1000);
setInterval(() => refreshLidar(), 250);
window.addEventListener("resize", () => drawLidar());
