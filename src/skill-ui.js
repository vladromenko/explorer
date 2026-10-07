"use strict";

let selectedSkillId = null;
let activeSkillEpisode = null;
let skillCatalogBusy = false;
let skillTrialSession = null;
let skillInterventionId = null;

function skillSyncObserve() {
    const observing = $("skillObserve").checked;
    $("observe").checked = observing;
    $("workflowObserve").checked = observing;
}

function skillPreparation(name) {
    if (/носок|sock/i.test(name)) {
        return "Положите носок на свободный пол в видимой области. Поставьте корзину в достижимом месте. Покажите весь маршрут: поиск, подъезд, захват, перевозку и отпускание в корзину. Для следующих показов меняйте сторону и расстояние.";
    }
    return "Разместите предмет и место назначения так, чтобы их было видно во время показа. Покажите весь процесс обычным ручным управлением. После попытки честно отметьте её результат.";
}

function skillSelected() {
    selectedSkillId = $("skillSelect").value || null;
    const option = $("skillSelect").selectedOptions[0];
    if (selectedSkillId && option) $("skillName").value = option.textContent;
    $("skillPreparation").textContent = skillPreparation($("skillName").value);
    skillRefresh();
}

async function skillCreate() {
    try {
        const name = $("skillName").value.trim();
        const item = await (await api("skills", {name})).json();
        selectedSkillId = item.id;
        $("skillPreparation").textContent = skillPreparation(item.name);
        $("skillRecordState").textContent = "Навык создан. Выберите управление и запишите первый показ.";
        await skillRefresh();
    } catch (error) {
        $("skillRecordState").textContent = error.message;
    }
}

async function skillRename() {
    if (!selectedSkillId) return $("skillRecordState").textContent = "Сначала создайте навык.";
    try {
        const result = await (await api("skills/" + selectedSkillId + "/rename", {
            name: $("skillName").value.trim()
        })).json();
        $("skillRecordState").textContent = "Название изменено; все записи остались с навыком " + result.id.slice(0, 8) + ".";
        await skillRefresh();
    } catch (error) {
        $("skillRecordState").textContent = error.message;
    }
}

async function skillChooseInput(kind) {
    skillSyncObserve();
    await selectTrainingInput(kind);
    $("skillKeyboard").classList.toggle("active", kind === "keyboard");
    $("skillGamepad").classList.toggle("active", kind === "gamepad");
    $("skillControlState").textContent = $("trainingInputState").textContent + " " + $("manualStatus").textContent;
}

async function skillEnableManual() {
    skillSyncObserve();
    await enableTrainingManual();
    $("skillControlState").textContent = $("trainingInputState").textContent + " " + $("manualStatus").textContent;
}

async function skillRecord() {
    if (!selectedSkillId) return $("skillRecordState").textContent = "Сначала нажмите «Обучаться» для выбранного навыка.";
    if (!$("skillObserve").checked) return $("skillRecordState").textContent = "Подтвердите наблюдение за роботом.";
    try {
        const name = $("skillName").value;
        const sock = /носок|sock/i.test(name);
        const result = await (await api("skills/" + selectedSkillId + "/record", {
            observing: true,
            object_label: sock ? "носок" : "",
            destination: sock ? "корзина" : ""
        })).json();
        activeSkillEpisode = result.active.id;
        demoSession = activeSkillEpisode;
        $("skillRecordState").textContent = "Запись идёт. Управляйте роботом как обычно; этапы отмечать не нужно.";
    } catch (error) {
        $("skillRecordState").textContent = error.message;
    }
}

async function skillFinish(outcome) {
    if (!selectedSkillId || !activeSkillEpisode) return $("skillRecordState").textContent = "Нет записи для завершения.";
    try {
        const result = await (await api("skills/" + selectedSkillId + "/finish", {
            episode_id: activeSkillEpisode, outcome
        })).json();
        demoSession = null;
        manual = false;
        const episode = result.recording.last;
        $("skillRecordState").textContent = episode && episode.quality && episode.quality.usable
            ? "Показ сохранён. " + (outcome === "success" ? "Обработка запланирована автоматически." : "Результат сохранён для анализа.")
            : "Показ сохранён, но для обучения нужен новый пример: " + (episode?.quality?.reason || episode?.reason || "недостаточно данных");
        await skillRefresh();
    } catch (error) {
        $("skillRecordState").textContent = error.message;
    }
}

async function skillStartTrial() {
    if (!selectedSkillId || !$("skillObserve").checked) {
        return $("skillTrialState").textContent = "Для физической попытки подтвердите наблюдение.";
    }
    try {
        const check = await (await api("skills/" + selectedSkillId + "/trial")).json();
        if (!check.ready) {
            $("skillTrialState").textContent = "Проверка пока недоступна: " + check.blocked_by.join("; ");
            return;
        }
        const state = await (await api("skills/" + selectedSkillId + "/trial", {observing: true})).json();
        skillTrialSession = state.session;
        $("skillTrialState").textContent = "Наблюдаемая попытка начата. STOP или ручной ввод отменяют её.";
    } catch (error) {
        $("skillTrialState").textContent = error.message;
    }
}

async function skillStopTrial() {
    if (!skillTrialSession) return;
    try {
        await api("skills/trial/stop", {});
        $("skillTrialState").textContent = "Попытка остановлена; результат нужно оценить отдельно.";
    } catch (error) {
        $("skillTrialState").textContent = error.message;
    }
    skillTrialSession = null;
}

async function skillLabelIntervention(outcome) {
    if (!skillInterventionId) return;
    try {
        const result = await (await api("skills/trial/intervention/outcome", {
            episode_id: skillInterventionId, outcome
        })).json();
        $("skillInterventionState").textContent = "Исправление сохранено: " + result.outcome +
            (result.quality?.usable ? ". При успешной оценке оно войдёт в набор." :
                ". Для обучения нужна более полная запись.");
        await skillRefresh();
    } catch (error) {
        $("skillInterventionState").textContent = error.message;
    }
}

async function skillRefresh() {
    if (skillCatalogBusy) return;
    skillCatalogBusy = true;
    try {
        const [data, trial] = await Promise.all([
            api("skills").then(response => response.json()),
            api("skills/trial/status").then(response => response.json())
        ]);
        const select = $("skillSelect");
        const previous = selectedSkillId || select.value;
        select.innerHTML = "<option value=\"\">Новый навык</option>";
        for (const item of data.skills) {
            const option = document.createElement("option");
            option.value = item.id;
            option.textContent = item.name;
            select.appendChild(option);
        }
        selectedSkillId = data.skills.some(item => item.id === previous) ? previous : data.skills[0]?.id || null;
        select.value = selectedSkillId || "";
        const skill = data.skills.find(item => item.id === selectedSkillId);
        if (data.active_recording?.skill_id === selectedSkillId) {
            activeSkillEpisode = data.active_recording.id;
            demoSession = activeSkillEpisode;
            $("skillRecordState").textContent = "● Идёт запись: " + data.active_recording.samples + " кадров" +
                (data.active_recording.sampling_warning ? " · " + data.active_recording.sampling_warning : "");
        }
        if (skill) {
            if (document.activeElement !== $("skillName")) $("skillName").value = skill.name;
            $("skillPreparation").textContent = skillPreparation(skill.name);
            $("skillNextAction").textContent = "Следующий шаг: " + skill.next_action;
            $("skillTrialPanel").hidden = skill.latest_job?.state !== "validated_offline";
            const intervention = trial.skill_id === selectedSkillId ? trial.intervention : null;
            skillInterventionId = intervention?.id || null;
            $("skillInterventionPanel").hidden = !intervention || intervention.state === "recording" || intervention.outcome !== "unknown";
            if (intervention?.state === "recording") {
                $("skillTrialState").textContent = "Записываются ваши исправляющие действия: " + intervention.samples + " кадров.";
            } else if (intervention && intervention.outcome === "unknown") {
                $("skillInterventionState").textContent = "Оцените результат исправления после остановки движения.";
            }
            $("skillDiagnostics").textContent = JSON.stringify({
                skill_id: skill.id,
                successful_usable: skill.successful_usable,
                failed: skill.failed,
                last_episode: skill.episodes[0] || null,
                latest_job: skill.latest_job || null,
                earlier_recordings: data.earlier_recordings
            }, null, 2);
        } else {
            $("skillTrialPanel").hidden = true;
            $("skillNextAction").textContent = "Следующий шаг: задайте название и нажмите «Обучаться».";
            $("skillDiagnostics").textContent = JSON.stringify({earlier_recordings: data.earlier_recordings}, null, 2);
        }
    } catch (error) {
        $("skillNextAction").textContent = "Нет связи с обучением: " + error.message;
    } finally {
        skillCatalogBusy = false;
    }
}

setInterval(() => {
    if (document.body.dataset.view === "training") skillRefresh();
}, 1200);
setInterval(async () => {
    if (!skillTrialSession) return;
    const observing = $("skillObserve").checked && document.visibilityState === "visible";
    try {
        const state = await (await api("skills/trial/heartbeat", {
            session: skillTrialSession, observing
        })).json();
        $("skillTrialState").textContent = state.phase + " · шагов: " + (state.steps || 0) +
            (state.reason ? " · " + state.reason : "");
        if (!observing || !state.busy) skillTrialSession = null;
    } catch (error) {
        skillTrialSession = null;
        $("skillTrialState").textContent = "Попытка остановлена: " + error.message;
    }
}, 250);
skillRefresh();
