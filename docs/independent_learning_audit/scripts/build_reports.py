"""Assemble audit documents from pinned history and newly measured evidence."""
import csv
import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

OUT=Path(__file__).resolve().parents[1]
REPO=OUT.parents[1]
OLD=(REPO/'docs/EXPLORER-LEARNING-REPORT-2026-10-19.ru.md').read_text()


def write(name,text):
    (OUT/name).write_text(text.rstrip()+'\n')


def section(number):
    marker=f'## {number}.'
    start=OLD.index(marker)
    match=re.search(r'^## (?:\d+\.|Приложение)',OLD[start+len(marker):],re.M)
    end=start+len(marker)+match.start() if match else len(OLD)
    value=OLD[start:end]
    definitions='\n'.join(line for line in OLD.splitlines() if re.match(r'^\[[^]]+\]:',line))
    return value+'\n'+definitions+'\n'


def main():
    data=json.loads((OUT/'dataset_statistics.json').read_text())
    forensic=json.loads((OUT/'examples/refusal-695/refusal_forensics.json').read_text())
    stats=json.loads((OUT/'evidence/checkpoint_inventory_summary.json').read_text())
    distribution_path=OUT/'evidence/runtime_distribution_followup.json'
    follow=json.loads(distribution_path.read_text()) if distribution_path.exists() else None
    history_intro='''# 02. Полная доступная история: 27 сентября — окончание аудита

## Границы восстановления

VERIFIED: доступный Git начинается27.09, доступная переписка —27.09. Более ранняя часть условного «последнего месяца» отсутствует; её события не реконструируются. Полный перечень коммитов всех сохранённых веток находится в `evidence/git_history.tsv`, все refs — `evidence/git_refs.txt`. Наличие remote-tracking ref не доказывает нынешнюю доступность аппаратуры.

## 27–30 сентября: до Mac-симуляции

- 27.09: sensing/микро-ROS/две LiDAR/camera/localLLM, watchdog и provisional mapping. Надёжный timeout базы и measured arm state оставались нерешёнными. Пользователь уже требовал учить программой/по его показам, не ручными действиями ассистента. Добавлены LeRobot/ACT и supervised execution, но smoke не обучил носку.
- 28.09: bounded lab, clocks, повторяемость руки, hand-eye и один наблюдаемый pickup/carry/release. Это инженерный интеграционный опыт с конечными командами, не ученик ACT. Hand-eye сначала не проходила критерий, затем новая серия приблизилась/прошла конкретный static test; переносимость под нагрузкой не проверена.
- 28–29.09: отдельная ветка firmware/native controller, CubeProgrammer, реальные сбои ARM_ENABLE/limits/TC, подготовка выпусков и отзывrc2. Важный доказанный ошибочный вывод: «нет штатного sync write» опровергнут исходником Arm_Sync_Write; torque broadcast мог включать сустав без свежего измерения. Старое заявление о физически проверенном формате/отсутствии рывка не было доказано. Отозванный образ не считать установленным. Детали в `source/docs/firmware-rc2-withdrawal-20260929.md` и `source/docs/ARM-RESTORATION-20260929.ru.md`.
- 29.09: возврат к factory-compatible robotio и отдельный watchdog образ; фактическую нынешнюю прошивку read-only по сети проверить не удалось. История содержит предложения разных imageSHA и их отмены. Нельзя из слова «готово» вывести, что каждый предложенный образ был записан. В текущей конфигурации joint state — команда, не измерение; уточнение пользователя обязательно имеет приоритет над промежуточной интерпретацией старых raw logs.
- 30.09: factory startup/reference pose, phone controls, guided calibration/graduation, smoothing/manual policies. Они решали управление и сбор показов, не доказывали автономную доставку.

## Исторический подробный отчёт1–19октября

Ниже встроен ранее составленный отчёт полностью, чтобы архив оставался самостоятельным. Его разделы о «сейчас обучается» — ИСТОРИЧЕСКИЙ срез22:36–22:49, а не состояние после пользовательской паузы. Текущий статус в17. Новые данные полного streams/visual sample и forensic replay приведены в05–07 и имеют приоритет над прежними предположениями.

'''
    write('02_FULL_HISTORY.md',history_intro+OLD)
    hardware='''# 03. Аппаратура, симуляция и достоверность параметров

## Действующая конфигурация и источники

VERIFIED по прямому уточнению пользователя и factory-коду: **обратной связи суставов в действующей системе нет**. FK/state строятся по отправленным командам. Сентябрьские ответы экспериментального native-контроллера не дают права переобъявить эту конфигурацию measured. Ассистент допустил слишком сильную промежуточную интерпретацию во время аудита; она отозвана.

Текущий live inventory Jetson недоступен: оба существующих SSH aliases дают DNS failure. Никакие actuator/firmware команды не выполнялись. Поэтому частоты и задержки ниже имеют дату/тип источника, а не выдаются за сегодняшнее измерение.

| Узел | Модель/назначение/связь/интерфейс | Частота и задержка | Feedback/команды/ограничения | Симуляция / обучение / inference |
| --- | --- | --- | --- | --- |
| Шасси | ROSMASTER M3 Pro,4mecanum, штатный MCU→USB→Jetson, micro-ROS velocity/odom | Исторический08.10 odom61msg/8с,maxinterval114мс; нынешняя latencyUNKNOWN | Bodyvx/vy/wz, encoder odom исторически; slip/STOP/footprint обязательны | Контактная mecanum модель; policy base outputs вroomпринудительно0, внешний навигатор |
| Рука | Номинально5осей+захват,6servochannels; штатные конечные servo endpoints | Hoststream40мс исторический; реальная joint responseUNKNOWN | Нет measured суставов; command cache не encoder; force/apertureUNKNOWN | MuJoCo torque-limitedPD с оценочными gains; action9D |
| MCU | Пользовательская спецификацияF407; исторический CubeProgrammer в переписке сообщал chip0x450/H7 | Текущая identityUNKNOWN | Исторические native/factory интерфейсы различаются; не менять firmware | MCUнеэмулируется побитно; симконтракт вышеуровневый |
| LiDAR0/1 | ДваTmini-plus,UART/MCU→ROS scan0/scan1 |08.10 каждый39msg/8с,max155/152мс | Range scans; blind zones/calibration; currentlatencyUNKNOWN | Симscan/ray модель; возможная идеализация noise/timing |
| WristRGB-D | DaBai/DCW2 в исторических логах,USB наJetson, cameraInfo+registereddepth |11.10 послеfix≈15fps; отдельноеисторическое108мс SDKdepthlag | Задержка доROS, SDKsync неподдерживался; geometryтолькоизсвежихframes | Simpinhole320×240actor/640×480reference; дисторсия/реальныйnoiseневоссозданыполностью |
| USBкамера | Logitech/external observed camera; точнаямодельcurrentUNKNOWN | Historicalvideoview; нетнезависимоготекущегоfpsprobe | Внешнее наблюдение инженера не actor sensorавтоматически | Нельзяподменятьwristviewвобучении |
| Jetsonосновной | OrinNano8GB поразвёртыванию, Linux/ROS2/CUDA |14.10 старыйACT oneforward0.967с, неполныйlatencybenchmark | ТекущаядоступностьDNSfalse; память/питаниеограничены | Inferencecandidateпослеотдельноготеста; неготовыйcurrentpackage |
| Mac | M4Pro24GiB,arm64macOS15.6.1, внешнийAPFS/T5 |MPSupdatesизмерены, подробности07/17 | SharedRAM,CPUphysics/IO, operatorpauseсохранена | РеальныеACTupdates, MuJoCoCPUphysics/render; неполныйGPUpipeline |
| ExtraJetson | OrinNX16GB поуказаниювладельца,aliasHomeBrainисторически | CurrentDNSfailure | НеобъявленподключённымкROSMASTER | Потенциальныйremoteinference/compute, требуетотдельногоизмерения |
| Pi/Hailo | RaspberryPi5/Hailo-10H поуказаниювладельца | UNKNOWN | Нетпроверенногоcurrentdeviceinterface/unsupportedACTcompiler | Возможнаяperceptionoffload — HYPOTHESIS; неуниверсальныйPyTorchaccelerator |

## Сравнение физики

| Параметр | Всимуляции | Наэкземпляре | Источник/достоверность | Ошибкапереноса / измерение |
| --- | --- | --- | --- | --- |
| CAD/FK |20STL+URDF/SRDF,22SHA, сторонняяревизияbdac682 | Некоторые staticposes/hand-eye, невсеdimensionsизмерены | VERIFIEDbytes, неVERIFIEDrealgeometry | Измеритьоси/offsets/базукамеры |
| Jointunits | Градусыendpoint, FКпоservo mapping | Командныеуглы, неfeedback | VERIFIEDcontract | Люфт/loadtracking; измерятьпоcamera/markerбезвыдуманногоencoder |
| PD |kp25,kv0.65,torque[1,1.2,1.2,.9,.7,.35]Nm | UNKNOWN | Estimates | БезопасныйsystemIDдвиженийпотребуетотдельногоразрешения |
| Contact | floorfriction.8, sockпример1.1, capsulepads4мм×28мм | UNKNOWN | Estimates | Sensitivity/материальныеизмерения |
| Cloth | flexshell7rings×8vertices, mass.022кг, Young10000Pa,thickness.0007м | Реальныйносокнеидентифицирован | Parametersнеизмерены | Взвешивание/геометрия/статическиеcomplianceтесты |
| Gripper |30→170°, affine linkage0→-1.1rad | Историческиносокдержалсяоколо160°; aperture/forceUNKNOWN | Наблюдение, некалибровкасилы | Проверитьmapping/податливость, неслепойtransfer |
| Camera |fx480.7079/fy480.7113,cx322.29/cy245.64reference640×480,near.015/far8 | ИсторическийcameraInfo; currentUNKNOWN | Записаннаяcalibration, неliveprobe | Сверитьintrinsics/extrinsics/timestampsнаrealframes |
| Distortion/delay | Идеальныйpinhole; coefficientsзаписаныкакprovenance | Историческиdistortion/SDKlagесть | VERIFIEDразличие | Реальныйrender/latencyrandomizationпоизмерению |
| Timing |physics.002с, action.05с,дополнительныйsettlingявно | MCU/USBservoresponseUNKNOWN | Simulationknown | Сверитьendpoints, end-to-endtimestamps |
| Localization | Actorиспользуетobservedmap/pose; полнаяошибкаodomнеидентифицирована | Повторнаялокализациянепринята | UNKNOWNrealaccuracy | Неиспользоватьground-truthbaseposeкакреальныйдатчик |

Источники и числовая provenance: `source/config/arm_model.json`, `source/simulator/model_config.json`, `evidence/historical/geometry-provenance.json`, исходникиexplorer_model/environment и историческиеhardware-документы. Никакие неизвестные значения в этой таблице не подставлены благоприятными числами.

'''
    write('03_HARDWARE_AND_SIMULATION.md',hardware+section(11))
    write('04_ALGORITHM_REGISTRY.md','# 04. Реестр алгоритмов и статусы применения\n\n'+section(6)+'''
## Обязательные поля, которым нельзя приписать несуществующие результаты

Реализация и параметры каждого семейства закреплены source/functionindex; затраты по каждому gradient run — experiments.csv/checkpoints.csv. Для ранних scorer/OutcomeModel/actor-critic нет полной подтверждённой физической учебной серии: compute hours, числоacceptedphysicalepisodes и learningcurve обозначаются UNKNOWN, а не нулями или успехом. Ridge phase-модели не использовались в runtime. CEM/programrepair оптимизируют кодовые параметры, а не ACTweights.

Выбор методов первоначально определялся доступной интеграцией, ограничениемMac/Jetson и уже существующими геометрическими исполнителями. История переписки19 показывает, что это не был контролируемый конкурс алгоритмов. Причина отсутствия полного решения у каждого подхода различна: недостаточнаяphysicalacceptance, отсутствующаясвязьdataset→consumer, ограниченныйrepairoperatorbank, local/fullshift, либо простоотсутствиевалидированногозапуска.

Сохранять: контракты/recorders/версии/gates, полезныеvalidatedgeometryкомпоненты и совместимыеданные. Заменятьтолько послеплана12. VERIFIED — наличиепараметров/updates/отдельногоуспеха; HYPOTHESIS — causalрольконкретногоалгоритма; UNKNOWN — сравнительное превосходство. Ни одна альтернативнаяVLA не обучалась в нынешнем d4run.
''')
    visual=json.loads((OUT/'evidence/visual_and_state_followup.json').read_text())
    dataset=f'''# 06. Датасеты: полный текущий samples stream и ограниченный визуальный аудит

## Что реально проверено заново

VERIFIED: {data['recording_count']} записей текущегоrun, {data['total_raw_rows']} исходных строк, {data['target_masked_rows']} masked student rows, {data['nonfinite_rows']} nonfinite rows, {data['nonincreasing_clock_pairs']} невозрастающих пар выбранного clock. Источники найдены по точному обучающему samplesSHA, а не по одному ID.

Инвентаризировано {data['raw_episode_manifests']} raw episode manifests поддержанных семейств; declared outcomes: {data['raw_declared_outcomes']}. Это декларации manifest, не новая переоценка успеха каждой физической попытки. Все datasets/runs с доступной provenance перечислены в `evidence/dataset_manifest_inventory.json`; старые image-потоки целиком повторно не декодировались. Поэтому фраза «все исторические изображения полностью проверены» здесь не используется.

Полныйтекущийidle: **{data['idle_fraction_of_unmasked']:.2%}** unmasked строк при определении maxarm(target-state)<0.1° и maxbasevelocity<.005. Это не просто оценка14показов. По всем629019validtargets nextstate точносовпадаетспредыдущейaction во всех629220переходахпотоков, включаяmaskedстроки: max0. Это подтверждаеткоманднуюсемантику,stateнеравенmeasuredqpos.

## Visual/scene similarity

Проверено {data['visual_sample_count']} кадров: initial/close/place поодномунаэпизод, SHA+декодирование+RGB16×12thumbnail. Ввыборке0missing/hashmismatch. Это **не** проверкавсех629959изображений. Близостьthumbnailнепроверяетсемантическуюидентичностьсцены.

Найдены **{visual['cross_split_exact_image_hash_groups']} группы побайтноодинаковыхRGBфайловмеждуtrain/validation**, охватывающие {visual['cross_split_exact_image_samples']} sampledframes. Для6группсовпадаюттакжеstate/clock,ноtargetотличаетсятольковчисленнойпогрешности(максимумоколо3e-7°). Это важныйnegativecontrol: **информативныхпротиворечивыхtargetsнаэтиходинаковыхстартахнеобнаружено**. Одинаковыйпустой/стартовыйкадрне доказываетодинаковуюсценуилиполнуюsceneleakage. Однако утверждение«RGBполностьюнезависимымеждуsplit»теперьопровергнутоограниченнойвыборкой.

Командныйаудитранее:1595neartrajectorypairs,524crosssplit,0exactsignatures. Нельзя смешиватьэто с4группамиimagehash. Полнаяscene-levelиерархиядубликатов/семантическихподобийостаётсяUNKNOWN.

## Распределения из первичных строк

| Признак | Min | Median | P95 | Max |
| --- | --- | --- | --- | --- |
'''
    for key in ['height','sock_position','basket_position','lighting','mass','young','friction','sock_rgb','basket_rgb']:
        d=data['domain_distributions'][key]
        dataset+=f"| {key} | {d['minimum']} | {d['median']} | {d['p95']} | {d['maximum']} |\n"
    dataset+='''
Начальнаякоманднаяпоза739/739одинакова: [90,90,90,90,90,30,0,0,0]. Длительностиmedian42.2с, min40.45/max53.2. Носокпригенерациипервоначальноx≈.28–.36м,y≈±.035м;корзинаx≈.43–.47м,y≈.267–.417м. Это локальный диапазон, не демонстрации поиска по квартире. Цвет/свет/материал меняются, но развёрнутыйroomstudentпопадаетвинуюгеометрию/ракурс/временнуюфазу.

Распределениякаждогоканала,фазы,targets,teacherprofilesииндивидуальныхзаписей—dataset_statistics.json иevidence/recording_statistics.json. Вcurrenttrain590/val149 исходныхзаписей; frames502939доreplay/583195после;val126080. Всырыхпотоках629959строк,изних940masked;эторазныеединицыучёта. Replay и фильтрацияэкспертныхсегментовнеравныrawcount.

## Student/teacher/current room

Историческиеteacherданныеподтверждаютsuccessсвоимстарымоценщиком,непринятымновым100-sceneпротоколом. Corrections19/24—локальнаяпилотнаясерия; fullroomstudentstatecorrectionsещёнедоказанывнужнойквоте. Runtimeinput695в05 неимеетзаписанногоexpertcounterfactual. NearestsampledRGB+stateиз2217кадров не являетсяnearestво всех629959наблюдениях.

Следовательно, distribution shift структурно VERIFIED, а его точная causalдоля и отсутствиепокрытиякаждогосостояния—UNKNOWN. Не объявлятьвсеteacherlabels ложными и не удалятьих; отдельно переоценитьпо currenttruthgate исовместимомуисполнению.

### Полный follow-up по диапазону состояния и clock

'''
    dataset+=('```json\n'+json.dumps(follow,ensure_ascii=False,indent=2)+'\n```\n') if follow else 'Follow-up ещё не завершён; не делать численных заявлений.\n'
    write('06_DATASET_AUDIT.md',dataset)
    write('07_TEACHER_AND_CONTROL.md','# 07. Teacher, управление и время\n\n'+section(8)+section(9)+section(15)+'''
## Различие physical teacher и runtime

VERIFIED: teacher получает privileged simulation geometry для генерации, actor получаетwhitelistRGB/state/clock. Hidden diagnostics остаютсявсыройзаписи, но не являютсяactorfeaturesвexport_dataset. TeacherпланируетIK/closure/lift/transfer/releaseчерезреальнуюMuJoCoфизику,сохраняетисходпослеокончания. Oldground_truthteacherнеравенновомуobservedroomexpert.

СтабильностьteacherширокогополногоroomсценарияUNKNOWN. Исторические426/700—teacherlocal,неполнаямиссия;paired7/10vs5/10неstudent. Новыйfullcollectorтребует≥8fullsuccessи≥4успешныхstudent-statecorrections,ноего62теста/20-actionpreflightнеподтверждаютэтуквоту.

В current streams VALID target-current максимум по6каналам≈[1.217,2.4,2.4,3,3,3.6]°,runtimeissued2°. Это не скалярная ошибка «градусы/радианы», а потенциально накопленная разница исполнения. Rawprediction+27.82°в695необъясняетсяоднойразницей3.6vs2,нуженtrajectory/stateanalysis.

Известный cadence дефект исправленпослеисторического fresh joint3refusal:12manualticksпосле10ACTдавали22сphysics/10сclockна200решениях. Regression cfe4aa7даёт10/10с, носледующий760-actionпрогонвсёравнонеподнялносок;timingrepairнебылполнымрешением.

Текущий trace695: physicalelapsed34.75с, pickclock6→40.75с. Разностьчасовсоответствуетфиксированномуphaseoffset6с,тоестьстарый22/10багнеобъясняетэтототказ. Приэтомtaskphaseостаётсяpick,хотяclockужеуходитвпоздниефазыteacher. Эторискsemanticphase/timecontract;егоcausalрольпроверятьпо06иclockablation.

Правильныйteacher→policytestдолженсравниватьне толькокоманды,ноqpostracking,видимостьпредмета,получениеконтакта,удержание,releaseиактуальнуюфинальнуюстабильность. Этислоиразделены;увеличениеtrainstepsнепроверяетихавтоматически.
''')
    path=OUT/'05_ACT_FORENSICS.md';s=path.read_text()
    lines=[f"VERIFIED: CPU replay первой команды совпал с исторической **точно**: max absolute error `{forensic['max_abs_reproduction_error']}`. Guard снова отказал: `{forensic['guard']}`.",
        '', 'Нормализованныйstate: `'+str(forensic['normalized_state'][0])+'`.', '',
        'Полный денормализованныйgripperchunk: `'+str([round(x[5],3) for x in forensic['full_denormalized_chunk'][0]])+'`.', '',
        '| Counterfactual | Clock,с | Gripper,° | Shoulder,° | Elbow,° |','| --- | --- | --- | --- | --- |']
    for row in forensic['counterfactuals']:
        a=row['first_endpoint'];lines.append(f"| {row['kind']} | {row['clock_s']:.2f} | {a[5]:.4f} | {a[1]:.4f} | {a[2]:.4f} |")
    lines+=['','Часы и RGB влияют на прогноз. Это опровергает слишком сильный вывод«изображение вообще не используется»,но не доказываетдостаточнуюvisualreactivity. Gray/blackконтрфактуалыOOD и неуспешныефизическиеэксперименты.',
        '', 'На694применёнgripper48°,cachedtarget48.1484°. Следующийfreshforward695выдал75.8205°: межchunkразрыв≈27.6721°. Вtraceестьодинcachedreplan;695неcached. Доказаначисленнаядискретностьнагранице,еёобучающаяпричинанеизолирована.']
    s=s.replace('<!-- FORENSIC_NUMBERS -->','\n'.join(lines));path.write_text(s)
    write('14_SOURCE_MANIFEST.md',f'''# 14. Манифест источников и полнота доступа

Функциональная ревизия1da8ecb26efaa798d4293f126d7d123aff7985e5. Кодсохранёнподsource/,хешивevidence/source_manifest.json,функции/строкивevidence/SOURCE_FUNCTION_INDEX.md. ПолнаяструктураGit—full_git_tree.txt;всеlocal/remoterefs—git_refs.txt;историявсехветок—git_history.tsv. Это не повторныйfetchудалённогоаппаратногоJetson.

Проверены **{stats['entries']} путей checkpoints**, **{stats['distinct_resolved_files']} различныхразрешённыхфайлов**, **{stats['distinct_hashes']} различныхSHA**,потоковопрочитано **{stats['physical_bytes']} байт** весов. Aliaslastперечислен,нонехешированповторно. checkpoints.csvсодержитточныепути,размеры,SHAиmissingtrainingstatefiles. Этохешивесов;всемoptimizer/RNGнеприписанаtensorsemanticvalidation.

Всеrunsсjob/provenanceинвентаризированывexperiments.csv/run_inventory.json/dataset_manifest_inventory.json. Текущийfullsamplesstreamпроверенв06;старыеисторическиеimagesнепрочитанывсе. Нельзяназыватьбайтнуюцелостностьвсехсохранённыхснимковдоказаннойтолькоповесуmodel.safetensors.

Тяжёлыеданные/весаостаютсяна/Volumes/ExplorerSim/ExplorerSim;вZIPвключеныихманифесты,не30GiBвесов. Дляточногоreplayна другомкомпьютеренуженcheckpointссоответствующимSHA;путь/размер/версиявcheckpoints.csvиexamples/refusal-695/refusal_forensics.json. Пакетнамереннонесодержитвнешниемоделивполномобъёме.

Очищеннаяперепискаупакованавevidence/conversation_messages.redacted.jsonl,ноисключенаизпубликацииGitчерезлокальный.gitignore:эточастнаякорреспонденция. ВGitсохраняютсявыводыиотобранныетехническиецитаты19,аполныйфайлпередаётсявладельцувZIP. Секретыдоступаневключаются.

audit_manifest.jsonперечисляетфайлыпакетасразмерами/SHA(кромесамогосебяициркулярногоSHA256SUMS). SHA256SUMSконтролируеттакжеманифест. ZIPпроверяетсяотдельнымполнымSHAрядомсархивом.
''')
    print('Core reports assembled')


if __name__=='__main__':
    main()
