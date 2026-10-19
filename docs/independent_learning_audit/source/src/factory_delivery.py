"""Read-only admission for physically checked factory robotio delivery settings."""
import hashlib
import json
import math
from pathlib import Path
import time

ARTIFACTS = {"config/delivery.json", "config/handeye-accepted.json", "config/gripper-accepted.json",
             "config/explorer.urdf", "config/explorer.srdf", "config/factory-arm-startup.json", "config/arm-motion.json"}
SCOPES = {"factory_arm", "camera_geometry", "gripper_contact", "localization"}
SCOPE_ARTIFACTS = {
    "factory_arm": {"config/explorer.urdf", "config/explorer.srdf", "config/factory-arm-startup.json", "config/arm-motion.json"},
    "camera_geometry": {"config/handeye-accepted.json", "config/explorer.urdf", "config/explorer.srdf", "config/delivery.json"},
    "gripper_contact": {"config/gripper-accepted.json", "config/delivery.json", "config/handeye-accepted.json", "config/explorer.urdf", "config/explorer.srdf"},
    "localization": {"config/delivery.json"},
}


def _read(root, name, digest=None):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Недоступен артефакт приёмки: " + str(name))
    raw = path.read_bytes()
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or hashlib.sha256(raw).hexdigest() != digest):
        raise ValueError("Артефакт приёмки изменён: " + str(name))
    return raw


def places_digest(places):
    """Bind selected named poses, excluding mutable save timestamps/DB internals."""
    rows = {name: {key: float(pose[key]) for key in ("x", "y", "yaw")} for name, pose in places.items()}
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_factory_settings(root, evidence=None):
    root = Path(root).resolve()
    if evidence is None:
        evidence = json.loads(_read(root, "config/factory-delivery-acceptance.json"))
    import numpy as np
    if (evidence.get("scope") != "factory_delivery_prerequisites" or evidence.get("joint_state_source") != "command_estimate"
            or evidence.get("controller_protocol") != "factory_micro_ros"):
        raise ValueError("Нужна приёмка доставки для factory robotio command-estimate")
    artifacts, records = evidence.get("artifacts", {}), evidence.get("physical_test_records", {})
    if not isinstance(artifacts, dict) or not ARTIFACTS.issubset(artifacts) or not isinstance(records, dict) or not records:
        raise ValueError("Неполный список настроек и физических попыток factory delivery")
    if set(artifacts) & set(records):
        raise ValueError("Настройки не являются физическими журналами")
    verified = {name: _read(root, name, digest) for name, digest in artifacts.items()}
    config = json.loads(verified["config/delivery.json"])
    handeye = json.loads(verified["config/handeye-accepted.json"])
    gripper = json.loads(verified["config/gripper-accepted.json"])
    epoch = config.get("map_epoch")
    places_sha = places_digest(config.get("place_poses", {}))
    if not isinstance(epoch, str) or not epoch or evidence.get("map_epoch") != epoch or evidence.get("places_sha256") != places_sha:
        raise ValueError("Приёмка не соответствует карте и именованным местам")
    found = set()
    for name, digest in records.items():
        record = json.loads(_read(root, name, digest))
        stamp = record.get("at")
        if (record.get("hardware_executed") is not True or record.get("simulation") is not False
                or record.get("operator_observed") is not True or record.get("outcome") != "passed"
                or record.get("joint_state_source") != "command_estimate" or record.get("controller_protocol") != "factory_micro_ros"
                or type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 < stamp <= time.time()):
            raise ValueError("Журнал не подтверждает наблюдаемую физическую попытку: " + name)
        kind = record.get("kind")
        if kind not in SCOPES:
            raise ValueError("Неизвестная область физической проверки: " + name)
        if kind in ("camera_geometry", "localization") and (record.get("map_epoch") != epoch or record.get("places_sha256") != places_sha):
            raise ValueError("Физическая попытка относится к другой карте или месту: " + name)
        bindings = record.get("artifacts", {})
        if not isinstance(bindings, dict) or any(bindings.get(key) != artifacts[key] for key in SCOPE_ARTIFACTS[kind]):
            raise ValueError("Физическая попытка относится к другим настройкам: " + name)
        if any(key in artifacts and value != artifacts[key] for key, value in bindings.items()):
            raise ValueError("Дополнительная привязка физической попытки устарела: " + name)
        observations = record.get("observation_artifacts", {})
        if not isinstance(observations, dict) or not observations or set(observations) & (set(artifacts) | set(records)):
            raise ValueError("Нет независимых сохранённых наблюдений попытки: " + name)
        for observation, sha in observations.items():
            _read(root, observation, sha)
        found.add(kind)
    if not SCOPES.issubset(found):
        raise ValueError("Не проверены отдельные области: " + ", ".join(sorted(SCOPES - found)))
    for profile, sha_key in ((handeye, "physical_validation_sha256"), (gripper, "evidence_sha256")):
        validation = profile.get("physical_validation_record")
        if not validation or not profile.get(sha_key):
            raise ValueError("Нет исходного физического журнала калибровки")
        _read(root, validation, profile[sha_key])
    if (handeye.get("execution_authorized") is not True or handeye.get("joint_state_source") != "command_estimate"
            or handeye.get("reference_mount") != "arm4" or gripper.get("execution_authorized") is not True
            or gripper.get("aperture_mm_calibrated") is not True):
        raise ValueError("Не принята камера или измеренное раскрытие захвата robotio")
    transform = np.asarray(handeye.get("camera_to_mount_reference"), dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or np.linalg.det(transform[:3, :3]) < .999):
        raise ValueError("Неверное принятое преобразование камеры")
    config.update(camera_to_mount=transform.tolist(), open_deg=gripper["open_deg"], close_deg=gripper["sock_close_deg"],
                  handeye_execution_authorized=True, handeye_physical_validation_record=handeye["physical_validation_record"],
                  joint_state_source="command_estimate", physical_delivery_verified=False)
    for key in ("transport_deg", "search_deg"):
        pose = np.asarray(config.get(key), dtype=float)
        limits = [180, 180, 180, 180, 270, 180]
        if pose.shape != (6,) or not np.isfinite(pose).all() or np.any(pose < 0) or np.any(pose > limits):
            raise ValueError("Нет проверенной позы " + key)
    for key, size in (("floor_plane_base", 4), ("grasp_quaternion_xyzw", 4)):
        value = np.asarray(config.get(key), dtype=float)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError("Непринятая геометрия " + key)
        norm = np.linalg.norm(value[:3] if key == "floor_plane_base" else value)
        if abs(norm - 1) > .001:
            raise ValueError("Непринятая геометрия " + key)
    for key, low, high in (("open_deg", 30, 170), ("close_deg", 30, 170), ("approach_height_m", .03, .10),
                           ("lift_height_m", .04, .12), ("grasp_tcp_offset_m", -.02, .04), ("gripper_linkage_rad", -1.54, 0)):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("Параметр вне проверенного диапазона: " + key)
    if config["close_deg"] <= config["open_deg"] or config.get("guarded_closure_enabled") is not True:
        raise ValueError("Factory delivery требует принятый диапазон и guarded closure")
    search, destination = config.get("search_places"), config.get("destination")
    if not isinstance(search, list) or not 1 <= len(search) <= 5 or any(not isinstance(item, str) or not item for item in search) or not isinstance(destination, str) or not destination:
        raise ValueError("Нет принятых мест поиска и доставки")
    for name in set(search + [destination]):
        pose = config.get("place_poses", {}).get(name, {})
        if any(type(pose.get(key)) not in (int, float) or not math.isfinite(pose[key]) for key in ("x", "y", "yaw")):
            raise ValueError("Нет принятой позы места: " + name)
    zone = config.get("drop_zone", {})
    center = np.asarray(zone.get("center_xyz"), dtype=float)
    if (center.shape != (3,) or not np.isfinite(center).all() or np.linalg.norm(center) > 1
            or not .03 <= float(zone.get("radius_m", 0)) <= .3 or not .005 <= float(zone.get("support_tolerance_m", 0)) <= .03):
        raise ValueError("Нет измеренной области размещения")
    return config


def factory_reference(root, arm, boot=None, allow_moving=False):
    """Mission guards validate sensors/ownership; this checks the command record."""
    state = json.loads((Path(root) / "data/arm-state.json").read_text())
    if not getattr(arm,"boot",None) or state.get("boot_id") != arm.boot or boot is not None and state.get("boot_id") != boot:
        raise ValueError("Исходное состояние руки относится к другому включению")
    phases = {"command_elapsed_observation_required"}
    if allow_moving:
        phases.add("command_in_progress")
    values = state.get("servo_deg", [])
    if (state.get("phase") not in phases or not isinstance(values,list) or len(values) != 6
            or any(type(value) not in (int, float) or not math.isfinite(value) or not 0<=value<=limit
                   for value,limit in zip(values,[180,180,180,180,270,180]))
            or state.get("cancelled") is True or state.get("measured") is not False):
        raise ValueError("Недействительное расчётное состояние factory руки")
    stamp,runtime,end = state.get("at"),state.get("runtime_ms"),state.get("ends_monotonic")
    if (type(stamp) not in (int,float) or not math.isfinite(stamp) or not 0<stamp<=time.time()+.1
            or type(runtime) not in (int,float) or not math.isfinite(runtime) or not 0<=runtime<=60000
            or type(end) not in (int,float) or not math.isfinite(end) or not 0<end<=time.monotonic()+60
            or state.get("source") not in ("operator_observed_reference","operator_factory_home","automatic_factory_startup","timed_factory_command_estimate")):
        raise ValueError("Нет достоверного времени или происхождения factory команды")
    fault = Path(root) / "data/arm-telemetry-fault.json"
    if fault.exists() and json.loads(fault.read_text()).get("at", 0) > state.get("at", 0):
        raise ValueError("Расчётная поза недействительна после сбоя связи")
    return state


def elapsed_reference(state):
    end = state.get("ends_monotonic")
    observed_without_command = state.get("source")=="operator_observed_reference" and state.get("publish_count")==0 and state.get("runtime_ms")==0
    if (state.get("phase") != "command_elapsed_observation_required" or not (state.get("command_completed") is True or observed_without_command)
            or type(end) not in (int, float) or not math.isfinite(end) or time.monotonic() < end + .05):
        raise ValueError("Предыдущая factory команда ещё не завершила свой срок")
    return {"attained": False, "measured": False, "command_completed": True,
            "already_at_command_goal": True, "q_estimated_deg": state["servo_deg"]}


def next_exercises(root):
    """Diagnostics only: never writes acceptance or moves the robot."""
    missing = sorted(name for name in ARTIFACTS if not (Path(root) / name).is_file())
    return {"missing_artifacts": missing, "required_records": sorted(SCOPES), "physical_delivery_verified": False,
            "next_action": "Наблюдаемые отдельные проверки: позы/отмена руки; camera XYZ на известных точках; раскрытие, контакт и удержание носка; возврат к местам поиска и корзине. Сохранить исходные наблюдения и SHA текущих настроек; затем оформить factory-delivery-acceptance.json."}


def exercise_source(scope, source):
    """Reject empty acceptance labels; require the applicable measured/test data."""
    if (scope not in SCOPES or source.get("hardware_executed") is not True or source.get("simulation") is not False
            or source.get("outcome") != "passed" or source.get("kind") != scope + "_exercise"
            or source.get("controller_protocol")!="factory_micro_ros" or source.get("joint_state_source")!="command_estimate"):
        raise ValueError("Нужен содержательный исходный журнал физического упражнения " + str(scope))
    if scope == "factory_arm":
        start, goal = source.get("start_deg"), source.get("goal_deg")
        if (not isinstance(start, list) or not isinstance(goal, list) or len(start) != 6 or len(goal) != 6
                or any(type(q) not in (int,float) or not math.isfinite(q) for q in start + goal)
                or max(abs(a-b) for a,b in zip(start,goal)) < 1 or source.get("command_completed") is not True
                or type(source.get("publish_count")) is not int or source["publish_count"] < 1
                or source.get("cancel_no_further_publications") is not True
                or not source.get("boot_id") or not source.get("trajectory_sha256")):
            raise ValueError("Нет фактической команды пути и проверки отмены руки")
    elif scope == "camera_geometry":
        rows=source.get("comparisons",[])
        if not isinstance(rows,list) or len(rows)<2:
            raise ValueError("Нужны сравнения известных точек минимум в двух позах")
        for row in rows:
            if not isinstance(row,dict) or any(type(row.get(key)) not in (int,float) or not math.isfinite(row[key])
                    or not 0<=row[key]<=limit for key,limit in (("translation_error_m",.018),("rotation_error_deg",3.))):
                raise ValueError("Сравнение камеры не прошло геометрические пределы")
    elif scope == "gripper_contact":
        from grasp_verification import verify_lift
        frames=source.get("before",[])+source.get("after",[])
        at=source.get("at")
        if (type(at) not in (int,float) or not math.isfinite(at) or not isinstance(frames,list) or not frames
                or any(not isinstance(frame,dict) or type(frame.get("at")) not in (int,float)
                       or not math.isfinite(frame["at"]) or not 0<=at-frame["at"]<=30 for frame in frames)
                or at-frames[-1]["at"]>2):
            raise ValueError("Измерения удержания не соответствуют времени физической попытки")
        if (type(source.get("open_aperture_mm")) not in (int,float) or not 20<=source["open_aperture_mm"]<=100
                or type(source.get("closed_gap_mm")) not in (int,float) or not 0<=source["closed_gap_mm"]<=15
                or source.get("guarded_contact_observed") is not True
                or verify_lift(source.get("before",[]),source.get("after",[]),True)["outcome"]!="success"):
            raise ValueError("Раскрытие/guarded contact/независимое удержание носка не подтверждены")
    else:
        lidar=source.get("lidar_median_absolute_error_m",[])
        if (not isinstance(lidar,list) or len(lidar)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=.08 for v in lidar)
                or any(type(source.get(key)) not in (int,float) or not math.isfinite(source[key]) or not 0<=source[key]<=limit
                       for key,limit in (("translation_error_m",.15),("yaw_error_deg",10.)))
                or not isinstance(source.get("places_checked"),list) or not source["places_checked"]
                or any(not isinstance(name,str) for name in source["places_checked"])):
            raise ValueError("Нет проверенного возврата к местам с двумя лидарами")


def record_exercise(root, scope, source_name, observation_names, operator_observed, operator_result):
    """Save an operator annotation of existing bound data; never move or calibrate."""
    root=Path(root).resolve()
    if operator_observed is not True or operator_result not in ("passed","failed"):
        raise ValueError("Нужно явное наблюдение и результат оператора")
    if scope not in SCOPES:
        raise ValueError("Неизвестная область проверки")
    source_raw=_read(root,source_name)
    source=json.loads(source_raw)
    exercise_source(scope,source)
    stamp=source.get("at")
    if type(stamp) not in (int,float) or not math.isfinite(stamp) or not 0<stamp<=time.time():
        raise ValueError("Недействительное время исходного упражнения")
    artifacts={name:hashlib.sha256(_read(root,name)).hexdigest() for name in SCOPE_ARTIFACTS[scope]}
    if any(source.get("artifacts",{}).get(name)!=sha for name,sha in artifacts.items()):
        raise ValueError("Исходный журнал не связан с текущими настройками; повторная физическая проверка необходима")
    observations={}
    if not observation_names:
        raise ValueError("Нужны независимые сохранённые кадры/наблюдения упражнения")
    for name in observation_names:
        raw=_read(root,name)
        if not raw or name in ARTIFACTS or name==source_name:
            raise ValueError("Настройка или сам журнал не являются независимым наблюдением")
        sha=hashlib.sha256(raw).hexdigest()
        if source.get("observation_artifacts",{}).get(name)!=sha:
            raise ValueError("Наблюдение не было привязано в исходном упражнении")
        observations[name]=sha
    observations[source_name]=hashlib.sha256(source_raw).hexdigest()
    record={"kind":scope,"at":stamp,"recorded_at":time.time(),"hardware_executed":True,"simulation":False,
            "operator_observed":True,"outcome":operator_result,"controller_protocol":"factory_micro_ros",
            "joint_state_source":"command_estimate","artifacts":artifacts,"observation_artifacts":observations}
    if scope in ("camera_geometry","localization"):
        config=json.loads(_read(root,"config/delivery.json"))
        epoch,places_sha=config["map_epoch"],places_digest(config["place_poses"])
        if source.get("map_epoch")!=epoch or source.get("places_sha256")!=places_sha:
            raise ValueError("Исходный журнал относится к другим местам/карте")
        record.update(map_epoch=epoch,places_sha256=places_sha)
        if scope=="localization" and set(source["places_checked"])!=set(config["search_places"]+[config["destination"]]):
            raise ValueError("Возврат проверен не во всех местах принятого сценария")
    folder=root/"data/factory-delivery-exercises"
    if not folder.resolve().is_relative_to(root):
        raise ValueError("Каталог упражнений выходит из проекта")
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(scope+"-"+str(time.time_ns())+".json")
    path.write_text(json.dumps(record,ensure_ascii=False,allow_nan=False,indent=2))
    return {"path":path.relative_to(root).as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"record":record}


def seal_factory_exercises(root, record_names, write=False):
    """Seal only a fully validated combination of current, existing records."""
    root=Path(root).resolve()
    artifacts={name:hashlib.sha256(_read(root,name)).hexdigest() for name in ARTIFACTS}
    config=json.loads(_read(root,"config/delivery.json"))
    records={name:hashlib.sha256(_read(root,name)).hexdigest() for name in record_names}
    evidence={"scope":"factory_delivery_prerequisites","controller_protocol":"factory_micro_ros",
              "joint_state_source":"command_estimate","map_epoch":config["map_epoch"],
              "places_sha256":places_digest(config["place_poses"]),"artifacts":artifacts,"physical_test_records":records,
              "physical_delivery_verified":False}
    load_factory_settings(root,evidence)
    # Also require original substantive source data for records produced here.
    for name in record_names:
        record=json.loads(_read(root,name))
        candidates=[json.loads(_read(root,source)) for source in record["observation_artifacts"] if source.endswith(".json")]
        sources=[source for source in candidates if isinstance(source,dict) and source.get("kind")==record["kind"]+"_exercise"]
        if not sources:
            raise ValueError("Нет исходного содержательного упражнения: "+name)
        for source in sources:
            exercise_source(record["kind"],source)
    path=root/"config/factory-delivery-acceptance.json"
    if write:
        if path.exists():
            previous=json.loads(path.read_text())
            if previous!=evidence:
                raise ValueError("Существующий manifest отличается; сохраните его перед явной заменой")
            return {"saved":False,"already_current":True,"physical_delivery_verified":False,"path":str(path),"evidence":evidence}
        temporary=path.with_suffix(".tmp")
        temporary.write_text(json.dumps(evidence,ensure_ascii=False,allow_nan=False,indent=2))
        temporary.replace(path)
    return {"saved":bool(write),"physical_delivery_verified":False,"path":str(path),"evidence":evidence}
