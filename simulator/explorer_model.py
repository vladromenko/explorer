"""Explorer CAD-derived MuJoCo model, with physical roller and cloth contacts.

No robot I/O. CAD geometry/kinematics and accepted extrinsics are preserved;
unknown actuator/material values are explicitly estimates in model_config.json.
Visual STL meshes never act as contact hulls. Wheel motion is generated only by
motor torques, passive roller joints, gravity and frictional contact.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ARM_JOINTS = tuple(f"arm{i}_Joint" for i in range(1, 6)) + ("rlink1_Joint",)
WHEEL_NAMES = ("lwheel1", "rwheel1", "lwheel2", "rwheel2")  # FL FR RL RR
WHEEL_JOINTS = tuple(f"{name}_Joint" for name in WHEEL_NAMES)


def model_config():
    return json.loads((Path(__file__).with_name("model_config.json")).read_text())


def _numbers(text):
    return [float(value) for value in text.split()]


def _s(values):
    return " ".join(format(float(value), ".12g") for value in values)


def _rpy_quat(rpy):
    r, p, y = [value / 2 for value in rpy]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]


def _matrix_quat(matrix):
    import numpy as np
    # Symmetric eigenvalue method remains well-conditioned near 180 degrees.
    m = np.asarray(matrix, dtype=float)
    k = np.array([[m[0,0]-m[1,1]-m[2,2], m[0,1]+m[1,0], m[0,2]+m[2,0], m[2,1]-m[1,2]],
                  [m[0,1]+m[1,0], m[1,1]-m[0,0]-m[2,2], m[1,2]+m[2,1], m[0,2]-m[2,0]],
                  [m[0,2]+m[2,0], m[1,2]+m[2,1], m[2,2]-m[0,0]-m[1,1], m[1,0]-m[0,1]],
                  [m[2,1]-m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1], m.trace()]]) / 3
    _, v = np.linalg.eigh(k)
    q = v[:, -1][[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def resolve_mesh_dir(mesh_dir=None, verify=True):
    """Find the complete asset set. Installer should supply a portable directory."""
    candidates = [Path(mesh_dir)] if mesh_dir else []
    if os.environ.get("EXPLORER_SIM_MESH_DIR"):
        candidates.append(Path(os.environ["EXPLORER_SIM_MESH_DIR"]))
    candidates += [ROOT / "vendor/m3_reference/meshes", Path("/Volumes/ExplorerSim/assets/m3_reference/meshes")]
    hashes = json.loads((ROOT / "config/arm_model.json").read_text())["asset_sha256"]
    expected = {Path(name).name: digest for name, digest in hashes.items() if name.endswith(".STL")}
    for folder in candidates:
        if all((folder / name).is_file() for name in expected):
            if verify:
                for name, digest in expected.items():
                    if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                        raise ValueError(f"CAD asset checksum mismatch: {folder / name}")
            return folder.resolve()
    raise FileNotFoundError("Explorer CAD meshes missing. Set EXPLORER_SIM_MESH_DIR to verified meshes directory.")


def servo_to_joint(servo_deg):
    """Six endpoint degrees -> five CAD angles plus uncalibrated linkage angle."""
    import numpy as np
    deg = np.asarray(servo_deg, dtype=float)
    if deg.shape != (6,) or not np.isfinite(deg).all():
        raise ValueError("Six finite servo endpoint degrees required")
    lower = np.array([0, 0, 0, 0, 0, 30.])
    upper = np.array([180, 180, 180, 180, 270, 170.])
    if np.any(deg < lower) or np.any(deg > upper):
        raise ValueError("Servo endpoints exceed model reference limits")
    q = np.deg2rad(deg - 90)
    q[0] = -q[0]
    q[5] = model_config()["gripper"]["linkage_closed_rad"] * (deg[5] - 30) / 140
    return q


def joint_to_servo(joint_rad):
    import numpy as np
    q = np.asarray(joint_rad, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("Six finite joint angles required")
    result = np.rad2deg(q) + 90
    result[0] = 90 - math.degrees(q[0])
    result[5] = 30 + q[5] * 140 / model_config()["gripper"]["linkage_closed_rad"]
    return result


def wheel_targets(vx, vy, yaw_rate):
    """Body twist -> physical hinge velocities, FL FR RL RR (rad/s).

    Wheel axes follow each URDF body frame; right hinge signs are reversed.
    Standard X roller layout, with positive body y to the left.
    """
    radius = model_config()["physics"]["wheel_radius_m"]
    lever = 0.0795 + (0.09775 - 0.0108533)
    return [-(vx-vy-lever*yaw_rate)/radius, (vx+vy+lever*yaw_rate)/radius,
            -(vx+vy-lever*yaw_rate)/radius, (vx-vy+lever*yaw_rate)/radius]


def _add_contact(body, name, **attrs):
    defaults = dict(name=name, contype="1", conaffinity="7", group="3", rgba="0.4 0.5 0.7 0", friction="0.8 0.005 0.0001", solref="0.008 1", margin="0.0003")
    defaults.update({key: str(value) for key, value in attrs.items()})
    return ET.SubElement(body, "geom", defaults)


def _sock(world, config, position):
    """Open cuff and sewn toe; triangulated cloth tube, fully free vertices."""
    c = config["sock"]
    points, faces = [], []
    n, rings = c["ring_vertices"], c["rings"]
    for i in range(rings):
        x = c["length_m"] * (i / (rings - 1) - .5)
        taper = .65 + .35 * i / (rings - 1)
        for j in range(n):
            theta = 2 * math.pi * j / n
            points.extend([x, c["width_m"] / 2 * taper * math.cos(theta), c["initial_thickness_m"] / 2 * math.sin(theta)])
    points.extend([-c["length_m"] / 2 - .008, 0, 0])
    toe = rings * n
    for i in range(rings - 1):
        for j in range(n):
            a, b = i*n+j, i*n+(j+1)%n
            d, e = (i+1)*n+j, (i+1)*n+(j+1)%n
            faces.extend([a, b, d, b, e, d])
    for j in range(n):
        faces.extend([toe, (j+1)%n, j])
    cloth = ET.SubElement(world, "flexcomp", name="sock", type="direct", dim="2", point=_s(points), element=" ".join(str(i) for i in faces), pos=_s(position), mass=str(c["mass_kg"]), radius=str(c["contact_radius_m"]), rgba="0.82 0.27 0.23 1")
    ET.SubElement(cloth, "edge", equality="true", damping="0.001", solref="0.01 1")
    ET.SubElement(cloth, "contact", contype="4", conaffinity="7", selfcollide="auto", internal="true", condim="3", friction=f'{c["friction"]} 0.005 0.0001', solref="0.008 1")
    ET.SubElement(cloth, "elasticity", young=str(c["young_pa"]), poisson="0.3", thickness=str(c["thickness_m"]), elastic2d="bend", damping="0.001")


def build_mjcf(mesh_dir=None, *, scene="sock", sock_position=(.33, 0., .015), bin_position=(.60, .25, 0.), camera_intrinsics=None, verify_assets=True):
    """Return self-contained MJCF XML with absolute, checksum-verified STL paths.

    scene='empty' excludes cloth/bin; 'sock' includes both; 'navigation' also
    adds obstacles. Intrinsics override must contain width,height,fx,fy,cx,cy.
    """
    if scene not in ("empty", "sock", "navigation", "cookie", "cube", "cloth"):
        raise ValueError("Unknown simulator scene")
    import numpy as np
    config = model_config()
    camera = dict(config["camera"])
    if camera_intrinsics:
        camera.update(camera_intrinsics)
    folder = resolve_mesh_dir(mesh_dir, verify=verify_assets)
    urdf = ET.parse(ROOT / "config/explorer.urdf").getroot()
    links = {item.get("name"): item for item in urdf.findall("link")}
    joints = urdf.findall("joint")
    xml = ET.Element("mujoco", model="Explorer_CAD_roller_cloth")
    ET.SubElement(xml, "compiler", angle="radian", inertiafromgeom="false", autolimits="true", balanceinertia="true")
    ET.SubElement(xml, "option", timestep=str(config["physics"]["timestep_s"]), gravity="0 0 -9.81", integrator="discrete", cone="elliptic", iterations="70", tolerance="1e-8")
    ET.SubElement(xml, "size", memory="192M")
    visual = ET.SubElement(xml, "visual")
    ET.SubElement(visual, "global", offwidth=str(camera["width"]), offheight=str(camera["height"]))
    ET.SubElement(visual, "map", znear="0.005", zfar="8")
    asset = ET.SubElement(xml, "asset")
    world = ET.SubElement(xml, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 .1", rgba="0.18 0.22 0.27 1", contype="2", conaffinity="5", friction=f'{config["physics"]["floor_friction"]} .005 .0001')
    ET.SubElement(world, "camera", name="overview", pos=".90 -.85 .60", xyaxes=".813 .582 0 -.209 .292 .933", fovy="45")
    equality = ET.SubElement(xml, "equality")
    contact = ET.SubElement(xml, "contact")
    for rule in ET.parse(ROOT / "config/explorer.srdf").getroot().findall("disable_collisions"):
        a, b = rule.get("link1"), rule.get("link2")
        if a != "base_footprint" and b != "base_footprint":
            ET.SubElement(contact, "exclude", body1=a, body2=b)
    actuators = ET.SubElement(xml, "actuator")
    bodies = {"base_link": ET.SubElement(world, "body", name="base_link", pos="0 0 .002")}
    ET.SubElement(bodies["base_link"], "freejoint", name="base_free")
    # Topological tree construction preserves URDF origins and local hinge axes.
    pending = [j for j in joints if j.find("child").get("link") != "base_link"]
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in bodies]
        if not ready:
            raise ValueError("URDF has disconnected links")
        for joint in ready:
            name = joint.find("child").get("link")
            origin = joint.find("origin")
            body = ET.SubElement(bodies[joint.find("parent").get("link")], "body", name=name, pos=origin.get("xyz", "0 0 0"), quat=_s(_rpy_quat(_numbers(origin.get("rpy", "0 0 0")))))
            bodies[name] = body
            if name in WHEEL_NAMES:
                ET.SubElement(body, "joint", name=joint.get("name"), type="hinge", axis="0 0 1", damping="0.001", armature="0.0001")
            elif joint.get("type") != "fixed":
                attrs = dict(name=joint.get("name"), type="hinge", axis=joint.find("axis").get("xyz"), damping="0.025", armature="0.00015")
                limit = joint.find("limit")
                if limit is not None:
                    attrs["range"] = f'{limit.get("lower")} {limit.get("upper")}'
                ET.SubElement(body, "joint", attrs)
                mimic = joint.find("mimic")
                if mimic is not None:
                    ET.SubElement(equality, "joint", joint1=joint.get("name"), joint2=mimic.get("joint"), polycoef=f'{mimic.get("offset", "0")} {mimic.get("multiplier", "1")} 0 0 0', solref="0.004 1")
            pending.remove(joint)
    for name, body in bodies.items():
        link = links[name]
        inertia = link.find("inertial")
        if inertia is not None and name not in ("imu_frame", "laser0_frame", "laser1_frame"):
            tensor = inertia.find("inertia")
            mass = float(inertia.find("mass").get("value"))
            if name in WHEEL_NAMES:
                mass -= config["physics"]["roller_count"] * config["physics"]["roller_mass_kg"]
            ET.SubElement(body, "inertial", pos=inertia.find("origin").get("xyz"), mass=str(mass), fullinertia=" ".join(tensor.get(key) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
        mesh = link.find("visual/geometry/mesh")
        if mesh is not None:
            filename = Path(mesh.get("filename")).name
            ET.SubElement(asset, "mesh", name=f"mesh_{name}", file=str(folder / filename), maxhullvert="32")
            rgba = link.find("visual/material/color").get("rgba")
            ET.SubElement(body, "geom", name=f"visual_{name}", type="mesh", mesh=f"mesh_{name}", contype="0", conaffinity="0", group="2", rgba=rgba)
    payload = ET.SubElement(bodies["base_link"], "body", name="estimated_electronics_payload", pos="-.03 0 .12")
    ET.SubElement(payload, "inertial", pos="0 0 0", mass=str(config["physics"]["estimated_payload_mass_kg"]), diaginertia=".002 .003 .003")
    _add_contact(bodies["base_link"], "chassis_contact", type="box", pos="-.014 0 .105", size=".12 .072 .055")
    _add_contact(bodies["arm1"], "arm1_contact", type="cylinder", pos="0 0 .011", size=".025 .017")
    for i, end in ((2,.07), (3,.07), (4,.063)):
        _add_contact(bodies[f"arm{i}"], f"arm{i}_contact", type="capsule", fromto=f"0 0 .008 0 0 {end}", size=".012")
    _add_contact(bodies["arm5"], "arm5_contact", type="box", pos=".015 0 .018", size=".024 .025 .018")
    _add_contact(bodies["DCW2"], "wrist_camera_contact", type="box", pos="-.009 -.002 -.010", size=".022 .044 .022")
    # The flat CAD Gripping part is a reference plate, not a solid jaw cavity.
    for name in ("rlink1", "llink1", "rlink3", "llink3"):
        _add_contact(bodies[name], f"{name}_contact", type="capsule", fromto="0 0 -.003 .03 0 -.003", size=".003", conaffinity="6")
    for name in ("rlink2", "llink2"):
        sign = 1 if name == "rlink2" else -1
        if scene in ('cookie','cube'):
            _add_contact(bodies[name], ('finger_right' if name=='rlink2' else 'finger_left'),
                         type='box',pos=f'.034 {sign*.006} -.005',size='.016 .006 .004',
                         friction='1.2 .01 .0001',conaffinity='6')
        else:
            _add_contact(bodies[name], ("finger_right" if name == "rlink2" else "finger_left"), type="capsule", fromto=f".02 {sign*.006} -.005 .048 {sign*.006} -.005", size=".004", friction="1.2 .01 .0001", conaffinity="6")
    # Explicit opposing-pad contact prevents commanded overclosure from
    # passing through the other finger; servo torque produces finite squeeze.
    ET.SubElement(contact, "pair", name="opposing_finger_contact", geom1="finger_left", geom2="finger_right", condim="3", friction=".4 .4 .005 .0001 .0001", solref="0.008 1")
    ET.SubElement(bodies["arm5"], "site", name="grasp_center", pos="-.00075 -.00011 .088", size=".004", rgba=".1 1 .2 .5")
    # CAD bounding radius 40 mm, 12 independently rotating 45-degree rollers.
    # Roller centers and shape are a deliberate estimated contact approximation.
    p = config["physics"]
    for index, name in enumerate(WHEEL_NAMES):
        body = bodies[name]
        _add_contact(body, f"{name}_hub", type="cylinder", pos="0 0 .0108533", size=".024 .01", contype="8", conaffinity="6")
        # Global roller axis at bottom is X +/- Y; X-layout diagonal pairs.
        handed = 1 if index in (0, 3) else -1
        for j in range(p["roller_count"]):
            theta = 2 * math.pi * j / p["roller_count"]
            center = [.032*math.cos(theta), .032*math.sin(theta), .0108533]
            axis = [-math.sin(theta)/math.sqrt(2), math.cos(theta)/math.sqrt(2), handed/math.sqrt(2)]
            roller = ET.SubElement(body, "body", name=f"{name}_roller{j}", pos=_s(center))
            ET.SubElement(roller, "joint", name=f"{name}_roller{j}_hinge", type="hinge", axis=_s(axis), damping=str(p["roller_damping"]))
            m, r, length = p["roller_mass_kg"], p["roller_radius_m"], 2*p["roller_half_length_m"]
            ET.SubElement(roller, "inertial", pos="0 0 0", mass=str(m), diaginertia=_s([m*(3*r*r+length*length)/12]*3))
            h = p["roller_half_length_m"]
            _add_contact(roller, f"{name}_roller{j}_contact", type="capsule", fromto=_s([-h*a for a in axis]+[h*a for a in axis]), size=str(r), contype="8", conaffinity="6", rgba=".15 .15 .17 1")
        ET.SubElement(actuators, "velocity", name=f"wheel{index}", joint=f"{name}_Joint", kv=str(p["wheel_kv"]), ctrlrange="-20 20", forcerange=f'-{p["wheel_torque_nm"]} {p["wheel_torque_nm"]}')
    limits = [(-math.pi/2, math.pi/2)]*4 + [(-math.pi/2, math.pi), (config["gripper"]["linkage_closed_rad"], 0)]
    for i, joint in enumerate(ARM_JOINTS):
        torque = p["servo_torque_nm"][i]
        ET.SubElement(actuators, "position", name=f"servo{i+1}", joint=joint, kp=str(p["servo_kp"]), kv=str(p["servo_kv"]), ctrlrange=_s(limits[i]), forcerange=f"-{torque} {torque}")
    handeye = json.loads((ROOT / "config/handeye-accepted.json").read_text())
    transform = np.asarray(handeye["camera_to_mount_reference"], dtype=float)
    mount = bodies[handeye["reference_mount"]]
    # MuJoCo cameras look along -Z with +Y up; ROS optical is +Z, +Y down.
    cam_quat = _matrix_quat(transform[:3,:3] @ np.diag([1., -1., -1.]))
    for camera_name in ("wrist", "wrist_depth"):
        ET.SubElement(mount, "camera", name=camera_name, pos=_s(transform[:3,3]), quat=_s(cam_quat), sensorsize=".0064 .0048", focalpixel=_s([camera["fx"], camera["fy"]]), principalpixel=_s([camera["cx"]-(camera["width"]-1)/2, (camera["height"]-1)/2-camera["cy"]]), resolution=f'{camera["width"]} {camera["height"]}')
    ET.SubElement(mount, "site", name="camera_color_optical_frame", pos=_s(transform[:3,3]), quat=_s(_matrix_quat(transform[:3,:3])), size=".003")
    for name in ("laser0_frame", "laser1_frame", "imu_frame"):
        ET.SubElement(bodies[name], "site", name=name, size=".003", rgba=".1 .8 1 1")
    if scene != "empty":
        if scene in ('cookie', 'cube'):
            size = [.028, .019, .018] if scene == 'cookie' else [.018]*3
            item = ET.SubElement(world, 'body', name='task_object', pos=_s(sock_position))
            ET.SubElement(item, 'freejoint', name='task_object_free')
            ET.SubElement(item, 'inertial', pos='0 0 0', mass='.025', diaginertia='0.000007 0.000007 0.000007')
            ET.SubElement(item, 'geom', name='task_object_contact', type='box', size=_s(size),
                          contype='4', conaffinity='7', friction='.8 .005 .0001',
                          rgba='.85 .65 .22 1' if scene == 'cookie' else '.25 .5 .85 1')
        else:
            if scene == 'cloth':
                config['sock'].update(length_m=.12, width_m=.045, mass_kg=.018)
            _sock(world, config, sock_position)
        c = config["bin"]
        hx, hy = c["inner_half_size_m"]
        if scene == 'cookie':
            hx = hy = .075
        thick, height = c["wall_thickness_m"], c["height_m"]
        if scene == 'cookie':
            height = .035  # A supported serving tray, not the laundry basket.
        backing = c.get("floor_collision_backing_m", 0.)
        bin_body = ET.SubElement(world, "body", name="bin", pos=_s(bin_position))
        # Preserve the visible top surface. Numerical backing below the ground
        # prevents a soft flex contact switching to the underside of a thin box.
        specs = [("bottom", [0,0,(thick-backing)/2], [hx+thick,hy+thick,(thick+backing)/2]),
                 ("left", [0,hy+thick/2,height/2], [hx+thick,thick/2,height/2]),
                 ("right", [0,-hy-thick/2,height/2], [hx+thick,thick/2,height/2]),
                 ("front", [hx+thick/2,0,height/2], [thick/2,hy,height/2]),
                 ("back", [-hx-thick/2,0,height/2], [thick/2,hy,height/2])]
        for name, pos, size in specs:
            ET.SubElement(bin_body, "geom", name=f"bin_{name}", type="box", pos=_s(pos), size=_s(size), contype="2", conaffinity="5", rgba=".15 .5 .55 1")
    if scene == "navigation":
        for i, (x,y) in enumerate(((1.,-.4), (-.6,.4), (.65,-.7))):
            ET.SubElement(world, "geom", name=f"obstacle_{i}", type="box", pos=f"{x} {y} .16", size=".1 .1 .16", contype="2", conaffinity="5", rgba=".7 .5 .3 1")
    if scene == 'cookie':
        # Two small rooms divided by a real collision wall and 70 cm doorway.
        for side in (-1, 1):
            ET.SubElement(world, 'geom', name=f'room_wall_{side}', type='box',
                          pos=f'.65 {side*.925} .25', size='.025 .575 .25',
                          contype='2', conaffinity='5', rgba='.55 .58 .62 1')
        for name,pos,size in (
                ('south',[.55,-.70,.25],[1.,.025,.25]),
                ('north',[.55,.80,.25],[1.,.025,.25]),
                ('west',[-.45,.05,.25],[.025,.75,.25]),
                ('east',[1.55,.05,.25],[.025,.75,.25])):
            ET.SubElement(world,'geom',name='room_boundary_'+name,type='box',pos=_s(pos),size=_s(size),
                          contype='2',conaffinity='5',rgba='.55 .58 .62 1')
    return ET.tostring(xml, encoding="unicode")


def load_model(**kwargs):
    """Compile CAD MJCF and return (mujoco.MjModel, mujoco.MjData)."""
    import mujoco
    model = mujoco.MjModel.from_xml_string(build_mjcf(**kwargs))
    data = mujoco.MjData(model)
    targets = servo_to_joint([90,90,90,90,90,30])
    for name, q in zip(ARM_JOINTS, targets):
        data.qpos[model.joint(name).qposadr[0]] = q
    for i, q in enumerate(targets):
        data.ctrl[model.actuator(f"servo{i+1}").id] = q
    mujoco.mj_forward(model, data)
    return model, data


def model_metadata():
    """Explicit provenance and unknowns for simulator status, data manifests."""
    config = model_config()
    config["arm_joints"] = list(ARM_JOINTS)
    config["wheel_joints"] = list(WHEEL_JOINTS)
    config["handeye"] = json.loads((ROOT / "config/handeye-accepted.json").read_text())
    config["lidar_geometry"] = json.loads((ROOT / "config/lidar_geometry.json").read_text())
    return config
