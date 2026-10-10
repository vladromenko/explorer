"""Demand-loaded official ONNX landmarks; passive results and explicit gesture proposals."""
import hashlib
import json
import math
import struct
from pathlib import Path
import threading
import time
import uuid

ZOO_REV = "47534e27c9851bb1128ccc0102f1145e27f23f98"
FACE_REV = "c97242ce7f2a554e288b50eabd9f5df957e78801"
MODEL_MANIFEST = {
    "palm": {"folder": "palm_detection_mediapipe", "file": "palm_detection_mediapipe_2023feb.onnx", "sha256": "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c", "size": 3905734, "license": "Apache-2.0", "revision": ZOO_REV},
    "hand": {"folder": "handpose_estimation_mediapipe", "file": "handpose_estimation_mediapipe_2023feb.onnx", "sha256": "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c", "size": 4099621, "license": "Apache-2.0", "revision": ZOO_REV},
    "person": {"folder": "person_detection_mediapipe", "file": "person_detection_mediapipe_2023mar.onnx", "sha256": "47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f", "size": 11990159, "license": "Apache-2.0", "revision": ZOO_REV},
    "pose": {"folder": "pose_estimation_mediapipe", "file": "pose_estimation_mediapipe_2023mar.onnx", "sha256": "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f", "size": 5557238, "license": "Apache-2.0", "revision": ZOO_REV},
    "face": {"folder": "face_detection_yunet", "file": "face_detection_yunet_2022mar.onnx", "sha256": "50ef07f702a31741ca46a4c0d947773b64143b9362780237bf0d427d6c79bab7", "size": 345478, "license": "MIT", "revision": FACE_REV},
}
MODEL_MANIFEST["face_v2"] = {"folder": "face_detection_yunet", "file": "face_detection_yunet_2023mar.onnx", "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", "size": 232589, "license": "MIT", "revision": ZOO_REV}
LICENSE_HASHES = {"palm": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30", "hand": "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd", "person": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30", "pose": "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd", "face": "c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958", "face_v2": "c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958"}
CV46_ARTIFACTS = {
    "hand": ("bdab34c88d32f2f70cfa880fd2eda4a98fb0040c1d8393624e6460dc70670f07", 37),
    "person": ("b11c68459aeba7bab58f44cb7d6ebe61086d8584bf1da5e2b3caf796bd5b6f66", 32),
    "pose": ("e22ced6555b9e9d66ca2ebb771307e43adff6458ea0045f38e745f4e40438661", 68),
}
ACTION_ALLOWLIST = frozenset(("look_left", "look_right", "look_up", "look_down", "open_gripper", "close_gripper", "wave", "nod", "trace_2d"))


def verify_model(directory, name):
    item = MODEL_MANIFEST[name]
    path = Path(directory) / item["file"]
    if not path.is_file():
        raise FileNotFoundError("Missing official model: " + item["file"])
    if path.stat().st_size != item["size"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("Official model checksum mismatch: " + item["file"])
    return path



def varint(buf,start):
 result,shift=0,0
 while True:
  b=buf[start];start+=1;result|=(b&127)<<shift
  if b<128:return result,start
  shift+=7

def encode_int(value):
 out=b""
 while value>=128:out+=bytes([(value&127)|128]);value>>=7
 return out+bytes([value])

def fields(buf):
 pos=0;out=[]
 while pos<len(buf):
  start=pos;tag,pos=varint(buf,pos);wire=tag&7;number=tag>>3
  if wire==0:value,pos=varint(buf,pos)
  elif wire==2:
   length,pos=varint(buf,pos);value=buf[pos:pos+length];pos+=length
  elif wire in (1,5):
   length=8 if wire==1 else 4;value=buf[pos:pos+length];pos+=length
  else:raise ValueError(wire)
  out.append((number,wire,value,buf[start:pos]))
 return out

def message(number,value):return encode_int(number*8+2)+encode_int(len(value))+value

def scalar(t):
 parts=fields(t)
 if any(x[0]==2 and x[2]!=1 for x in parts):return None
 for number,wire,value,raw in parts:
  if number==9 and len(value)==4:return struct.unpack("<f",value)[0]
  if number==4 and wire in (2,5) and len(value)==4:return struct.unpack("<f",value)[0]
 return None

def _cv46_bytes(data):
 mf=fields(data);graph=next(x[2] for x in mf if x[0]==7);gf=fields(graph);constants={}
 for number,wire,value,raw in gf:
  if number==5:
   tf=fields(value);name=next((x[2].decode() for x in tf if x[0]==8),None);val=scalar(value)
   if val is not None:constants[name]=val
   rawval=next((x[2] for x in tf if x[0]==9),None)
   dtype=next((x[2] for x in tf if x[0]==2),None)
   if dtype==7 and rawval is not None and len(rawval)<=64:constants[name]=list(struct.unpack("<"+"q"*(len(rawval)//8),rawval))
  if number==1:
   nf=fields(value);kind=next(x[2] for x in nf if x[0]==4)
   if kind==b"Constant":
    name=next(x[2].decode() for x in nf if x[0]==2)
    for af in [fields(x[2]) for x in nf if x[0]==5]:
     tensor=next((x[2] for x in af if x[0]==5),None)
     if tensor:constants[name]=scalar(tensor)
 new=[];count=0
 for number,wire,value,raw in gf:
  if number==1:
   nf=fields(value);kind=next(x[2] for x in nf if x[0]==4);inputs=[x[2].decode() for x in nf if x[0]==1]
   if kind==b"Clip" and len(inputs)==3:
    low,high=constants[inputs[1]],constants[inputs[2]]
    if (low, high) != (0., 6.): raise ValueError("Only verified ReLU6 constants can be lowered")
    replacement=b"";seen=0
    for n,w,v,r in nf:
     if n==1:seen+=1
     if n!=1 or seen==1:replacement+=r
    for name,limit in [("min",low),("max",high)]:
     attr=message(1,name.encode())+encode_int(2*8+5)+struct.pack("<f",limit)+encode_int(20*8)+encode_int(1)
     replacement+=message(5,attr)
    raw=message(1,replacement);count+=1
  if number==1 and kind==b"Squeeze" and len(inputs)==2:
   axes=constants[inputs[1]];replacement=b"";seen=0
   for n,w,v,r in nf:
    if n==1:seen+=1
    if n!=1 or seen==1:replacement+=r
   attr=message(1,b"axes")+b"".join(encode_int(8*8)+encode_int(axis) for axis in axes)+encode_int(20*8)+encode_int(7)
   replacement+=message(5,attr);raw=message(1,replacement);count+=1
  if number==1 and kind==b"Gemm":
   outputs=[x[2] for x in nf if x[0]==2];name=next(x[2] for x in nf if x[0]==3)
   attributes={next(x[2].decode() for x in fields(a[2]) if x[0]==1):next((x[2] for x in fields(a[2]) if x[0]==3),None) for a in nf if a[0]==5}
   if attributes != {"transA": 0, "transB": 0}: raise ValueError("Unexpected Gemm attributes")
   middle=outputs[0]+b"__explorer_matmul"
   one=message(1,inputs[0].encode())+message(1,inputs[1].encode())+message(2,middle)+message(3,name+b"_matmul")+message(4,b"MatMul")
   two=message(1,middle)+message(1,inputs[2].encode())+message(2,outputs[0])+message(3,name+b"_add")+message(4,b"Add")
   raw=message(1,one)+message(1,two);count+=1
  new.append(raw)
 ng=b"".join(new);out=b"".join(message(7,ng) if n==7 else raw for n,w,v,raw in mf)
 return out,count


def cv46_artifact(directory, name, create=False):
    """Pinned OpenCV 4.6 importer dialect; originals remain unchanged.

    Constant Clip/Squeeze inputs become old importer attributes; Gemm becomes
    mathematically identical MatMul+Add. This is not a generic ONNX opset upgrade.
    """
    source = verify_model(directory, name)
    if name not in CV46_ARTIFACTS:
        return source
    target = source.with_name(source.stem + ".cv46.onnx")
    digest, expected_count = CV46_ARTIFACTS[name]
    if create:
        data, count = _cv46_bytes(source.read_bytes())
        if count != expected_count or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Compatibility conversion differs from verified artifact")
        pending = target.with_suffix(".pending")
        try:
            pending.write_bytes(data)
            pending.replace(target)
        finally:
            pending.unlink(missing_ok=True)
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise ValueError("Missing/invalid OpenCV 4.6 compatibility artifact; run installer")
    return target


def anchors(grids):
    """Fixed-size SSD centers generated in stride order, never inferred from boxes."""
    return [[(x + .5) / side, (y + .5) / side]
            for side, repeats in grids for y in range(side) for x in range(side) for _ in range(repeats)]


def hand_gesture(points):
    """Conservative geometry on 21 model landmarks, with an ambiguous neutral state."""
    if len(points) != 21 or any(len(p) < 2 or not all(math.isfinite(v) for v in p[:2]) for p in points):
        return {"gesture": "neutral", "reason": "invalid_landmarks"}
    distance = lambda a, b: math.hypot(points[a][0] - points[b][0], points[a][1] - points[b][1])
    scale = distance(0, 9)
    if scale < 8:
        return {"gesture": "neutral", "reason": "hand_too_small"}
    extended = []
    for base, joint, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        a = [points[base][k] - points[joint][k] for k in (0, 1)]
        b = [points[tip][k] - points[joint][k] for k in (0, 1)]
        norm = math.hypot(*a) * math.hypot(*b)
        angle = math.degrees(math.acos(max(-1, min(1, sum(a[k] * b[k] for k in (0, 1)) / max(norm, 1e-9)))))
        extended.append(angle > 155 and distance(0, tip) > distance(0, joint) * 1.15)
    gesture = "open_palm" if all(extended) else "point" if extended == [True, False, False, False] else "fist" if not any(extended) else "neutral"
    if distance(4, 8) / scale < .25 and any(extended[1:]):
        gesture = "pinch"
    dx, dy = points[8][0] - points[0][0], points[8][1] - points[0][1]
    direction = ("right" if dx > 0 else "left") if abs(dx) > abs(dy) else ("down" if dy > 0 else "up")
    return {"gesture": gesture, "direction_image": direction if gesture == "point" else None,
            "index_tip_px": list(points[8][:2]), "fingers_extended": extended, "contact_measured": False}


class HumanPerception:
    def __init__(self, root, model_directory=None):
        self.root = Path(root)
        self.directory = Path(model_directory or self.root / "models" / "human-perception")
        self.lock = threading.RLock()
        self.networks = {}
        self.loaded_model_evidence = {}
        self.cv = None
        self.np = None

    def _libraries(self):
        if self.cv is None:
            import cv2
            import numpy
            self.cv, self.np = cv2, numpy
        return self.cv, self.np

    def _net(self, name):
        if name not in self.networks:
            cv, _ = self._libraries()
            version = tuple(int(piece) for piece in cv.__version__.split(".")[:2])
            selected = "face_v2" if name == "face" and version >= (4, 8) else name
            path = verify_model(self.directory, selected)
            if version < (4, 8) and name in CV46_ARTIFACTS:
                path = cv46_artifact(self.directory, name, create=False)
            path = str(path)
            if name == "face":
                net = cv.FaceDetectorYN.create(path, "", (320, 320), .8, .3, 20, cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU)
            else:
                net = cv.dnn.readNetFromONNX(path)
                net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            self.networks[name] = net
            self.loaded_model_evidence[name] = {"source_sha256": MODEL_MANIFEST[selected]["sha256"],
                "runtime_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "runtime_filename": Path(path).name,
                "format": "opencv46_importer_dialect" if path.endswith(".cv46.onnx") else "official_onnx"}
        return self.networks[name]

    def unload(self):
        with self.lock:
            self.networks.clear()
            self.loaded_model_evidence.clear()

    def _forward(self, name, blob):
        net = self._net(name)
        net.setInput(blob)
        return net.forward(net.getUnconnectedOutLayersNames())

    def _letterbox(self, image, size):
        cv, np = self._libraries()
        h, w = image.shape[:2]
        scale = min(size / w, size / h)
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        left, top = (size - rw) // 2, (size - rh) // 2
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[top:top + rh, left:left + rw] = cv.resize(image, (rw, rh))
        return canvas, scale, np.array([left, top], dtype=np.float32)

    def _detect(self, image, name):
        cv, np = self._libraries()
        size, grid, count = (192, ((24, 2), (12, 6)), 7) if name == "palm" else (224, ((28, 2), (14, 2), (7, 6)), 4)
        canvas, scale, offset = self._letterbox(image, size)
        tensor = cv.cvtColor(canvas, cv.COLOR_BGR2RGB).astype(np.float32) / 255
        if name == "person":
            tensor = (tensor - .5) * 2
            # MediaPipe padding is zero in the normalized tensor.
            left, top = offset.astype(int)
            rw, rh = round(image.shape[1] * scale), round(image.shape[0] * scale)
            pad = np.zeros_like(tensor)
            pad[top:top + rh, left:left + rw] = tensor[top:top + rh, left:left + rw]
            tensor = pad.transpose(2, 0, 1)
        outputs = self._forward(name, tensor[None])
        locations, logits = outputs[0].reshape(-1, 4 + count * 2), outputs[1].reshape(-1)
        centers = np.asarray(anchors(grid), dtype=np.float32)
        if len(locations) != len(centers) or len(logits) != len(centers):
            raise ValueError("Unexpected official detector tensor layout")
        scores = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
        center = locations[:, :2] + centers * size
        half = locations[:, 2:4] / 2
        corners = np.c_[(center - half - offset) / scale, (center + half - offset) / scale]
        xywh = np.c_[corners[:, :2], corners[:, 2:4] - corners[:, :2]]
        indices = np.asarray(cv.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), .7, .3, top_k=20)).reshape(-1)
        result = []
        for index in indices[:2]:
            keypoints = (locations[index, 4:].reshape(count, 2) + centers[index] * size - offset) / scale
            result.append({"bbox": corners[index], "keypoints": keypoints, "confidence": float(scores[index])})
        return result

    def _crop(self, image, center, upward, side, size):
        cv, np = self._libraries()
        upward = np.asarray(upward, dtype=np.float32)
        length = np.linalg.norm(upward)
        if not math.isfinite(float(side)) or side < 8 or side > 4 * max(image.shape[:2]) or length < 1e-5:
            raise ValueError("Unusable model-detected oriented ROI")
        up = upward / length
        right = np.array([-up[1], up[0]])
        center = np.asarray(center, dtype=np.float32)
        source = np.array([center - right * side / 2 + up * side / 2,
                           center + right * side / 2 + up * side / 2,
                           center - right * side / 2 - up * side / 2], dtype=np.float32)
        destination = np.array([[0, 0], [size, 0], [0, size]], dtype=np.float32)
        affine = cv.getAffineTransform(source, destination)
        crop = cv.warpAffine(image, affine, (size, size), borderMode=cv.BORDER_CONSTANT)
        tensor = cv.cvtColor(crop, cv.COLOR_BGR2RGB).astype(np.float32)[None] / 255
        return tensor, cv.invertAffineTransform(affine), side / size

    def _landmarks(self, image, detection, name):
        _, np = self._libraries()
        keys = detection["keypoints"]
        if name == "hand":
            up = keys[2] - keys[0]
            width = detection["bbox"][2:] - detection["bbox"][:2]
            center = (detection["bbox"][:2] + detection["bbox"][2:]) / 2
            center = center + up / max(np.linalg.norm(up), 1e-9) * width[1] * .4
            tensor, inverse, zscale = self._crop(image, center, up, max(width) * 3, 224)
            outputs = self._forward(name, tensor)
            if len(outputs) != 4 or outputs[0].size != 63 or outputs[1].size != 1:
                raise ValueError("Unexpected official hand landmark tensor layout")
            confidence = float(outputs[1].reshape(-1)[0])
            points = outputs[0].reshape(21, 3).astype(float)
        else:
            radius = np.linalg.norm(keys[1] - keys[0])
            tensor, inverse, zscale = self._crop(image, keys[0], keys[1] - keys[0], radius * 2, 256)
            outputs = self._forward(name, tensor)
            if len(outputs) != 5 or outputs[0].size != 195 or outputs[1].size != 1:
                raise ValueError("Unexpected official body landmark tensor layout")
            confidence = float(outputs[1].reshape(-1)[0])
            points = outputs[0].reshape(39, 5)[:33].astype(float)
            points[:, 3:] = 1 / (1 + np.exp(-np.clip(points[:, 3:], -60, 60)))
        if not math.isfinite(confidence) or confidence < .8 or not np.isfinite(points).all():
            return None
        points[:, :2] = np.c_[points[:, :2], np.ones(len(points))] @ inverse.T
        points[:, 2] *= zscale
        row = {"landmarks": points.tolist(), "confidence": confidence, "coordinate_frame": "image_pixels",
               "z_kind": "model_relative_not_measured_depth", "metric_pose_measured": False,
               "landmark_visibility_available": name == "pose"}
        if name == "hand":
            row.update(hand_gesture(row["landmarks"]))
            row["handedness_model_probability"] = float(outputs[2].reshape(-1)[0])
        return row

    def _faces(self, image):
        cv, np = self._libraries()
        if float(np.std(cv.cvtColor(image, cv.COLOR_BGR2GRAY))) < 2:
            raise ValueError("Face landmarks unavailable on an image with insufficient contrast")
        canvas, scale, offset = self._letterbox(image, 320)
        _, rows = self._net("face").detect(canvas)
        result = []
        if rows is not None:
            for row in rows[:2]:
                result.append({"bbox": np.r_[(row[:2] - offset) / scale, (row[:2] + row[2:4] - offset) / scale].tolist(), "bbox_format": "xyxy",
                    "landmarks": ((row[4:14].reshape(5, 2) - offset) / scale).tolist(),
                    "landmark_names": ["right_eye", "left_eye", "nose", "right_mouth", "left_mouth"],
                    "confidence": float(row[-1]), "face_mesh_available": False, "identity_recognition": False})
        return result

    def analyze(self, image, stamp, modes=("hand",), frame_id="", now=None):
        modes = tuple(modes)
        if not modes or len(set(modes)) != len(modes) or set(modes) - {"hand", "pose", "face"}:
            raise ValueError("Modes must be distinct hand/pose/face")
        checked_at = time.time() if now is None else now
        if type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= checked_at - stamp < .7:
            raise ValueError("Human-perception frame is stale")
        if len(image.shape) != 3 or image.shape[2] != 3 or max(image.shape[:2]) > 1280:
            raise ValueError("Expected bounded BGR camera frame")
        started = time.monotonic()
        result = {"frame_id": frame_id, "image_stamp": stamp, "image_size": [image.shape[1], image.shape[0]],
                  "motion_authorized": False, "mode_results": {}, "errors": {}, "source": "official_onnx_cpu",
                  "loaded_models": [], "model_revision": ZOO_REV}
        with self.lock:
            for mode in modes:
                try:
                    if mode == "face":
                        rows = self._faces(image)
                    else:
                        detections = self._detect(image, "palm" if mode == "hand" else "person")
                        rows = [row for detection in detections if (row := self._landmarks(image, detection, mode)) is not None]
                    result["mode_results"][mode] = rows
                except (OSError, ValueError, RuntimeError, ImportError) as exc:
                    result["errors"][mode] = str(exc)
                except Exception as exc:
                    # cv2.error is unavailable until OpenCV is imported; preserve exact failure.
                    result["errors"][mode] = type(exc).__name__ + ": " + str(exc)
            result["loaded_models"] = list(self.networks)
            result["runtime_models"] = dict(self.loaded_model_evidence)
        elapsed = time.monotonic() - started
        result.update(latency_s=elapsed, finished_at=checked_at + elapsed, stale=checked_at + elapsed - stamp >= .7)
        return result

    def prepare(self, modes=("hand",)):
        modes=tuple(modes)
        if not modes or len(set(modes))!=len(modes) or set(modes)-{"hand","pose","face"}:
            raise ValueError("Modes must be distinct hand/pose/face")
        with self.lock:
            for mode in modes:
                names=("face",) if mode=="face" else ("palm","hand") if mode=="hand" else ("person","pose")
                for name in names:self._net(name)
        return {"loaded_models":list(self.networks),"executed":False}

    def observe(self, modes=("hand",)):
        # Load models BEFORE acquiring the exposure; cold inference must not
        # accidentally consume the pre-initialization stale camera frame.
        self.prepare(modes)
        cv, np = self._libraries()
        metadata = json.loads((self.root / "data" / "camera-frame.json").read_text())
        jpeg = (self.root / "data" / "frame-raw.jpg").read_bytes()
        if hashlib.sha256(jpeg).hexdigest() != metadata.get("jpeg_sha256"):
            raise ValueError("Cached camera frame changed during reading; retry")
        image = cv.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid cached camera JPEG")
        result=self.analyze(image, metadata["image_stamp"], modes, metadata["frame_id"])
        with self.lock:self.latest=(metadata["frame_id"],jpeg,metadata)
        result["image_endpoint"]="/api/vision/humans/frame/"+metadata["frame_id"]
        result["image_sha256"]=metadata["jpeg_sha256"]
        return result

    def compatibility(self):
        """Offline synthetic forward only: proves runtime support, not detection accuracy."""
        cv, np = self._libraries()
        report = {"opencv": cv.__version__, "backend": "opencv_cpu", "components": {}}
        with self.lock:
            for name, shape in (("palm", (1, 192, 192, 3)), ("hand", (1, 224, 224, 3)), ("person", (1, 3, 224, 224)), ("pose", (1, 256, 256, 3)), ("face", (320, 320, 3))):
                try:
                    outputs = self._net(name).detect(np.zeros(shape, np.uint8)) if name == "face" else self._forward(name, np.zeros(shape, np.float32))
                    report["components"][name] = {"available": True, "output_shapes": [None if output is None else list(getattr(output, "shape", ())) for output in outputs]}
                except Exception as exc:
                    report["components"][name] = {"available": False, "reason": str(exc)}
        report["runtime_models"] = dict(self.loaded_model_evidence)
        return report


class GestureGate:
    """Trusted caller issues a bounded grant; output is a proposal, never actuator I/O."""
    def __init__(self):
        self.context = None
        self.candidate = None
        self.last_gesture = None
        self.last_stamp = None
        self.stroke = []

    def begin(self, session_id, mapping, operator_region, expires_s=20, physical_authorized=False, now=None):
        if not isinstance(session_id, str) or not 8 <= len(session_id) <= 100:
            raise ValueError("Explicit operator session required")
        if not isinstance(mapping, dict) or not mapping or set(mapping) - {"open_palm", "fist", "pinch", "point"} or set(mapping.values()) - ACTION_ALLOWLIST:
            raise ValueError("Only caller-selected allowlisted gesture groups")
        if len(operator_region) != 4 or not all(type(v) in (int, float) and math.isfinite(v) for v in operator_region) or operator_region[2] <= operator_region[0] or operator_region[3] <= operator_region[1]:
            raise ValueError("Select the operator hand region explicitly")
        if type(expires_s) not in (int, float) or not 1 <= expires_s <= 60 or type(physical_authorized) is not bool:
            raise ValueError("Grant duration 1..60 seconds and explicit physical authorization")
        self.context = {"session_id": session_id, "mapping": dict(mapping), "region": list(operator_region),
                        "expires_at": (time.time() if now is None else now) + expires_s, "physical_authorized": physical_authorized}
        self.candidate = self.last_gesture = self.last_stamp = None
        self.stroke = []
        return dict(self.context)

    def cancel(self):
        self.context = None
        self.candidate = self.last_gesture = self.last_stamp = None
        self.stroke = []
        return {"state": "cancelled", "motion_authorized": False}

    def update(self, observation, now=None):
        if not isinstance(observation, dict):
            raise ValueError("Expected landmark observation")
        checked_at = time.time() if now is None else now
        stamp = observation.get("image_stamp", 0)
        base = {"state": "neutral", "motion_authorized": False, "proposal": None, "drawing_2d": [], "frame_id": observation.get("frame_id")}
        if self.context is None or checked_at >= self.context["expires_at"]:
            self.cancel()
            return dict(base, reason="explicit_context_missing_or_expired")
        if type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= checked_at - stamp < .7 or observation.get("stale"):
            self.candidate = self.last_gesture = None
            return dict(base, reason="stale_landmarks")
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return dict(base, reason="duplicate_or_reordered_frame")
        if self.last_stamp is not None and stamp - self.last_stamp > .7:
            self.candidate = self.last_gesture = None
        self.last_stamp = stamp
        region = self.context["region"]
        hands = observation.get("mode_results", {}).get("hand", [])
        def valid_hand(hand):
            points = hand.get("landmarks", [])
            return (type(hand.get("confidence")) in (int, float) and math.isfinite(hand["confidence"]) and hand["confidence"] >= .8
                    and isinstance(points, list) and len(points) == 21
                    and all(isinstance(point, (list, tuple)) and len(point) >= 2
                            and all(type(value) in (int, float) and math.isfinite(value) for value in point[:2]) for point in points))
        owned = [hand for hand in hands if isinstance(hand, dict) and valid_hand(hand)
                 and region[0] <= hand["landmarks"][0][0] <= region[2] and region[1] <= hand["landmarks"][0][1] <= region[3]]
        if len(owned) != 1:
            self.candidate = self.last_gesture = None
            return dict(base, reason="operator_hand_absent_or_ambiguous")
        hand = owned[0]
        gesture = hand.get("gesture", "neutral")
        action = self.context["mapping"].get(gesture)
        if action is None:
            self.candidate = self.last_gesture = None
            self.stroke = []
            return dict(base, reason="neutral_or_unmapped_gesture")
        if action == "trace_2d":
            size = observation.get("image_size", [])
            tip = hand.get("index_tip_px", [])
            if len(size) != 2 or min(size) <= 0 or len(tip) != 2 or not all(math.isfinite(v) for v in tip):
                return dict(base, reason="drawing_coordinates_unavailable")
            if not 0 <= tip[0] < size[0] or not 0 <= tip[1] < size[1]:
                self.stroke = []
                return dict(base, reason="drawing_tip_outside_image")
            self.stroke.append([tip[0] / size[0], tip[1] / size[1]])
            self.stroke = self.stroke[-256:]
            return dict(base, state="drawing", drawing_2d=list(self.stroke), reason="image_trace_only_not_robot_trajectory")
        if self.candidate is None or self.candidate[0] != gesture:
            self.candidate = (gesture, stamp)
            return dict(base, reason="gesture_settling")
        if stamp - self.candidate[1] < .25 or gesture == self.last_gesture:
            return dict(base, reason="settling_or_already_proposed")
        self.last_gesture = gesture
        proposal = {"id": uuid.uuid4().hex, "session_id": self.context["session_id"], "action": action,
                    "gesture": gesture, "direction_image": hand.get("direction_image"), "image_stamp": stamp,
                    "expires_at": min(self.context["expires_at"], checked_at + .5), "requires_executor_validation": True}
        return dict(base, state="proposal", proposal=proposal, motion_authorized=self.context["physical_authorized"])
