#!/usr/bin/env python3
"""Install pinned official weights only; no pip/system changes or camera/actuator access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from human_perception import HumanPerception, MODEL_MANIFEST, LICENSE_HASHES, CV46_ARTIFACTS, cv46_artifact, verify_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "models" / "human-perception")
    parser.add_argument("--verify", action="store_true", help="Verify installed model hashes without network")
    parser.add_argument("--check-runtime", action="store_true", help="Offline synthetic CPU inference; no camera is opened")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, item in MODEL_MANIFEST.items():
        base = "https://raw.githubusercontent.com/opencv/opencv_zoo/" + item["revision"] + "/models/" + item["folder"] + "/"
        download = "https://media.githubusercontent.com/media/opencv/opencv_zoo/" + item["revision"] + "/models/" + item["folder"] + "/" + item["file"]
        destination = args.directory / item["file"]
        if not args.verify and not destination.is_file():
            pending = destination.with_suffix(".download")
            try:
                with urllib.request.urlopen(download, timeout=60) as response, pending.open("wb") as stream:
                    total = 0
                    while block := response.read(1024 * 1024):
                        total += len(block)
                        if total > item["size"]:
                            raise ValueError("Download exceeds pinned size")
                        stream.write(block)
                if total != item["size"] or hashlib.sha256(pending.read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError("Download hash/size does not match official Git LFS pointer")
                pending.replace(destination)
            finally:
                pending.unlink(missing_ok=True)
        verify_model(args.directory, name)
        license_path = args.directory / (name + "-LICENSE.txt")
        if not args.verify:
            with urllib.request.urlopen(base + "LICENSE", timeout=30) as response:
                text = response.read(65536)
            license_path.write_bytes(text)
        if not license_path.is_file() or hashlib.sha256(license_path.read_bytes()).hexdigest() != LICENSE_HASHES[name]:
            raise ValueError("Missing upstream license: " + str(license_path))
        compatible = cv46_artifact(args.directory, name, create=not args.verify) if name in CV46_ARTIFACTS else destination
        manifest[name] = dict(item, compatibility_file=compatible.name, compatibility_sha256=hashlib.sha256(compatible.read_bytes()).hexdigest(), model_url=download, license_url=base + "LICENSE", license_sha256=hashlib.sha256(license_path.read_bytes()).hexdigest())
        print(name + ": verified " + item["sha256"])
    if not args.verify:
        (args.directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.check_runtime:
        report = HumanPerception(ROOT, args.directory).compatibility()
        print(json.dumps(report, indent=2))
        if not all(row["available"] for row in report["components"].values()):
            return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print("Unavailable: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
