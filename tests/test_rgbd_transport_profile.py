"""Transport compatibility and real launcher environment selection."""
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).parents[1]
NS = {"dds": "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"}


def test_local_image_profile_preserves_udp_and_sizes_shared_memory_for_history():
    rgbd = ET.parse(ROOT / "config" / "fastdds-rgbd.xml").getroot()
    original = ET.parse(ROOT / "config" / "fastdds.xml").getroot()
    descriptors = {item.findtext("dds:transport_id", namespaces=NS): item
        for item in rgbd.findall("dds:transport_descriptors/dds:transport_descriptor", NS)}
    udp = descriptors["explorer_udp"]
    existing_udp = original.find("dds:transport_descriptors/dds:transport_descriptor", NS)
    assert [(item.tag, item.text.strip()) for item in udp] == [
        (item.tag, item.text.strip()) for item in existing_udp]
    shared = descriptors["explorer_rgbd_shm"]
    assert shared.findtext("dds:type", namespaces=NS) == "SHM"
    # History can contain ten images of each stream, not merely one packet.
    message_pair = 640 * 480 * (3 + 2) + 4096
    assert int(shared.findtext("dds:segment_size", namespaces=NS)) >= 20 * message_pair
    participant = rgbd.find("dds:participant", NS)
    assert participant.attrib["is_default_profile"] == "true"
    used = participant.findall("dds:rtps/dds:userTransports/dds:transport_id", NS)
    assert {item.text for item in used} == set(descriptors)
    assert participant.findtext("dds:rtps/dds:useBuiltinTransports", namespaces=NS) == "false"


@pytest.mark.parametrize("component,expected", [
    ("camera", "fastdds-rgbd.xml"),
    ("perception", "fastdds-rgbd.xml"),
    ("control", "fastdds.xml"),
    ("watchdog", "fastdds.xml"),
])
def test_existing_launcher_changes_only_camera_and_perception_profile(tmp_path, component, expected):
    fake_bin = tmp_path / "executables"
    fake_bin.mkdir()
    env = tmp_path / "env.sh"
    env.write_text("export EXPLORER_ROOT=\"" + str(tmp_path) + "\"\n"
        "export FASTRTPS_DEFAULT_PROFILES_FILE=\"$EXPLORER_ROOT/config/fastdds.xml\"\n")
    stub = "#!/bin/bash\nif [ \"$1\" = \"bin/wait-clock.py\" ]; then exit 0; fi\n" \
        "printf \"%s\\n\" \"$FASTRTPS_DEFAULT_PROFILES_FILE\"\nprintf \"%s\\n\" \"$@\"\n"
    for name in ("python3", "ros2"):
        program = fake_bin / name
        program.write_text(stub)
        program.chmod(0o755)
    # Only replace the fixed host env path; execute all real case branches.
    script = (ROOT / "bin" / "run.sh").read_text().replace(
        "/home/vlad/Explorer/bin/env.sh", str(env))
    process = subprocess.run(["bash", "-c", script, "run.sh", component],
        env={**os.environ, "PATH": str(fake_bin) + ":" + os.environ["PATH"]},
        text=True, capture_output=True, check=True)
    assert process.stdout.splitlines()[0] == str(tmp_path / "config" / expected)
    if component == "camera":
        assert "depth_registration:=true" in process.stdout
        assert "frame_aggregate_mode:=disable" in process.stdout
