"""BatteryState, measured Jetson rails, telemetry and critical shutdown coordination.

No motor/arm publisher; STOP requests use the existing deterministic owner.
Never estimates battery current from Jetson-only input current.
"""
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import time
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String

ROOT = Path(os.environ.get('EXPLORER_ROOT', '/home/vlad/Explorer'))
OPTIONAL = ('llm', 'learning', 'train', 'speech', 'preview-*', 'grounding-*', 'perception', 'camera', 'planning', 'navigation')


def read_json(path):
    try:return json.loads(path.read_text())
    except (OSError, ValueError):return {}


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_rails():
    rails = {}
    for hwmon in Path('/sys/class/hwmon').glob('hwmon*'):
        try:
            if (hwmon/'name').read_text().strip() == 'ina3221':
                for i in (1, 2, 3):
                    label = (hwmon/f'in{i}_label').read_text().strip()
                    volts = float((hwmon/f'in{i}_input').read_text())/1000
                    amps = float((hwmon/f'curr{i}_input').read_text())/1000
                    rails[label] = dict(voltage_v=volts, current_a=amps, power_w=volts*amps)
        except (OSError, ValueError):pass
    return rails


class PowerManager(Node):
    def __init__(self):
        super().__init__('explorer_power')
        self.config = yaml.safe_load((ROOT/'config/power.yaml').read_text())
        self.battery_pub = self.create_publisher(BatteryState, '/explorer/battery_state', 1)
        self.state_pub = self.create_publisher(String, '/explorer/power_state', 1)
        self.stop_pub = self.create_publisher(String, '/explorer/request', 1)
        self.critical_at = None
        self.shutdown_attempted = False
        self.previous_state = None
        self.last_log = 0.
        self.previous_cpu = None
        self.last_llm_request = time.monotonic()
        self.last_llm_check = 0.
        self.last_mode_check = 0.
        self.mode = 'unknown'
        self.shutdown_available = False
        self.create_timer(1., self.tick)

    def optional_stop(self, names):
        # systemd performs the work asynchronously; never delay the safety owner.
        subprocess.run(['systemctl', '--user', '--no-block', 'stop']+
                       [f'explorer-{n}.service' for n in names], timeout=2, check=False)

    def stop(self):
        self.stop_pub.publish(String(data=json.dumps(dict(op='stop', at=time.monotonic(), source='power'))))

    def resources(self):
        result = dict(jetson_rails=read_rails(), whole_robot_power_w=None, nvpmodel=self.mode)
        try:
            cpu = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
            total, idle = sum(cpu[:8]), cpu[3]+cpu[4]
            result['cpu_percent'] = None
            if self.previous_cpu and total > self.previous_cpu[0]:
                result['cpu_percent'] = 100*(1-(idle-self.previous_cpu[1])/(total-self.previous_cpu[0]))
            self.previous_cpu = total, idle
            mem = {s.split(':')[0]:int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
            result['ram_available_mb'] = mem['MemAvailable']/1024
        except (OSError, ValueError, KeyError):pass
        result['temperatures_c'] = {}
        for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
            try:result['temperatures_c'][(p.parent/'type').read_text().strip()] = float(p.read_text())/1000
            except (OSError, ValueError):pass
        result['gpu_percent'] = None
        for p in (Path('/sys/devices/platform/17000000.gpu/load'), Path('/sys/devices/17000000.ga10b/load')):
            if p.exists():
                try:result['gpu_percent'] = float(p.read_text())/10
                except (OSError, ValueError):pass
        return result

    def tick(self):
        now = time.monotonic()
        status = read_json(ROOT/'data/status.json')
        fresh = 0 <= time.time()-status.get('at', 0) < 2
        power = status.get('power', {}) if fresh else {}
        state = power.get('state', 'UNKNOWN')
        voltage = status.get('battery') if fresh and status.get('sensor_age', {}).get('battery', 99)<3 else None
        battery = BatteryState()
        battery.header.stamp = self.get_clock().now().to_msg()
        battery.header.frame_id = 'base_link'
        battery.voltage = float(voltage) if voltage is not None else math.nan
        battery.temperature = battery.current = battery.charge = battery.capacity = battery.design_capacity = battery.percentage = math.nan
        battery.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_UNKNOWN
        battery.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_UNKNOWN
        battery.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_UNKNOWN
        battery.present = voltage is not None
        self.battery_pub.publish(battery)
        if state in ('CRITICAL', 'CHARGING', 'UNKNOWN'):self.stop()
        if state != self.previous_state:
            self.get_logger().info(f'Power state {self.previous_state} -> {state}')
            self.previous_state = state
            if state in ('LOW_POWER', 'CHARGING', 'UNKNOWN'):
                self.optional_stop(('llm','train','speech','preview-*','grounding-*'))
            if state == 'CRITICAL':
                self.critical_at = now
                self.optional_stop(OPTIONAL)
                atomic_json(ROOT/'data/critical-shutdown.json', dict(at=time.time(), phase='stopped', voltage=voltage,
                            arm='No automatic pose/torque change: uncalibrated; preserve holding torque'))
        if self.critical_at is not None:
            self.stop()
            if now-self.critical_at >= self.config['policy']['shutdown_delay_s'] and not self.shutdown_attempted:
                self.shutdown_attempted = True
                try:
                    dbpath = ROOT/'data/world.sqlite3'
                    if dbpath.exists():
                        with sqlite3.connect(dbpath, timeout=1) as db:db.execute('PRAGMA wal_checkpoint(FULL)')
                    os.sync()
                    result = subprocess.run(['sudo', '-n', '/usr/local/sbin/explorer-poweroff'], capture_output=True, text=True, timeout=5)
                    atomic_json(ROOT/'data/critical-shutdown.json', dict(at=time.time(), phase='shutdown_requested' if result.returncode == 0 else 'shutdown_failed',
                                returncode=result.returncode, error=result.stderr[-500:]))
                except (OSError, subprocess.TimeoutExpired, sqlite3.Error) as exc:
                    atomic_json(ROOT/'data/critical-shutdown.json', dict(at=time.time(), phase='shutdown_failed', error=str(exc)))
        if now-self.last_mode_check > 60:
            self.last_mode_check = now
            try:
                self.mode = subprocess.run(['nvpmodel', '-q'], capture_output=True, text=True, timeout=2).stdout.strip()
            except (OSError, subprocess.TimeoutExpired):self.mode = 'unknown'
            try:
                helper=Path('/usr/local/sbin/explorer-poweroff')
                self.shutdown_available=(helper.is_file() and helper.stat().st_uid==0 and
                    subprocess.run(['sudo','-n','-l','--',str(helper)],capture_output=True,timeout=2).returncode==0)
            except (OSError,subprocess.TimeoutExpired):self.shutdown_available=False
        demand = read_json(ROOT/'data/llm-demand.json')
        if 0 <= now-demand.get('monotonic', -1e9) < self.config['workloads']['llm_idle_timeout_s']:
            self.last_llm_request = now
        if now-self.last_llm_check > 15:
            self.last_llm_check = now
            if now-self.last_llm_request > self.config['workloads']['llm_idle_timeout_s']:self.optional_stop(('llm',))
        resources = self.resources()
        record = dict(at=time.time(), state=state, battery_voltage_v=voltage, battery_current_a=None,
                      soc=None, runtime_minutes=None, charging=power.get('charging'), resources=resources,
                      commanded_velocity=status.get('velocity'), arm_activity=status.get('arm_command_state',{}),
                      perception=read_json(ROOT/'data/perception.json').get('inference_ms'),
                      robot_mode=status.get('mode'), shutdown=read_json(ROOT/'data/critical-shutdown.json'))
        record['automatic_shutdown_available']=self.shutdown_available
        record['shutdown_setup_required']=not self.shutdown_available
        atomic_json(ROOT/'data/power.json', record)
        self.state_pub.publish(String(data=json.dumps(record, allow_nan=False)))
        if now-self.last_log >= 5:
            self.last_log = now
            with (ROOT/'data/power-telemetry.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')


def main():
    rclpy.init()
    node = PowerManager()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.shutdown()


if __name__ == '__main__':main()
