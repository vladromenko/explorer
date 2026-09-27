"""Independent process stops the base if the actuator owner stops ticking.

This complements, but cannot replace, a verified MCU timeout on serial loss.
"""
import time
import signal
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import UInt64
from geometry_msgs.msg import Twist

class Watchdog(Node):
    def __init__(self):
        super().__init__('explorer_watchdog')
        self.last = 0
        self.pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.create_subscription(UInt64, '/explorer/control_heartbeat', self.beat, 1)
        self.create_timer(.05, self.check)
    def beat(self,msg):
        if 0 <= time.monotonic_ns()-msg.data < 200_000_000:
            self.last=time.monotonic()
    def check(self):
        if time.monotonic()-self.last>.2:
            self.pub.publish(Twist())

rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
def shutdown_signal(signum,frame):raise KeyboardInterrupt
signal.signal(signal.SIGTERM,shutdown_signal)
signal.signal(signal.SIGINT,shutdown_signal)
node=Watchdog()
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    for _ in range(5):
        node.pub.publish(Twist())
        time.sleep(.02)
    node.destroy_node()
    rclpy.shutdown()
