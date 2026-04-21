"""
Starter hook for item #12 (ROS2 bridge).
Replace with real rclpy publisher/subscriber wiring when ROS2 is available.
"""


class ROS2BridgeStub:
    def __init__(self):
        self.connected = False
        self.last_published = None

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def publish_joint_targets(self, motors_deg):
        if not self.connected:
            return False
        self.last_published = [float(v) for v in motors_deg]
        return True
