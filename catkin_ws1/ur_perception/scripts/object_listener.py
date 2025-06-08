#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

class ObjectListener:
    def __init__(self):
        rospy.init_node('object_listener', anonymous=True)
        self.object_type = None
        self.object_pose = None

        rospy.Subscriber("/detected_object", String, self.type_callback)
        rospy.Subscriber("/detected_object_pose", PoseStamped, self.pose_callback)

    def type_callback(self, msg):
        self.object_type = msg.data
        self.print_object()

    def pose_callback(self, msg):
        self.object_pose = msg
        self.print_object()

    def print_object(self):
        if self.object_type and self.object_pose:
            x = self.object_pose.pose.position.x
            y = self.object_pose.pose.position.y
            z = self.object_pose.pose.position.z
            print(f"Обнаружен объект: {self.object_type}, координаты: ({x:.3f}, {y:.3f}, {z:.3f})")
            # Чтобы не выводить по 100 раз подряд
            self.object_type = None
            self.object_pose = None

if __name__ == "__main__":
    listener = ObjectListener()
    rospy.spin()
