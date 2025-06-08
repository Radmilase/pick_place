#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

class SortingFSM:
    def __init__(self, arm):
        rospy.init_node("sorting_fsm_node")
        self.state = "SEARCH"
        self.detected_object = None
        self.arm = arm

        rospy.Subscriber("/detected_object", String, self.object_callback)
        rospy.loginfo("[FSM] Started, searching for objects...")

    def object_callback(self, msg):
        if self.state == "SEARCH":
            self.detected_object = msg.data
            rospy.loginfo(f"[FSM] Detected: {self.detected_object}")
            if self.detected_object in ["can", "tetra-pak"]:
                self.state = "PICK"

    def spin(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.state == "SEARCH":
                # Двигай манипулятор по очереди над потенциальными объектами
                self.arm.look_for_objects()
            elif self.state == "PICK":
                if self.arm.pick_object(self.detected_object):
                    self.state = "PLACE"
                else:
                    self.state = "SEARCH"
            elif self.state == "PLACE":
                self.arm.place_object(self.detected_object)
                self.state = "SEARCH"
            rate.sleep()

if __name__ == "__main__":
    from arm_control_node import ArmControl  # Импортируй свой класс управления!
    arm = ArmControl()
    fsm = SortingFSM(arm)
    fsm.spin()
