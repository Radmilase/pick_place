#!/usr/bin/env python3
import rospy
from moveit_commander import MoveGroupCommander
from your_package.srv import PickPlaceResponse, PickPlace

class ArmController:
    def __init__(self):
        rospy.init_node('arm_controller')
        self.arm = MoveGroupCommander("arm_group")
        self.gripper = MoveGroupCommander("gripper_group")
        
        # Координаты корзин (задаются вручную)
        self.bins = {
            "can": [0.5, -0.4, 0.1],    # bin1 (слева)
            "carton": [0.5, 0.4, 0.1]   # bin2 (справа)
        }
        
        rospy.Service('/pick_place', PickPlace, self.handle_pick_place)

    def handle_pick_place(self, req):
        if req.action == "pick":
            self.pick(req.pose)
        elif req.action == "place":
            self.place(req.object_class)
        return PickPlaceResponse(success=True)

    def pick(self, pose):
        self.arm.set_pose_target(pose)
        self.arm.go(wait=True)
        self.gripper.set_joint_value_target([0.0, 0.0])  # Захват
        self.gripper.go(wait=True)

    def place(self, object_class):
        target = self.bins[object_class]
        self.arm.set_position_target(target)
        self.arm.go(wait=True)
        self.gripper.set_joint_value_target([0.04, 0.04])  # Отпускание
        self.gripper.go(wait=True)

if __name__ == "__main__":
    ArmController()
    rospy.spin()