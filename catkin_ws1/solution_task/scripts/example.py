#!/usr/bin/env python3

import rospy
import cv2
import os
from cv_bridge import CvBridge
import time


from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Range
import actionlib
from control_msgs.msg import *
from trajectory_msgs.msg import *

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Time
from std_msgs.msg import Header
from std_msgs.msg import Duration
import numpy as np


from controller_manager_msgs.srv import SwitchController

JOINT_NAMES = ['shoulder_pan_joint','shoulder_lift_joint',  'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

class Example():

    def __init__(self):
        rospy.loginfo("[Example] loaging")
        rospy.on_shutdown(self.shutdown)
        self.gui = os.getenv('GUI')=='true' or os.getenv('GUI')=='True'

        sub_image_topic_name = "/hand_eye/camera/rgb/image_raw"
        sub_point_cloud_topic_name = "/hand_eye/camera/depth/points"

        self.joint_position_publisher = rospy.Publisher("/joint_group_eff_controller/command", Float64MultiArray, queue_size=1)
        self.joint_trajectory_publisher = rospy.Publisher('/eff_joint_traj_controller/command', JointTrajectory, queue_size=1)
        self.gripper_trajectory_publisher = rospy.Publisher('/gripper/command', JointTrajectory, queue_size=3)
        self.curent_image = None
        self.joints_pose = None
        self.joints_velocity = None
        self.bridge = CvBridge()

        rospy.wait_for_service("/controller_manager/switch_controller")
        self.switch_controller_service = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)

        self.camera_subscriber = rospy.Subscriber(sub_image_topic_name, Image, self.camera_callback)
        self.point_cloud_subscriber = rospy.Subscriber(sub_point_cloud_topic_name, PointCloud2, self.point_cloud_callback)
        self.joint_state_subscriber = rospy.Subscriber("/joint_states", JointState, self.joint_states_callback)

        rospy.loginfo("[Example] loaded")
        self.current_controller = "joint_group_eff_controller"

    def __del__(self):
        pass

    def switch_controller(self, start_controllers: list, stop_controllers: list):
        if start_controllers[0] != self.current_controller:
            response = self.switch_controller_service(
                start_controllers=start_controllers, stop_controllers=stop_controllers,
                strictness=1, start_asap=False, timeout=0.2
            )
            if not response.ok:
                print("Switch controller: ", response.ok)
                print(response)
            self.current_controller = start_controllers[0]

    def go_to_using_trajectory(self, goal: np.ndarray, moving_time: float = 5):
        self.switch_controller(
            start_controllers=["eff_joint_traj_controller"],
            stop_controllers=["joint_group_eff_controller"]
        )
        joints_trj = JointTrajectory()
        joints_trj.header = Header()
        joints_trj.header.stamp = rospy.get_rostime()
        joints_trj.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = goal.tolist()
        point.time_from_start = rospy.Duration.from_sec(moving_time)
        joints_trj.points.append(point)

        self.joint_trajectory_publisher.publish(joints_trj)
        rospy.loginfo("Trajectory sended")
        rospy.sleep(moving_time)
        
    def go_to_using_servo(self, goal: np.ndarray):
        # Do not use trajectory planning before start moving, go to goal immediately
        self.switch_controller(
            start_controllers=["joint_group_eff_controller"],
            stop_controllers=["eff_joint_traj_controller"]
        )
        msg = Float64MultiArray()
        msg.data = goal.tolist()
        self.joint_position_publisher.publish(msg)
    
    def open_gripper(self):
        gripper_trj = JointTrajectory()
        gripper_trj.header = Header()
        gripper_trj.header.stamp = rospy.get_rostime()
        gripper_trj.joint_names = ['gripper_finger1_joint']
        point = JointTrajectoryPoint()
        point.positions = [0.0]
        point.time_from_start = rospy.Duration.from_sec(1.0)
        gripper_trj.points.append(point)

        self.gripper_trajectory_publisher.publish(gripper_trj)
        rospy.loginfo("Open gripper")
        rospy.sleep(1)

    def close_gripper(self):
        gripper_trj = JointTrajectory()
        gripper_trj.header = Header()
        gripper_trj.header.stamp = rospy.get_rostime()
        gripper_trj.joint_names = ['gripper_finger1_joint']
        point = JointTrajectoryPoint()
        point.positions = [0.2]
        point.time_from_start = rospy.Duration.from_sec(1.0)
        gripper_trj.points.append(point)

        self.gripper_trajectory_publisher.publish(gripper_trj)
        rospy.loginfo("Close gripper")
        rospy.sleep(1)


    def camera_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg)
        # some processing here
        if self.gui:
            cv2.imshow("output", frame)
            cv2.waitKey(1)
    
    def point_cloud_callback(self, msg):
        pass

    def joint_states_callback(self, msg: JointState):
        self.joints_pose = msg.position
        self.joints_velocity = msg.velocity
        # print("Current joint position: ", self.joints_pose)
    
    def shutdown(self):
        # stop robots here
        self.switch_controller(
            start_controllers=["joint_group_eff_controller"],
            stop_controllers=["eff_joint_traj_controller"]
        )

    def spin(self):

        rate = rospy.Rate(30)
        t0 = rospy.get_time()
        while not rospy.is_shutdown():
            t = rospy.get_time() - t0
            if t%30<10:
                self.go_to_using_servo(np.array([np.sin(t)+1.57079632679, -1.57079632679, 1.57079632679, -1.57079632679, -1.57079632679, 0.0]))
                rate.sleep()
            elif t%30<15:
                print("To home position")
                self.open_gripper()
                self.go_to_using_trajectory(
                    np.array([1.57079632679, -1.57079632679, 1.57079632679, -1.57079632679, -1.57079632679, 0.0]),
                    moving_time =5
                )
                self.close_gripper()
                # ATTENTION: Dont use system time for control and estimating, use simulation time instead

            


def main(args=None):
    rospy.init_node("example_node")

    exp = Example()
    exp.spin()


if __name__ == "__main__":
    main()
