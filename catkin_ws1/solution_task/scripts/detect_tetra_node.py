#!/usr/bin/env python3


import sys
sys.path.append('/home/rad/yolov5')  # Добавляем путь к yolov5

import rospy
import cv2
import os
import time
import numpy as np
import torch
import json

from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2, PointField, Range
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Time, Header, Duration
from controller_manager_msgs.srv import SwitchController
from control_msgs.msg import *
from trajectory_msgs.msg import *

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes

JOINT_NAMES = ['shoulder_pan_joint','shoulder_lift_joint',  'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

class Example():
    def __init__(self):
        rospy.loginfo("[Example] loading")
        rospy.on_shutdown(self.shutdown)
        self.gui = os.getenv('GUI') in ['true', 'True']

        sub_image_topic_name = "/hand_eye/camera/rgb/image_raw"
        sub_point_cloud_topic_name = "/hand_eye/camera/depth/points"

        self.joint_position_publisher = rospy.Publisher("/joint_group_eff_controller/command", Float64MultiArray, queue_size=1)
        self.joint_trajectory_publisher = rospy.Publisher('/eff_joint_traj_controller/command', JointTrajectory, queue_size=1)
        self.gripper_trajectory_publisher = rospy.Publisher('/gripper/command', JointTrajectory, queue_size=3)
        self.current_image = None
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
                rospy.logwarn(f"Switch controller failed: {response}")
            else:
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
        rospy.loginfo("Trajectory sent")
        rospy.sleep(moving_time)
        
    def go_to_using_servo(self, goal: np.ndarray):
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
    
    def shutdown(self):
        self.switch_controller(
            start_controllers=["joint_group_eff_controller"],
            stop_controllers=["eff_joint_traj_controller"]
        )

    def spin(self):
        rate = rospy.Rate(30)
        t0 = rospy.get_time()
        while not rospy.is_shutdown():
            t = rospy.get_time() - t0
            if t % 30 < 10:
                self.go_to_using_servo(np.array([np.sin(t)+1.57079632679, -1.57079632679, 1.57079632679, -1.57079632679, -1.57079632679, 0.0]))
                rate.sleep()
            elif t % 30 < 15:
                rospy.loginfo("To home position")
                self.open_gripper()
                self.go_to_using_trajectory(
                    np.array([1.57079632679, -1.57079632679, 1.57079632679, -1.57079632679, -1.57079632679, 0.0]),
                    moving_time=5
                )
                self.close_gripper()

class TetraPakDetector:
    def __init__(self):
        # rospy.init_node('tetra_pak_detector')
        
        # Загрузка модели YOLOv5
        self.model = attempt_load('/home/rad/yolov5/runs/train/custom_train7/weights/best.pt')
        self.names = ['object', 'tetra']
        self.conf_thres = 0.5
        
        # Инициализация CV Bridge
        self.bridge = CvBridge()
        
        # Подписка на изображение с камеры
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw", 
            Image, 
            self.image_callback
        )
        
        # Подписка на облако точек
        self.pc_sub = rospy.Subscriber(
            "/camera/depth/points", 
            PointCloud2, 
            self.pc_callback
        )
        
        # Публикация результатов детекции
        self.detection_pub = rospy.Publisher(
            "/detected_objects", 
            String, 
            queue_size=10
        )
        
        self.latest_pc = None

    def pc_callback(self, msg):
        self.latest_pc = msg

    def image_callback(self, msg):
        try:
            # Конвертация ROS Image в OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Детекция объектов
            results = self.detect_objects(cv_image)
            
            # Публикация результатов
            self.publish_detections(results, msg.header)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def detect_objects(self, image):
        # Препроцессинг изображения
        img = cv2.resize(image, (640, 640))
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        
        # Вывод модели
        pred = self.model(torch.from_numpy(img).unsqueeze(0))[0]
        pred = non_max_suppression(pred, self.conf_thres, 0.45)
        
        # Обработка результатов
        detections = []
        for det in pred:
            if det is not None and len(det):
                for *xyxy, conf, cls in reversed(det):
                    if self.names[int(cls)] == 'tetra':
                        detections.append({
                            "bbox": [int(x) for x in xyxy],
                            "confidence": float(conf),
                            "class": self.names[int(cls)]
                        })
        return detections

    def publish_detections(self, detections, header):
        if self.latest_pc is None:
            return
        
        msg = {
            "header": {
                "stamp": header.stamp.to_sec(),
                "frame_id": header.frame_id
            },
            "objects": detections
        }
        self.detection_pub.publish(json.dumps(msg))


def main():
    # Запуск управляющей ноды робота
    rospy.init_node("example_node", anonymous=True)
    example = Example()

    # # Запуск ноды детектора
    detector = TetraPakDetector()

    # Запускаем параллельно две ноды
    # Для простоты запустим в одном потоке, можно использовать threading, если нужно
    import threading

    example_thread = threading.Thread(target=example.spin)
    example_thread.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

if __name__ == "__main__":
    main()


