#!/usr/bin/env python3
import rospy
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64

class ArmControl:
    def __init__(self):
        # Названия суставов (см. /joint_states — порядок может быть другим!)
        self.joint_names = [
            'elbow_joint',
            'shoulder_lift_joint',
            'shoulder_pan_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        self.pub = rospy.Publisher('/eff_joint_traj_controller/command', JointTrajectory, queue_size=1)
        rospy.sleep(1)  # подождать инициализацию паблишера

        # Позиции поиска (сканирования) — ПОДСТАВЬ свои значения!
        self.search_positions = [
            np.array([1.58, -1.56, 0.0, -1.56, -1.57, 0.0]),  # пример для первой точки
            np.array([1.40, -1.10, 0.0, -1.45, -1.25, 0.0]),  # пример для второй точки
        ]
        self.search_idx = 0

        # Позиции для корзин
        self.trash_positions = {
            "can": np.array([0.7, 0.606, 0.25]),        # координаты корзины для банок (XYZ)
            "tetra-pak": np.array([0.7, -0.606, 0.25])  # координаты корзины для тетрапаков (XYZ)
        }

    def look_for_objects(self):
        pos = self.search_positions[self.search_idx]
        self.move_to_joint(pos)
        rospy.sleep(0.5)
        self.search_idx = (self.search_idx + 1) % len(self.search_positions)

    def pick_object(self, obj_class):
        rospy.loginfo(f"[Arm] Попытка захватить объект класса: {obj_class}")
        above = self.search_positions[(self.search_idx - 1) % len(self.search_positions)]
        self.move_to_joint(above)
        rospy.sleep(0.5)
        self.open_gripper()
        # Опускаем "вниз" — изменяем второй сустав
        low = above.copy()
        low[1] -= 0.15  # уменьшить shoulder_lift_joint (примерно вниз)
        self.move_to_joint(low)
        rospy.sleep(0.5)
        self.close_gripper()
        rospy.sleep(0.5)
        self.move_to_joint(above)
        rospy.sleep(0.5)
        return True

    def place_object(self, obj_class):
        xyz = self.trash_positions[obj_class]
        self.move_to_cartesian(xyz)
        rospy.sleep(0.5)
        self.open_gripper()
        rospy.loginfo(f"[Arm] Объект помещён в корзину: {obj_class}")

    def move_to_joint(self, pos):
        jt = JointTrajectory()
        jt.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = pos.tolist()
        pt.time_from_start.secs = 2
        jt.points.append(pt)
        self.pub.publish(jt)
        rospy.loginfo(f"[Arm] Двигаю манипулятор в: {pos}")
        rospy.sleep(2)  # дать время на движение

    def move_to_cartesian(self, xyz):
        rospy.loginfo(f"[Arm] (Заглушка) Перемещаю в XYZ: {xyz}")
        # Тут может быть IK, или заранее рассчитанные joint-углы для корзины!

    def open_gripper(self):
        pub = rospy.Publisher('/gripper/command', Float64, queue_size=1)
        rospy.sleep(0.2)
        pub.publish(0.03)  # открыто, подбери своё значение!
        rospy.loginfo("[Arm] Открываю хвататель")
        rospy.sleep(0.5)

    def close_gripper(self):
        pub = rospy.Publisher('/gripper/command', Float64, queue_size=1)
        rospy.sleep(0.2)
        pub.publish(0.0)   # закрыто, подбери своё значение!
        rospy.loginfo("[Arm] Закрываю хвататель")
        rospy.sleep(0.5)
