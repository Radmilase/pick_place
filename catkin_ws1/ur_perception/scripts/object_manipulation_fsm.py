#!/usr/bin/env python3

import rospy
import smach
import smach_ros

import moveit_commander
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String

# --- Гриппер ---
gripper_pub = None

def open_gripper():
    global gripper_pub
    if gripper_pub is None:
        gripper_pub = rospy.Publisher('/gripper/command', JointTrajectory, queue_size=1)
        rospy.sleep(0.5)
    msg = JointTrajectory()
    msg.joint_names = ['gripper_finger1_joint']
    point = JointTrajectoryPoint()
    point.positions = [0.0]  # открыть
    point.time_from_start = rospy.Duration(1.0)
    msg.points = [point]
    gripper_pub.publish(msg)
    rospy.loginfo("Гриппер открыт")

def close_gripper():
    global gripper_pub
    if gripper_pub is None:
        gripper_pub = rospy.Publisher('/gripper/command', JointTrajectory, queue_size=1)
        rospy.sleep(0.5)
    msg = JointTrajectory()
    msg.joint_names = ['gripper_finger1_joint']
    point = JointTrajectoryPoint()
    point.positions = [0.8]  # закрыть, проверь это значение в симуляции!
    point.time_from_start = rospy.Duration(1.0)
    msg.points = [point]
    gripper_pub.publish(msg)
    rospy.loginfo("Гриппер закрыт")

# --- Состояния FSM ---

class Idle(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['object_detected'], output_keys=['object_type', 'object_pose'])
        self.detected_type = None
        self.detected_pose = None
        rospy.Subscriber("/detected_object", String, self.callback)
        # Пример: если ты сделаешь отдельный топик для позы
        rospy.Subscriber("/detected_object_pose", PoseStamped, self.pose_callback)

    def callback(self, msg):
        self.detected_type = msg.data

    def pose_callback(self, msg):
        self.detected_pose = msg

    def execute(self, userdata):
        rospy.loginfo('Состояние: Idle')
        while not rospy.is_shutdown():
            if self.detected_type is not None and self.detected_pose is not None:
                userdata.object_type = self.detected_type
                userdata.object_pose = self.detected_pose
                self.detected_type = None
                self.detected_pose = None
                return 'object_detected'
            rospy.sleep(0.1)

class PickObject(smach.State):
    def __init__(self, move_group):
        smach.State.__init__(self, outcomes=['object_picked'], input_keys=['object_pose'])
        self.move_group = move_group

    def execute(self, userdata):
        rospy.loginfo('Состояние: PickObject')
        pose = userdata.object_pose
        self.move_group.set_pose_target(pose)
        self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        close_gripper()
        rospy.sleep(1.0)
        return 'object_picked'

class PlaceObject(smach.State):
    def __init__(self, move_group, place_pose):
        smach.State.__init__(self, outcomes=['object_placed'])
        self.move_group = move_group
        self.place_pose = place_pose

    def execute(self, userdata):
        rospy.loginfo('Состояние: PlaceObject')
        self.move_group.set_pose_target(self.place_pose)
        self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        open_gripper()
        rospy.sleep(1.0)
        return 'object_placed'

class ReturnHome(smach.State):
    def __init__(self, move_group, home_joints):
        smach.State.__init__(self, outcomes=['returned'])
        self.move_group = move_group
        self.home_joints = home_joints

    def execute(self, userdata):
        rospy.loginfo('Состояние: ReturnHome')
        self.move_group.go(self.home_joints, wait=True)
        self.move_group.stop()
        return 'returned'

def main():
    rospy.init_node('object_manipulation_fsm')
    moveit_commander.roscpp_initialize([])
    move_group = moveit_commander.MoveGroupCommander('manipulator') # Имя группы из твоего moveit_config

    # --- Точки для pick, place, home (замени на свои!) ---
    # Типовая "pick" поза (где объект)
    pick_pose = PoseStamped()
    pick_pose.header.frame_id = 'base_link'
    pick_pose.pose.position.x = 0.0
    pick_pose.pose.position.y = 0.0
    pick_pose.pose.position.z = 0.7
    pick_pose.pose.orientation.w = 1.0

    # "Place" — где корзина для can
    can_place_pose = PoseStamped()
    can_place_pose.header.frame_id = 'base_link'
    can_place_pose.pose.position.x = 0.7
    can_place_pose.pose.position.y = 0.6
    can_place_pose.pose.position.z = 0.002000
    can_place_pose.pose.orientation.w = 1.0

    # "Place" — где корзина для tetrapack
    tetrapack_place_pose = PoseStamped()
    tetrapack_place_pose.header.frame_id = 'base_link'
    tetrapack_place_pose.pose.position.x = 0.6
    tetrapack_place_pose.pose.position.y = -0.2
    tetrapack_place_pose.pose.position.z = 0.2
    tetrapack_place_pose.pose.orientation.w = 1.0

    home_joints = [
    1.5707972694130632,    # shoulder_pan_joint
    -1.567004803999342,    # shoulder_lift_joint
    1.5895563321300497,    # elbow_joint
    -1.5635959737633243,   # wrist_1_joint
    -1.5708445536206561,   # wrist_2_joint
    5.445944025783689e-05  # wrist_3_joint \
    ]


    sm = smach.StateMachine(outcomes=['finished'])
    sm.userdata.object_type = None
    sm.userdata.object_pose = pick_pose  # пока — фиктивно

    with sm:
        smach.StateMachine.add('IDLE', Idle(),
            transitions={'object_detected': 'PICK_OBJECT'})
        smach.StateMachine.add('PICK_OBJECT', PickObject(move_group),
            transitions={'object_picked': 'DECIDE_PLACE'})
        
        # FSM развилка: куда класть (can или tetrapack)
        def place_decider(userdata, outcomes):
            if userdata.object_type == 'can':
                return 'CAN'
            else:
                return 'TETRA'
        smach.StateMachine.add('DECIDE_PLACE',
            smach.CBState(place_decider, input_keys=['object_type'], outcomes=['CAN', 'TETRA']),
            transitions={'CAN':'PLACE_CAN', 'TETRA':'PLACE_TETRA'})

        smach.StateMachine.add('PLACE_CAN', PlaceObject(move_group, can_place_pose),
            transitions={'object_placed': 'RETURN_HOME'})
        smach.StateMachine.add('PLACE_TETRA', PlaceObject(move_group, tetrapack_place_pose),
            transitions={'object_placed': 'RETURN_HOME'})
        smach.StateMachine.add('RETURN_HOME', ReturnHome(move_group, home_joints),
            transitions={'returned':'IDLE'})

    outcome = sm.execute()

if __name__ == '__main__':
    main()
