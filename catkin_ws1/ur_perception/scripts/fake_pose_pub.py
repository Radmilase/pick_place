#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped

rospy.init_node('fake_pose_pub')
pub = rospy.Publisher('/detected_object_pose', PoseStamped, queue_size=1)
rate = rospy.Rate(1)

while not rospy.is_shutdown():
    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x = 0.4
    pose.pose.position.y = 0.2
    pose.pose.position.z = 0.15
    pose.pose.orientation.w = 1.0
    pub.publish(pose)
    rate.sleep()
