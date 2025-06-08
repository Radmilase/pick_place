#!/usr/bin/env python3
import rospy
from enum import Enum, auto
from your_package.srv import PickPlace, PickPlaceRequest

class State(Enum):
    SEARCH = auto()
    PICK = auto()
    PLACE = auto()

class FSM:
    def __init__(self):
        rospy.init_node('fsm_node')
        self.state = State.SEARCH
        self.current_object = None
        self.pick_place_client = rospy.ServiceProxy('/pick_place', PickPlace)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.state == State.SEARCH:
                self.search()
            elif self.state == State.PICK:
                self.pick()
            elif self.state == State.PLACE:
                self.place()
            rate.sleep()

    def search(self):
        rospy.loginfo("Поиск объектов...")
        # Ждем данные от YOLO (пример подписки)
        data = rospy.wait_for_message('/detected_objects', DetectedObjects)
        if data.objects:
            self.current_object = data.objects[0]
            self.state = State.PICK

    def pick(self):
        rospy.loginfo(f"Захват объекта: {self.current_object.class_label}")
        req = PickPlaceRequest()
        req.action = "pick"
        req.pose = self.current_object.pose
        resp = self.pick_place_client(req)
        if resp.success:
            self.state = State.PLACE

    def place(self):
        rospy.loginfo(f"Перенос в корзину")
        req = PickPlaceRequest()
        req.action = "place"
        req.object_class = self.current_object.class_label  # "can" или "carton"
        resp = self.pick_place_client(req)
        if resp.success:
            self.state = State.SEARCH

if __name__ == "__main__":
    FSM().run()