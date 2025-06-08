#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO

class YOLOv11ClassifierNode:
    def __init__(self):
        rospy.init_node("object_classifier_node")

        self.model_path = rospy.get_param("~model_path", "/home/rad/robot_programming/catkin_ws/src/best.pt")
        self.conf_threshold = rospy.get_param("~confidence_threshold", 0.5)

        rospy.loginfo(f"Загрузка YOLOv11 модели: {self.model_path}")
        self.model = YOLO(self.model_path)

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("hand_eye/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.class_pub = rospy.Publisher("/detected_object", String, queue_size=10)
        self.result_image_pub = rospy.Publisher("/yolo_result_image", Image, queue_size=1)

        rospy.loginfo("YOLOv11 Object Classifier запущен и ждёт картинки 🚀")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Ошибка при получении изображения: {e}")
            return

        results = self.model.predict(frame, conf=self.conf_threshold)
        result_frame = frame.copy()

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes[i]
                conf = box.conf.item()
                if conf < self.conf_threshold:
                    continue
                # YOLOv11 возвращает xyxy координаты в виде массива shape (1,4)
                x1, y1, x2, y2 = [int(coord.item()) for coord in box.xyxy[0]]
                label_idx = int(box.cls.item())
                label = self.model.names[label_idx]
                # Нарисовать рамку
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Нарисовать подпись
                cv2.putText(result_frame, f'{label} {conf:.2f}', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Публикуем самый уверенный класс (можно убрать если не надо)
            best_idx = boxes.conf.argmax().item()
            label_idx = int(boxes.cls[best_idx].item())
            label = self.model.names[label_idx]
            conf = boxes.conf[best_idx].item()
            if conf >= self.conf_threshold:
                rospy.loginfo(f"Объект: {label}, уверенность: {conf:.2f}")
                self.class_pub.publish(String(label))

        # Публикуем картинку с рамками для RViz
        result_msg = self.bridge.cv2_to_imgmsg(result_frame, encoding="bgr8")
        self.result_image_pub.publish(result_msg)

if __name__ == "__main__":
    try:
        node = YOLOv11ClassifierNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
