#!/usr/bin/env python3
import sys
sys.path.append('/home/rad/yolov5')  # Добавляем путь к yolov5
print(sys.path)
import rospy
import cv2
import numpy as np
import torch
import json
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes

#from yolov5.models.experimental import attempt_load
#from yolov5.utils.general import non_max_suppression, scale_boxes

class TetraPakDetector:
    def __init__(self):
        rospy.init_node('tetra_pak_detector')
        
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

if __name__ == '__main__':
    try:
        detector = TetraPakDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
