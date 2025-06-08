#!/bin/bash

# Активируем среду ROS
source ~/catkin_ws/devel/setup.bash

# Запускаем ноды в фоне (чтобы работали параллельно)
roslaunch solution_master Ex.launch &
roslaunch solution_master tetra_detection.launch &

# Ждем завершения (если нужно)
wait