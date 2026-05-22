# ROS2 Sensor Recording Guide

This guide explains how to activate the RGB-T sensor nodes, control the pan-tilt motor, record calibration data, and extract synchronized RGB-T images.

## Requirements

- Ubuntu 22.04
- ROS2 Humble
- ROS2 workspace: `~/ros2_ws`

## 1. Activate Camera Nodes

### 1.1 Activate CellPlus Thermal Camera Node

```bash
ros2 launch komipo_sensor_driver cellplus_cam_launch.py
```

### 1.2 Activate Harrier RGB Camera Node

```bash
ros2 launch komipo_sensor_driver harrier_cam_launch.py
```

## 2. Activate Pan-Tilt Motor

Before controlling the pan-tilt motor, initialize the CAN port.

```bash
ros2 launch komipo_sensor_driver init_can_port_launch.py can_port:=can0
```

## 3. Pan-Tilt Motor Test

Use the following command to test whether the pan-tilt motor works correctly.

```bash
ros2 launch komipo_sensor_driver gl2_motor_launch.py \
motor1_position:=-0.8 \
motor1_velocity:=0.5 \
motor2_position:=-0.2 \
motor2_velocity:=0.1
```

## 4. Motor Control and Rosbag Recording

Run the pan-tilt motor sequence and record sensor data.

```bash
ros2 launch komipo_sensor_driver moter_serise_launch.py \
config:=/home/aidin/ros2_ws/src/komipo-perception-ros2/config/moter_serise_control.yaml
```

Before running this command, check the motor sequence configuration file.

```bash
/home/aidin/ros2_ws/src/komipo-perception-ros2/config/moter_serise_control.yaml
```

## 5. Extract Synchronized RGB-T Data

After recording rosbag files, extract synchronized RGB and thermal images using `fusion_sync`.

```bash
ros2 run fusion_sync fusion_sync --ros-args \
-p input_dir:=[input folder] \
-p output_dir:=[output folder]
```

Example:

```bash
ros2 run fusion_sync fusion_sync --ros-args \
-p input_dir:=/home/aidin/ros_bag/Calibration \
-p output_dir:=/home/aidin/extracted_data
```

### 5.1 Input Folder Structure

The input folder should follow the structure below.

```text
Input folder
└── Calibration
    └── pan-0.400000_tilt0.000000
        ├── 7
        │   ├── 7_0.db3
        │   └── metadata.yaml
        └── 8
            ├── 8_0.db3
            └── metadata.yaml
```

## Overall Workflow

```text
1. Source ROS2 workspace
2. Turn off head light
3. Launch thermal camera node
4. Launch RGB camera node
5. Initialize CAN port
6. Test pan-tilt motor
7. Run motor sequence and record rosbag
8. Extract synchronized RGB-T images
```

