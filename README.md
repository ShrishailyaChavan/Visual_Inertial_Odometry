# Visual Inertial Odometry

[Course Website](https://rbe549.github.io/spring2023/proj/p4/#classical)

MSCKF (Multi-State Constraint Kalman Filter) is an EKF based tightly-coupled visual-inertial odometry algorithm. [S-MSCKF](https://arxiv.org/abs/1712.00036) is MSCKF's stereo version. This project is a Python reimplemention of S-MSCKF, the code is directly translated from official C++ implementation [KumarRobotics/msckf_vio](https://github.com/KumarRobotics/msckf_vio).

For algorithm details, please refer to:

1. Robust Stereo Visual Inertial Odometry for Fast Autonomous Flight, Ke Sun et al. (2017)
2. A Multi-State Constraint Kalman Filterfor Vision-aided Inertial Navigation, Anastasios I. Mourikis et al. (2006)

## Dataset

Used the Machine Hall 01 easy or (MH_01_easy) subset of the [EuRoC dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) to test our implementation. Please download the data in ASL data format from [here](https://www.google.com). Here, the data is collected using a VI sensor carried by a quadrotor flying a trajectory. The ground truth is provided by a sub-mm accurate Vicon Motion capture system.

## Function Structure of msckf

![Function Structure of msckf](https://github.com/ShrishailyaChavan/Visual_Inertial_Odometry/blob/main/photos/msckf.png)


## Pre-requisite

 - Python3
 - VS Code
 - numpy
- scipy
- cv2
- [pangolin](https://github.com/uoip/pangolin)

## Run the code

Change the location to

```sh
 python vio.py --view --path path/to/your/EuRoC_MAV_dataset/MH_01_easy


```
or 
```sh
 python vio.py --path path/to/your/EuRoC_MAV_dataset/MH_01_easy (no visualization)
```
