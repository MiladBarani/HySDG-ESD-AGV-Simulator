# HySDG-ESD: Dynamic Obstacle Detection for Moving AGVs Using LiDAR and Ego-Motion Compensation

This repository provides a complete simulation and perception framework
for dynamic and static obstacle detection around an Autonomous Guided
Vehicle (AGV) using 2D LiDAR, clustering, multi-object tracking, Kalman
filtering, and HySDG-ESD--based dynamic classification with ego-motion
compensation.

The framework simulates a moving AGV in a bounded environment and
performs real-time obstacle detection, tracking, and classification in
the global reference frame.

------------------------------------------------------------------------

## ✨ Main Features

-   2D LiDAR scan simulation with configurable noise and field of view\
-   HDBSCAN-based obstacle clustering\
-   Ego-motion compensation using rotation matrices\
-   Multi-object tracking using Hungarian assignment and Kalman
    filtering\
-   Adaptive velocity damping\
-   HySDG-ESD dynamic classification\
-   Moving AGV always inside the environment\
-   Interactive visualization with multiple scenarios

------------------------------------------------------------------------

## 🧠 Method Overview

Pipeline: 1. LiDAR simulation\
2. HDBSCAN clustering\
3. Coordinate transformation to world frame\
4. Multi-object tracking with EKF\
5. Relative-velocity and HySDG-ESD based classification\
6. Visualization

------------------------------------------------------------------------

## 📂 Repository Structure

AGV-Dynamic-Obstacle-Detection/ 

├── HySDG_EKF_Hdbscan_Enhanced.py/

├── AGV_MOVING_SYSTEM_HDBSCAN_Enhanced.py/

├── README.md/

├── requirements.txt/

├── figures/

└── results/

------------------------------------------------------------------------

## ▶️ Installation

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🚀 Run

``` bash
1️⃣ HySDG_EKF_Hdbscan_Fast_Enhanced.py 😍👍
2️⃣ AGV_MOVING_SYSTEM_HDBSCAN_Enhanced.py 😊
```

------------------------------------------------------------------------

## 🎮 Scenarios

-   Scenario 1 -- Static obstacles/
-   Scenario 2 -- Dynamic obstacles/
-   Scenario 3 -- Mixed

------------------------------------------------------------------------

## 📄 License

MIT License

------------------------------------------------------------------------

## 👤 Author

MILAD JAFARI BARANI
PhD Researcher -- Explainable AI & Intelligent Systems
