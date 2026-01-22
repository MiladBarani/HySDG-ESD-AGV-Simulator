# =========================
# FINAL VERSION - با Rotation Matrix برای چرخش AGV
# =========================

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment

# =============================================================================
# Data Structures
# =============================================================================

class ObstacleState(Enum):
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"
    UNKNOWN = "UNKNOWN"

@dataclass
class LidarPoint:
    angle: float
    distance: float
    x: float
    y: float

@dataclass
class Obstacle:
    id: int
    center: np.ndarray
    velocity: np.ndarray
    points: List[LidarPoint]
    state: ObstacleState
    d_eq: float
    d_dot: float
    confidence: float
    last_seen: int
    state_history: List[ObstacleState]
    velocity_history: List[float]

# =============================================================================
# Extended Kalman Filter با مدل Constant Velocity
# =============================================================================

class ExtendedKalmanFilterCV:
    """
    Extended Kalman Filter با:
    - Process noise محافظه‌کارانه
    - Innovation gate برای outlier rejection
    - Adaptive velocity damping
    """
    def __init__(self, dt: float, process_noise: float = 0.05, measurement_noise: float = 0.2):
        self.dt = dt
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise - بسیار محافظه‌کارانه برای موانع استاتیک
        q = process_noise
        self.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ]) * 0.3  # کاهش بیشتر
        
        # Measurement noise
        self.R = np.eye(2) * measurement_noise
        
        # State and covariance
        self.P = np.eye(4) * 5.0
        self.x = np.zeros((4, 1))
        self.initialized = False
        self.update_count = 0
        
        # Innovation history برای تطبیق adaptive
        self.innovation_history = []

    def predict(self):
        if not self.initialized:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> np.ndarray:
        z = np.array(z).reshape(2, 1)
        
        if not self.initialized:
            self.x[0:2] = z
            self.x[2:4] = 0  # فرض اولیه: ثابت
            self.initialized = True
            self.update_count = 1
            return self.x.copy()
        
        self.update_count += 1
        
        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        # Mahalanobis distance برای outlier rejection
        try:
            S_inv = np.linalg.inv(S)
            mahal_dist = np.sqrt(y.T @ S_inv @ y)[0, 0]
        except:
            mahal_dist = 0
        
        # Innovation gate
        if mahal_dist > 5.0:  # threshold
            return self.x.copy()
        
        # Kalman gain
        K = self.P @ self.H.T @ S_inv
        
        # Update state
        self.x = self.x + K @ y
        
        # ⭐⭐⭐ Adaptive velocity damping
        # در فریم‌های اولیه و وقتی innovation کم است، سرعت را محدود کن
        self.innovation_history.append(np.linalg.norm(y))
        if len(self.innovation_history) > 10:
            self.innovation_history.pop(0)
        
        avg_innovation = np.mean(self.innovation_history) if self.innovation_history else 0
        
        if self.update_count < 10:
            # فریم‌های اولیه: فرض ثابت
            damping = 0.05
        elif avg_innovation < 0.1:
            # Innovation کم: احتمالاً ثابت
            damping = 0.2
        else:
            # Innovation زیاد: احتمالاً متحرک
            damping = 1.0
        
        self.x[2:4] *= damping
        
        # Update covariance
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.x.copy()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = self.x[0:2].flatten()
        vel = self.x[2:4].flatten()
        return pos, vel

# =============================================================================
# Coordinate Transform Utilities
# =============================================================================

def rotation_matrix_2d(theta: float) -> np.ndarray:
    """
    ماتریس چرخش 2D
    theta: زاویه به رادیان
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def transform_to_world_frame(pos_agv_frame: np.ndarray, agv_pos: np.ndarray, 
                            agv_heading: float) -> np.ndarray:
    """
    تبدیل موقعیت از AGV frame به World frame
    
    pos_agv_frame: موقعیت در coordinate system AGV
    agv_pos: موقعیت AGV در world
    agv_heading: جهت AGV (رادیان)
    """
    R = rotation_matrix_2d(agv_heading)
    pos_world = agv_pos + R @ pos_agv_frame
    return pos_world

# =============================================================================
# LiDAR Processing
# =============================================================================

class LidarProcessor:
    def __init__(self, min_angle=-135.0, max_angle=135.0,
                 max_range=10.0, min_points=6, cluster_distance=0.5):
        self.min_angle = np.deg2rad(min_angle)
        self.max_angle = np.deg2rad(max_angle)
        self.max_range = max_range
        self.min_points = min_points
        self.cluster_distance = cluster_distance

    def parse_scan(self, ranges: np.ndarray, angles: np.ndarray) -> List[LidarPoint]:
        points = []
        for angle, distance in zip(angles, ranges):
            if (distance < 0.1 or distance > self.max_range or
                angle < self.min_angle or angle > self.max_angle):
                continue
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            points.append(LidarPoint(angle, distance, x, y))
        return points

    def cluster_points(self, points: List[LidarPoint]) -> List[List[LidarPoint]]:
        if len(points) < 3:
            return []

        X = np.array([[p.x, p.y] for p in points])
        db = DBSCAN(eps=0.4, min_samples=3).fit(X)

        labels = db.labels_
        clusters = []

        for label in set(labels):
            if label == -1:
                continue
            cluster = [points[i] for i in range(len(points)) if labels[i] == label]
            if len(cluster) >= 3:
                clusters.append(cluster)

        return clusters

# =============================================================================
# Multi-Object Tracker با Rotation-Aware Transform
# =============================================================================

class MultiObjectTracker:
    def __init__(self, dt: float, max_distance: float = 2.0, max_age: int = 5):
        self.dt = dt
        self.max_distance = max_distance
        self.max_age = max_age
        self.obstacles: List[Obstacle] = []
        self.next_id = 0
        self.current_time = 0
        self.kalman_filters = {}

    def predict_all(self):
        for obs in self.obstacles:
            if obs.id in self.kalman_filters:
                self.kalman_filters[obs.id].predict()

    def update(self, clusters: List[List[LidarPoint]], agv_pos: np.ndarray, 
               agv_vel: np.ndarray, agv_heading: float):
        """
        ⭐⭐⭐ با اضافه شدن agv_heading برای تبدیل صحیح coordinate
        """
        self.current_time += 1
        self.predict_all()

        if not clusters:
            self._remove_old_obstacles()
            return

        # ⭐⭐⭐ تبدیل cluster centers به world frame با rotation
        cluster_centers_world = []
        for cluster in clusters:
            # محاسبه center در AGV frame
            x_mean = np.mean([p.x for p in cluster])
            y_mean = np.mean([p.y for p in cluster])
            center_agv_frame = np.array([x_mean, y_mean])
            
            # ⭐⭐⭐ تبدیل به world frame با در نظر گرفتن rotation
            center_world = transform_to_world_frame(center_agv_frame, agv_pos, agv_heading)
            cluster_centers_world.append(center_world)

        # Hungarian Algorithm
        if self.obstacles and cluster_centers_world:
            cost_matrix = np.zeros((len(self.obstacles), len(cluster_centers_world)))
            
            for i, obs in enumerate(self.obstacles):
                for j, center_world in enumerate(cluster_centers_world):
                    dist = np.linalg.norm(obs.center - center_world)
                    cost_matrix[i, j] = dist if dist < self.max_distance else 1e6
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_obs = set()
            matched_clusters = set()
            
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < self.max_distance:
                    self._update_obstacle(i, cluster_centers_world[j], clusters[j], 
                                        agv_pos, agv_vel)
                    matched_obs.add(i)
                    matched_clusters.add(j)
        else:
            matched_obs = set()
            matched_clusters = set()

        for j, cluster in enumerate(clusters):
            if j not in matched_clusters:
                self._create_obstacle(cluster_centers_world[j], cluster, agv_pos, agv_vel)

        self._remove_old_obstacles()

    def _update_obstacle(self, idx: int, center_world: np.ndarray, 
                        cluster: List[LidarPoint], agv_pos: np.ndarray, agv_vel: np.ndarray):
        obs = self.obstacles[idx]

        if obs.id not in self.kalman_filters:
            self.kalman_filters[obs.id] = ExtendedKalmanFilterCV(self.dt)

        # ⭐ Kalman با موقعیت world frame
        self.kalman_filters[obs.id].update(center_world)
        pos_world, vel_world = self.kalman_filters[obs.id].get_state()

        # محاسبه d_eq و d_dot
        result = self._compute_hysdg_esd(pos_world, vel_world, agv_pos, agv_vel, obs.d_eq)

        obs.center = pos_world
        obs.velocity = vel_world
        obs.points = cluster
        obs.d_eq = result['d_eq']
        obs.d_dot = result['d_dot']
        
        # ذخیره تاریخچه
        vel_magnitude = np.linalg.norm(vel_world)
        obs.velocity_history.append(vel_magnitude)
        if len(obs.velocity_history) > 20:  # افزایش history
            obs.velocity_history.pop(0)
        
        obs.state_history.append(result['state'])
        if len(obs.state_history) > 20:
            obs.state_history.pop(0)
        
        # تصمیم‌گیری
        obs.state = self._advanced_classification(obs)
        
        obs.last_seen = self.current_time
        obs.confidence = min(1.0, obs.confidence + 0.1)

    def _advanced_classification(self, obs: Obstacle) -> ObstacleState:
        """
        ⭐⭐⭐ الگوریتم تصمیم‌گیری محافظه‌کارانه با فیلتر میانگین متحرک
        """
        if len(obs.velocity_history) < 8:
            return ObstacleState.UNKNOWN
        
        # آمار سرعت
        recent_velocities = obs.velocity_history[-15:]  # 1.5 ثانیه اخیر
        avg_velocity = np.mean(recent_velocities)
        std_velocity = np.std(recent_velocities)
        max_velocity = np.max(recent_velocities)
        min_velocity = np.min(recent_velocities)
        
        # رای‌گیری از تاریخچه
        if len(obs.state_history) >= 8:
            recent_history = obs.state_history[-15:]
            static_count = sum(1 for s in recent_history if s == ObstacleState.STATIC)
            dynamic_count = sum(1 for s in recent_history if s == ObstacleState.DYNAMIC)
        else:
            static_count = 0
            dynamic_count = 0
        
        # ⭐⭐⭐ تصمیم‌گیری چند سطحی
        
        # سطح 1: قطعاً STATIC
        if (avg_velocity < 0.12 and           # میانگین بسیار کم
            max_velocity < 0.20 and           # حداکثر هم کم
            std_velocity < 0.10 and           # پایداری بالا
            min_velocity < 0.15):             # حداقل هم کم
            return ObstacleState.STATIC
        
        # سطح 2: قطعاً DYNAMIC
        elif (avg_velocity > 0.40 or          # میانگین بالا
              max_velocity > 0.60 or          # حداکثر بالا
              (avg_velocity > 0.25 and std_velocity > 0.15)):  # سرعت متوسط با نوسان
            return ObstacleState.DYNAMIC
        
        # سطح 3: استفاده از رای‌گیری
        elif static_count > dynamic_count + 7:
            return ObstacleState.STATIC
        elif dynamic_count > static_count + 7:
            return ObstacleState.DYNAMIC
        
        # سطح 4: محافظه‌کارانه - بر اساس میانگین
        elif avg_velocity < 0.18:
            return ObstacleState.STATIC
        else:
            return ObstacleState.DYNAMIC

    def _create_obstacle(self, center_world: np.ndarray, cluster: List[LidarPoint],
                        agv_pos: np.ndarray, agv_vel: np.ndarray):
        kf = ExtendedKalmanFilterCV(self.dt)
        kf.update(center_world)
        pos_world, vel_world = kf.get_state()

        self.kalman_filters[self.next_id] = kf
        result = self._compute_hysdg_esd(pos_world, vel_world, agv_pos, agv_vel, None)

        obstacle = Obstacle(
            id=self.next_id,
            center=pos_world,
            velocity=vel_world,
            points=cluster,
            state=ObstacleState.UNKNOWN,
            d_eq=result['d_eq'],
            d_dot=result['d_dot'],
            confidence=0.2,
            last_seen=self.current_time,
            state_history=[result['state']],
            velocity_history=[np.linalg.norm(vel_world)]
        )

        self.obstacles.append(obstacle)
        self.next_id += 1

    def _compute_hysdg_esd(self, pos_world: np.ndarray, vel_world: np.ndarray, 
                          agv_pos: np.ndarray, agv_vel: np.ndarray, 
                          prev_d_eq: Optional[float], lambda_esd: float = 1.0):
        """
        محاسبه HySDG-ESD
        """
        r_t = pos_world - agv_pos
        u_t = vel_world - agv_vel
        
        d = np.linalg.norm(r_t)
        u_norm = np.linalg.norm(u_t)

        if u_norm < 1e-6:
            d_eq = d
        else:
            d_eq = d - lambda_esd * np.dot(r_t, u_t) / u_norm

        if prev_d_eq is None:
            d_dot = 0.0
        else:
            d_dot = (d_eq - prev_d_eq) / self.dt

        # state برای history
        if u_norm < 0.20 and abs(d_dot) < 0.15:
            state = ObstacleState.STATIC
        else:
            state = ObstacleState.DYNAMIC

        return {'d_eq': d_eq, 'd_dot': d_dot, 'state': state}

    def _remove_old_obstacles(self):
        self.obstacles = [obs for obs in self.obstacles 
                         if self.current_time - obs.last_seen <= self.max_age]
        active_ids = {obs.id for obs in self.obstacles}
        self.kalman_filters = {k: v for k, v in self.kalman_filters.items() 
                              if k in active_ids}

    def get_obstacles(self):
        return self.obstacles

# =============================================================================
# Main System
# =============================================================================

class AGVObstacleDetectionSystem:
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.lidar_processor = LidarProcessor()
        self.tracker = MultiObjectTracker(dt)

    def process_scan(self, ranges, angles, agv_pos, agv_vel, agv_heading):
        """
        ⭐⭐⭐ حالا agv_heading هم لازم است
        """
        points = self.lidar_processor.parse_scan(ranges, angles)
        clusters = self.lidar_processor.cluster_points(points)
        self.tracker.update(clusters, agv_pos, agv_vel, agv_heading)
        return self.tracker.get_obstacles()

    def get_critical_obstacles(self, safety_distance=2.0):
        return [obs for obs in self.tracker.get_obstacles() 
                if obs.d_eq < safety_distance and obs.confidence > 0.5]

    def get_dynamic_obstacles(self):
        return [obs for obs in self.tracker.get_obstacles() 
                if obs.state == ObstacleState.DYNAMIC]