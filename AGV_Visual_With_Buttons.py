import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrow, Wedge, Rectangle
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection
from typing import List, Tuple
from collections import deque
sys.stdout.reconfigure(encoding='utf-8')

from HySDG_ESD_Kalman_V4 import AGVObstacleDetectionSystem

# =========================
# LiDAR Simulator
# =========================
class LidarSimulator:
    def __init__(self, fov: float = 270.0, numrays: int = 250, maxrange: float = 30, noisestd: float = 0.03):
        self.fov = np.deg2rad(fov)
        self.numrays = numrays
        self.maxrange = maxrange
        self.noisestd = noisestd
        startangle = -self.fov / 2
        endangle = self.fov / 2
        self.angles = np.linspace(startangle, endangle, numrays)

    def scan(self, obstacles: List[dict], agvpos: np.ndarray, agv_heading: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        اسکن LiDAR با در نظر گرفتن جهت AGV
        agv_heading: زاویه AGV نسبت به محور X (رادیان)
        """
        ranges = np.full(self.numrays, self.maxrange)
        
        for angleidx, angle in enumerate(self.angles):
            # زاویه مطلق = زاویه AGV + زاویه نسبی LiDAR
            absolute_angle = agv_heading + angle
            raydir = np.array([np.cos(absolute_angle), np.sin(absolute_angle)])
            mindist = self.maxrange
            
            for obs in obstacles:
                obscenter = obs['center']
                obsradius = obs.get('radius', 0.3)
                relpos = obscenter - agvpos
                proj = np.dot(relpos, raydir)
                
                if proj > 0:
                    closestpoint = proj * raydir
                    disttocenter = np.linalg.norm(relpos - closestpoint)
                    if disttocenter <= obsradius:
                        dist = proj - np.sqrt(obsradius**2 - disttocenter**2)
                        if 0 < dist < mindist:
                            mindist = dist
            ranges[angleidx] = mindist
        
        ranges += np.random.normal(0, self.noisestd, ranges.shape)
        ranges = np.clip(ranges, 0.1, self.maxrange)
        return ranges, self.angles

# =========================
# AGV Controller با حرکت تصادفی
# =========================
class RandomPathAGV:
    def __init__(self, start_pos: np.ndarray, speed: float = 1.5):
        self.pos = start_pos.copy()
        self.speed = speed
        self.heading = 0.0  # زاویه حرکت (رادیان)
        self.vel = np.array([speed, 0.0])
        
        # پارامترهای حرکت تصادفی
        self.target_heading = 0.0
        self.change_direction_timer = 0
        self.change_direction_interval = np.random.randint(30, 80)  # هر 3-8 ثانیه
        
    def update(self, dt: float, world_bounds: tuple):
        """
        به‌روزرسانی موقعیت AGV با حرکت تصادفی
        world_bounds: (x_min, x_max, y_min, y_max)
        """
        self.change_direction_timer += 1
        
        # تغییر جهت تصادفی
        if self.change_direction_timer >= self.change_direction_interval:
            self.target_heading = np.random.uniform(-np.pi/3, np.pi/3)  # ±60 درجه
            self.change_direction_interval = np.random.randint(30, 80)
            self.change_direction_timer = 0
        
        # تغییر نرم جهت (smooth steering)
        heading_diff = self.target_heading - self.heading
        # محدود کردن به [-π, π]
        if heading_diff > np.pi:
            heading_diff -= 2 * np.pi
        elif heading_diff < -np.pi:
            heading_diff += 2 * np.pi
        
        # تغییر نرم با نرخ چرخش محدود
        max_turn_rate = 0.05  # رادیان در فریم
        self.heading += np.clip(heading_diff, -max_turn_rate, max_turn_rate)
        
        # محاسبه سرعت بر اساس جهت
        self.vel = np.array([
            self.speed * np.cos(self.heading),
            self.speed * np.sin(self.heading)
        ])
        
        # به‌روزرسانی موقعیت
        new_pos = self.pos + self.vel * dt
        
        # بررسی برخورد با دیوارها و برگشت
        x_min, x_max, y_min, y_max = world_bounds
        
        if new_pos[0] < x_min or new_pos[0] > x_max:
            self.heading = np.pi - self.heading  # بازتاب افقی
            self.target_heading = self.heading
            new_pos[0] = np.clip(new_pos[0], x_min, x_max)
        
        if new_pos[1] < y_min or new_pos[1] > y_max:
            self.heading = -self.heading  # بازتاب عمودی
            self.target_heading = self.heading
            new_pos[1] = np.clip(new_pos[1], y_min, y_max)
        
        self.pos = new_pos
        return self.pos, self.vel, self.heading

# =========================
# Global simulation state
# =========================
dt = 0.1
steps = 600  # 60 ثانیه
current_scenario = 1

# محیط شبیه‌سازی
WORLD_BOUNDS = (-2, 30, -6, 6)

# AGV متحرک
agv = RandomPathAGV(start_pos=np.array([-1.0, 0.0]), speed=1.5)

system = AGVObstacleDetectionSystem(dt)
lidarsim = LidarSimulator(maxrange=30.0, numrays=250)

# Scenario states
obs1pos = obs2pos = obs3pos = None
obs1vel = obs2vel = obs3vel = None
static_obs = []

# Trajectory tracking
trajectories = {}
agv_trajectory = deque(maxlen=100)  # مسیر طی شده AGV
lidar_history = deque(maxlen=30)

def reset_scenario(scn):
    global system, current_scenario, trajectories, lidar_history, agv
    global obs1pos, obs2pos, obs3pos, obs1vel, obs2vel, obs3vel, static_obs

    current_scenario = scn
    system = AGVObstacleDetectionSystem(dt)
    trajectories = {}
    lidar_history.clear()
    agv_trajectory.clear()
    
    # بازنشانی AGV به موقعیت اولیه
    agv = RandomPathAGV(start_pos=np.array([-1.0, 0.0]), speed=1.5)

    if scn == 1:
        # سناریو 1: موانع استاتیک پراکنده
        static_obs = [
            {'center': np.array([5.0, 0.0]), 'radius': 0.5},
            {'center': np.array([8.0, 3.0]), 'radius': 0.4},
            {'center': np.array([12.0, -2.5]), 'radius': 0.45},
            {'center': np.array([15.0, 1.5]), 'radius': 0.5},
            {'center': np.array([18.0, -3.0]), 'radius': 0.4},
            {'center': np.array([22.0, 2.0]), 'radius': 0.5},
            {'center': np.array([25.0, -1.0]), 'radius': 0.45},
            {'center': np.array([10.0, 4.5]), 'radius': 0.4},
            {'center': np.array([20.0, -4.5]), 'radius': 0.5}
        ]

    elif scn == 2:
        # سناریو 2: موانع دینامیک با مسیرهای مختلف
        obs1pos = np.array([20.0, 0.0])
        obs1vel = np.array([-1.2, 0.0])  # حرکت به چپ
        
        obs2pos = np.array([10.0, 5.0])
        obs2vel = np.array([0.0, -0.8])  # حرکت به پایین
        
        obs3pos = np.array([25.0, -4.0])
        obs3vel = np.array([-0.6, 0.7])  # حرکت مورب

    elif scn == 3:
        # سناریو 3: ترکیبی - استاتیک + دینامیک
        static_obs = [
            {'center': np.array([10.0, 2.5]), 'radius': 0.5},
            {'center': np.array([18.0, -2.0]), 'radius': 0.45},
            {'center': np.array([24.0, 1.0]), 'radius': 0.4},
            {'center': np.array([14.0, -3.5]), 'radius': 0.5}
        ]
        
        obs1pos = np.array([22.0, -3.0])
        obs1vel = np.array([-0.8, 0.5])
        
        obs2pos = np.array([12.0, 4.5])
        obs2vel = np.array([0.3, -0.6])

# =========================
# Visualization Setup
# =========================
plt.style.use('fast')
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor('#F5F5F5')

# Axes setup - با محدوده گسترده‌تر
ax_main = fig.add_axes([0.05, 0.52, 0.50, 0.43])
ax_main.set_xlim(WORLD_BOUNDS[0], WORLD_BOUNDS[1])
ax_main.set_ylim(WORLD_BOUNDS[2], WORLD_BOUNDS[3])
ax_main.set_aspect('equal', adjustable='box')
ax_main.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax_main.set_xlabel('X (m)', fontsize=10, fontweight='bold')
ax_main.set_ylabel('Y (m)', fontsize=10, fontweight='bold')

ax_lidar = fig.add_axes([0.60, 0.52, 0.35, 0.43], projection='polar')
ax_lidar.set_theta_zero_location('E')
ax_lidar.set_ylim(0, 16)
ax_lidar.grid(True, alpha=0.4, linestyle=':', linewidth=1)
ax_lidar.tick_params(labelsize=8)

ax_deq = fig.add_axes([0.05, 0.08, 0.50, 0.38])
ax_deq.set_xlabel('Time (s)', fontsize=9, fontweight='bold')
ax_deq.set_ylabel('d_eq (m)', fontsize=9, fontweight='bold')
ax_deq.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
ax_deq.tick_params(labelsize=8)

ax_info = fig.add_axes([0.60, 0.24, 0.35, 0.22])
ax_info.axis('off')
ax_info.set_xlim(0, 1)
ax_info.set_ylim(0, 1)

# Buttons
btn_width = 0.30
btn_height = 0.055
btn_left = 0.63
btn_spacing = 0.012

ax_btn1 = fig.add_axes([btn_left, 0.08, btn_width, btn_height])
ax_btn2 = fig.add_axes([btn_left, 0.08 + btn_height + btn_spacing, btn_width, btn_height])
ax_btn3 = fig.add_axes([btn_left, 0.08 + 2*(btn_height + btn_spacing), btn_width, btn_height])

btn1 = Button(ax_btn1, 'Scenario 1 (Static)', color='#C8E6C9', hovercolor='#81C784')
btn2 = Button(ax_btn2, 'Scenario 2 (Dynamic)', color='#FFECB3', hovercolor='#FFD54F')
btn3 = Button(ax_btn3, 'Scenario 3 (Mixed)', color='#B3E5FC', hovercolor='#4FC3F7')

for btn in [btn1, btn2, btn3]:
    btn.label.set_fontsize(9)
    btn.label.set_fontweight('bold')

# Data storage
time_data = deque(maxlen=200)
deq_data = {}

# =========================
# Animation function
# =========================
def animate(frame):
    global obs1pos, obs2pos, obs3pos, time_data, deq_data, trajectories

    current_time = frame * dt
    time_data.append(current_time)

    # ===== به‌روزرسانی AGV با حرکت تصادفی =====
    agv_pos, agv_vel, agv_heading = agv.update(dt, WORLD_BOUNDS)
    agv_trajectory.append(agv_pos.copy())

    # ===== به‌روزرسانی موانع =====
    obstacles = []

    if current_scenario == 1:
        obstacles = static_obs

    elif current_scenario == 2:
        obs1pos[:] = obs1pos + obs1vel * dt
        obs2pos[:] = obs2pos + obs2vel * dt
        obs3pos[:] = obs3pos + obs3vel * dt
        
        # برگشت از دیوارها
        if obs1pos[0] < WORLD_BOUNDS[0] or obs1pos[0] > WORLD_BOUNDS[1]:
            obs1vel[0] *= -1
        if obs2pos[1] < WORLD_BOUNDS[2] or obs2pos[1] > WORLD_BOUNDS[3]:
            obs2vel[1] *= -1
        if obs3pos[0] < WORLD_BOUNDS[0] or obs3pos[0] > WORLD_BOUNDS[1]:
            obs3vel[0] *= -1
        if obs3pos[1] < WORLD_BOUNDS[2] or obs3pos[1] > WORLD_BOUNDS[3]:
            obs3vel[1] *= -1
        
        obstacles = [
            {'center': obs1pos.copy(), 'radius': 0.4},
            {'center': obs2pos.copy(), 'radius': 0.35},
            {'center': obs3pos.copy(), 'radius': 0.4}
        ]

    elif current_scenario == 3:
        obs1pos[:] = obs1pos + obs1vel * dt
        obs2pos[:] = obs2pos + obs2vel * dt
        
        if obs1pos[0] < WORLD_BOUNDS[0] or obs1pos[0] > WORLD_BOUNDS[1]:
            obs1vel[0] *= -1
        if obs1pos[1] < WORLD_BOUNDS[2] or obs1pos[1] > WORLD_BOUNDS[3]:
            obs1vel[1] *= -1
        if obs2pos[0] < WORLD_BOUNDS[0] or obs2pos[0] > WORLD_BOUNDS[1]:
            obs2vel[0] *= -1
        if obs2pos[1] < WORLD_BOUNDS[2] or obs2pos[1] > WORLD_BOUNDS[3]:
            obs2vel[1] *= -1
        
        obstacles = static_obs + [
            {'center': obs1pos.copy(), 'radius': 0.35},
            {'center': obs2pos.copy(), 'radius': 0.3}
        ]

    # ===== LiDAR scan با موقعیت، سرعت و جهت AGV =====
    ranges, angles = lidarsim.scan(obstacles, agv_pos, agv_heading)
    detected = system.process_scan(ranges, angles, agv_pos, agv_vel, agv_heading)
    
    lidar_history.append((ranges.copy(), angles.copy()))

    # ===== Panel 1: Main Scene =====
    ax_main.clear()
    ax_main.set_xlim(WORLD_BOUNDS[0], WORLD_BOUNDS[1])
    ax_main.set_ylim(WORLD_BOUNDS[2], WORLD_BOUNDS[3])
    ax_main.set_aspect('equal', adjustable='box')
    ax_main.set_xlabel('X (m)', fontsize=10, fontweight='bold')
    ax_main.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
    ax_main.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # رسم مرزهای محیط
    border = Rectangle((WORLD_BOUNDS[0], WORLD_BOUNDS[2]), 
                       WORLD_BOUNDS[1]-WORLD_BOUNDS[0], 
                       WORLD_BOUNDS[3]-WORLD_BOUNDS[2],
                       fill=False, edgecolor='black', linewidth=2, linestyle='--')
    ax_main.add_patch(border)

    # FOV LiDAR با در نظر گرفتن جهت AGV
    fov_angle = 270
    fov_start_abs = np.rad2deg(agv_heading) - 135
    wedge = Wedge(agv_pos, 15, fov_start_abs, fov_start_abs + fov_angle, 
                  fc='lightyellow', ec='orange', alpha=0.12, lw=1.5)
    ax_main.add_patch(wedge)

    # رسم اشعه LiDAR با LineCollection
    step = max(1, len(ranges)//25)
    lines = []
    colors = []
    for i in range(0, len(ranges), step):
        r, a = ranges[i], angles[i]
        if r < 29.5:
            absolute_angle = agv_heading + a
            endpoint = agv_pos + r * np.array([np.cos(absolute_angle), np.sin(absolute_angle)])
            lines.append([agv_pos, endpoint])
            if r < 2.0:
                colors.append((1, 0, 0, 0.15))
            elif r < 5.0:
                colors.append((0, 1, 0, 0.08))
            else:
                colors.append((0, 0, 1, 0.08))
    
    if lines:
        lc = LineCollection(lines, colors=colors, linewidths=0.8)
        ax_main.add_collection(lc)

    # رسم مسیر طی شده AGV
    if len(agv_trajectory) > 2:
        traj = np.array(agv_trajectory)
        ax_main.plot(traj[:, 0], traj[:, 1], 'b--', alpha=0.4, linewidth=2, label='AGV Path')

    # رسم AGV با جهت صحیح
    agv_circle = Circle(agv_pos, 0.5, fc='dodgerblue', ec='navy', lw=2.5, zorder=10)
    ax_main.add_patch(agv_circle)
    
    # فلش جهت AGV
    arrow_length = 0.8
    arrow = FancyArrow(agv_pos[0], agv_pos[1], 
                      arrow_length * np.cos(agv_heading), 
                      arrow_length * np.sin(agv_heading),
                      width=0.25, head_width=0.5, head_length=0.35,
                      fc='white', ec='navy', lw=2.5, zorder=11)
    ax_main.add_patch(arrow)
    
    # دایره ایمنی
    ax_main.add_patch(Circle(agv_pos, 2.0, fc='none', ec='red', 
                            ls='--', lw=1.5, alpha=0.4, zorder=5))

    # رسم موانع واقعی
    for obs in obstacles:
        ax_main.add_patch(Circle(obs['center'], obs['radius'], 
                                fc='crimson', ec='darkred', alpha=0.5, lw=2.5))

    # رسم موانع تشخیص داده شده
    static_count = dynamic_count = 0
    
    for obs in detected:
        if not hasattr(obs.state, 'value'):
            continue

        state_str = obs.state.value
        obs_id = obs.id

        if obs_id not in trajectories:
            trajectories[obs_id] = deque(maxlen=40)
        trajectories[obs_id].append(obs.center.copy())

        if 'STATIC' in state_str:
            color, edge_color = 'limegreen', 'darkgreen'
            static_count += 1
        elif 'DYNAMIC' in state_str:
            color, edge_color = 'gold', 'darkorange'
            dynamic_count += 1
        else:
            continue

        ax_main.add_patch(Circle(obs.center, 0.32, fc=color, ec=edge_color, 
                                lw=2.5, alpha=0.85, zorder=8))
        
        if 'DYNAMIC' in state_str and np.linalg.norm(obs.velocity) > 0.08:
            vel_scale = 2.5
            arrow = FancyArrow(obs.center[0], obs.center[1],
                              obs.velocity[0] * vel_scale, obs.velocity[1] * vel_scale,
                              width=0.1, head_width=0.25, head_length=0.2,
                              fc='purple', ec='indigo', lw=2, alpha=0.8, zorder=9)
            ax_main.add_patch(arrow)

        if frame % 5 == 0 and len(trajectories[obs_id]) > 3:
            traj = np.array(trajectories[obs_id])
            ax_main.plot(traj[:, 0], traj[:, 1], '--', color=edge_color, 
                        alpha=0.6, lw=1.5, zorder=5)

        vel_norm = np.linalg.norm(obs.velocity)  # حالا این همان سرعت world است!
        label_text = f'ID:{obs_id}\n{state_str[:3]}\nv:{vel_norm:.2f}m/s'
        ax_main.text(obs.center[0], obs.center[1] + 0.7, label_text, 
                    ha='center', va='bottom', fontsize=7.5,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=edge_color, alpha=0.95, lw=2), zorder=12)

        if obs_id not in deq_data:
            deq_data[obs_id] = deque(maxlen=100)
        deq_data[obs_id].append(obs.d_eq)

    ax_main.set_title(f'AGV Moving System | Scenario {current_scenario} | Frame {frame} | Time: {current_time:.1f}s\n' +
                     f'AGV Pos: ({agv_pos[0]:.1f}, {agv_pos[1]:.1f}) | Vel: {np.linalg.norm(agv_vel):.2f} m/s | ' +
                     f'Detected: {len(detected)} (S:{static_count} D:{dynamic_count})',
                     fontsize=10, fontweight='bold', pad=6)

    # ===== Panel 2: LiDAR =====
    if frame % 3 == 0:
        ax_lidar.clear()
        ax_lidar.set_theta_zero_location('E')
        ax_lidar.set_ylim(0, 16)
        ax_lidar.grid(True, alpha=0.4, linestyle=':', linewidth=1)
        ax_lidar.tick_params(labelsize=8)
        ax_lidar.set_title('LIDAR POLAR VIEW', fontsize=11, fontweight='bold', pad=10)
        
        if len(lidar_history) > 0:
            current_ranges, current_angles = lidar_history[-1]
            step = 3
            ax_lidar.scatter(current_angles[::step], current_ranges[::step], 
                           c=current_ranges[::step], cmap='RdYlGn_r', s=10, alpha=0.7, 
                           vmin=0, vmax=15, edgecolors='none')
            ax_lidar.plot(current_angles[::step], current_ranges[::step], 
                         'b-', alpha=0.25, linewidth=0.8)
            
            danger_indices = current_ranges < 2.0
            if np.any(danger_indices):
                ax_lidar.scatter(current_angles[danger_indices], current_ranges[danger_indices], 
                               c='red', s=20, marker='o', alpha=0.9, 
                               edgecolors='darkred', linewidths=1.2, zorder=10)
        
        ax_lidar.set_rticks([4, 8, 12, 16])

    # ===== Panel 3: Distance chart =====
    if frame % 5 == 0:
        ax_deq.clear()
        ax_deq.set_xlabel('Time (s)', fontsize=9, fontweight='bold')
        ax_deq.set_ylabel('d_eq (m)', fontsize=9, fontweight='bold')
        ax_deq.set_title('Distance Metrics', fontsize=10, fontweight='bold', pad=6)
        ax_deq.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
        ax_deq.tick_params(labelsize=8)
        
        times_list = list(time_data)
        for obs_id, values_deque in deq_data.items():
            values = list(values_deque)
            if len(values) > 0:
                times = times_list[-len(values):]
                ax_deq.plot(times, values, '-', label=f'Obs {obs_id}', 
                           linewidth=1.5, alpha=0.85)
        
        if deq_data:
            ax_deq.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.9)
        
        ax_deq.set_xlim(max(0, current_time - 10), max(10, current_time))
        ax_deq.axhline(y=2.0, color='r', linestyle='--', linewidth=1.5, alpha=0.6)
        
        if deq_data:
            all_values = [v for values in deq_data.values() for v in values]
            if all_values:
                y_max = max(all_values)
                ax_deq.set_ylim(-0.5, max(y_max + 1, 3))

    # ===== Panel 4: Info =====
    if frame % 10 == 0:
        ax_info.clear()
        ax_info.axis('off')
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        
        ax_info.text(0.5, 0.95, 'SYSTEM STATUS', ha='center', va='top', 
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#FCE4EC', 
                             edgecolor='#C2185B', linewidth=1.5, alpha=0.95))
        
        info_lines = [
            f"Time: {current_time:.1f}s  |  Frame: {frame}/{steps}",
            f"AGV: ({agv_pos[0]:.1f}, {agv_pos[1]:.1f}) | V={np.linalg.norm(agv_vel):.2f}m/s",
            f"Heading: {np.rad2deg(agv_heading):.1f}deg",
            f"Scenario: {current_scenario}  |  Detected: {len(detected)}",
            f"Static: {static_count}  |  Dynamic: {dynamic_count}",
            "=" * 42
        ]
        
        y_pos = 0.78
        for line in info_lines:
            ax_info.text(0.05, y_pos, line, ha='left', va='top', 
                        fontsize=7.5, family='monospace')
            y_pos -= 0.10
        
        if detected:
            ax_info.text(0.05, y_pos, 'CLOSEST:', ha='left', va='top', 
                        fontsize=8, fontweight='bold')
            y_pos -= 0.09
            
            sorted_obs = sorted(detected, key=lambda x: x.d_eq)[:2]
            for i, obs in enumerate(sorted_obs, 1):
                state_mark = '[S]' if obs.state.value == 'STATIC' else '[D]'
                info_line = f"#{i} {state_mark} ID={obs.id} d={obs.d_eq:.2f}m v={np.linalg.norm(obs.velocity):.2f}m/s"
                ax_info.text(0.08, y_pos, info_line, ha='left', va='top', 
                            fontsize=7, family='monospace')
                y_pos -= 0.09

# =========================
# Button callbacks
# =========================
def on_click_1(event):
    global time_data, deq_data
    time_data.clear()
    deq_data = {}
    reset_scenario(1)

def on_click_2(event):
    global time_data, deq_data
    time_data.clear()
    deq_data = {}
    reset_scenario(2)

def on_click_3(event):
    global time_data, deq_data
    time_data.clear()
    deq_data = {}
    reset_scenario(3)

btn1.on_clicked(on_click_1)
btn2.on_clicked(on_click_2)
btn3.on_clicked(on_click_3)

# =========================
# Start
# =========================
print("="*80)
print("AGV MOVING OBSTACLE DETECTION SYSTEM - FINAL VERSION WITH ROTATION")
print("="*80)
print("CRITICAL FIXES:")
print("  [+++] Rotation-aware coordinate transformation")
print("  [+++] Uses rotation matrix for AGV heading changes")
print("  [+++] Extended Kalman Filter with adaptive velocity damping")
print("  [+++] Multi-level classification with moving average filter")
print("="*80)
print("HOW IT WORKS:")
print("  1. LiDAR detects obstacle at (x, y) relative to AGV")
print("  2. Transform: pos_world = agv_pos + R(heading) @ (x, y)")
print("  3. EKF tracks in world frame → correct velocity estimation")
print("  4. Adaptive damping: low innovation → assume STATIC")
print("  5. Multi-level classification with conservative thresholds")
print("="*80)
print("KEY IMPROVEMENTS:")
print("  • Rotation matrix handles AGV direction changes")
print("  • Innovation-based adaptive velocity damping")
print("  • Extended history (20 frames = 2 seconds)")
print("  • 4-level classification algorithm")
print("  • Conservative thresholds: default to STATIC when uncertain")
print("="*80)

reset_scenario(1)
ani = animation.FuncAnimation(fig, animate, frames=steps, interval=100, repeat=True)
plt.show()