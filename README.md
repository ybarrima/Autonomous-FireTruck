# 🚒 Autonomous Fire Truck — Deep Reinforcement Learning in CARLA

An autonomous fire truck navigation system trained using Deep Reinforcement Learning (DRL) in the [CARLA Simulator](https://carla.org/). The agent learns to drive a fire truck through urban environments using LiDAR point clouds, semantic camera images, and bird's-eye-view observations — built entirely from scratch with a custom OpenAI Gym wrapper, without relying on pre-built RL APIs.

> **B.Sc. Graduation Project** — Arab Academy for Science, Technology and Maritime Transport (AASTMT), Computer Engineering Department, July 2023.

---

## Overview

Firefighting is inherently dangerous, and deploying autonomous vehicles to navigate toward fire sites can reduce risk to human life. This project develops a full autonomous driving pipeline for a fire truck that can:

- **Navigate autonomously** through urban traffic using Deep RL (TD3 / DDPG via Stable-Baselines3).
- **Perceive its environment** through semantic segmentation cameras, LiDAR sensors, and a bird's-eye-view renderer.
- **Follow planned routes** with a built-in waypoint planner and traffic-light / vehicle hazard detection.
- **Integrate with ROS** for real-world deployment on an Nvidia Jetson TX2 with RPLidar and depth camera hardware.

The RL agent was evaluated across 500k+ timesteps using PPO, DDPG, and TD3 in both OpenAI Gym's CarRacing environment (for rapid prototyping) and the custom CARLA-Gym wrapper (for realistic simulation).

---

## Repository Structure

```
Autonomous-FireTruck/
│
├── carla_gym_wrapper/           # Custom OpenAI Gym environment for CARLA
│   ├── carla_env.py             # CarlaEnv class — Gym-compatible CARLA environment
│   └── birdeye_render.py        # Bird's-eye view renderer, route planner, map rendering
│
├── trials/                      # Early development scripts (initial experiments)
│   ├── GatheringData.py         # Data collection from CARLA sensors
│   ├── ReinforcementTrial.py    # First RL training experiments
│   ├── SensorTrial.py           # Sensor integration testing
│   └── VisualizingData.py       # Visualization of collected data
│
├── notebooks/
│   └── CheckCarlaEnv.ipynb      # Interactive notebook for testing the Gym wrapper
│
└── README.md
```

---

## CARLA-Gym Wrapper

The core of this project is a **custom OpenAI Gym wrapper** (`carla_gym_wrapper/`) that bridges CARLA and Stable-Baselines3, enabling standard RL training loops on the simulator. This was built from scratch rather than using existing wrappers to give full control over the observation space, reward shaping, and episode logic.

### Key Components

**`CarlaEnv` (carla_env.py)** — The main Gym environment class.

- Wraps CARLA as a `gym.Env` with proper `reset()`, `step()`, `render()`, and `seed()` methods.
- Spawns the ego vehicle (`vehicle.carlamotors.firetruck`), traffic, and pedestrians.
- Attaches collision, LiDAR, and semantic segmentation camera sensors.
- Runs in synchronous mode for deterministic training.

**Observation Space** — A dictionary with four channels:

| Key | Shape | Description |
|-----|-------|-------------|
| `camera` | `(256, 256, 3)` | Semantic segmentation (CityScapes palette) |
| `lidar` | `(256, 256, 3)` | 2D histogram of LiDAR point cloud with waypoint overlay |
| `state` | `(4,)` | `[lateral_distance, delta_yaw, speed, vehicle_front]` |

An optional PIXOR mode adds roadmap, vehicle classification, and regression maps for object detection research.

**Action Space** — Continuous `Box(2)`:

| Index | Range | Meaning |
|-------|-------|---------|
| 0 | `[-3.0, 3.0]` | Acceleration (positive → throttle, negative → brake) |
| 1 | `[-0.3, 0.3]` | Steering angle |

A discrete mode is also supported with configurable acceleration/steering bins.

**Reward Function:**

```
r = 200 * r_collision + 1 * longitudinal_speed + 10 * r_fast
    + 1 * r_out_of_lane + 5 * r_steering² + 0.2 * r_lateral_accel - 0.1
```

The reward encourages maintaining desired speed (8 m/s), staying in lane, smooth steering, and heavily penalizes collisions (-200).

**Episode Termination** — An episode ends when:
- The fire truck collides with anything.
- Maximum timesteps (200) are reached.
- The vehicle arrives at a destination.
- The vehicle goes too far out of lane (>2.0 m).

**`BirdeyeRender` (birdeye_render.py)** — Renders a top-down view of the simulation:

- Draws the full road network from CARLA's map topology.
- Renders vehicle and pedestrian bounding boxes with color-coded history trails.
- Overlays planned waypoints (blue for clear, purple when a red light is detected).
- Supports ego-centric rotation and clipping for the observation pipeline.

**`RoutePlanner`** — A waypoint-following route planner:

- Samples waypoints from CARLA's road network.
- Computes road options at intersections (straight, left, right).
- Detects traffic light states and vehicle hazards ahead.

---

## Configuration

The environment is configured via a parameters dictionary:

```python
params = {
    'number_of_vehicles': 20,
    'number_of_walkers': 0,
    'display_size': 256,
    'max_past_step': 1,
    'dt': 0.1,
    'discrete': False,
    'continuous_accel_range': [-3.0, 3.0],
    'continuous_steer_range': [-0.3, 0.3],
    'ego_vehicle_filter': 'vehicle.carlamotors.firetruck',
    'port': 20000,
    'town': 'Town02_Opt',
    'task_mode': 'random',       # 'random' or 'roundabout'
    'max_time_episode': 200,
    'max_waypt': 12,
    'obs_range': 32,             # meters
    'lidar_bin': 0.125,          # meters per bin
    'd_behind': 12,              # meters behind ego in observation
    'out_lane_thres': 2.0,       # meters
    'desired_speed': 8,          # m/s
    'max_ego_spawn_times': 200,
    'display_route': True,
    'pixor': False,
}
```

---

## Training

The agent is trained using **Stable-Baselines3** with either DDPG or TD3:

```python
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

# Create the environment
env = CarlaEnv(params=params)

# Add exploration noise
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions)
)

# Train
model = TD3('MultiInputPolicy', env,
            buffer_size=8000,
            action_noise=action_noise,
            tensorboard_log='./logs/',
            verbose=1)
model.learn(total_timesteps=100_000)
model.save('td3_firetruck')
```

---

## Prerequisites

- **CARLA Simulator** 0.9.x (tested with 0.9.13)
- **Python** 3.7
- **Dependencies:**
  ```
  carla
  pygame
  gym
  stable-baselines3
  numpy
  scikit-image
  open3d
  matplotlib
  torch
  ```

### Setup

1. Start the CARLA server:
   ```bash
   ./CarlaUE4.sh -quality-level=Low -carla-rpc-port=20000
   ```

2. Install dependencies:
   ```bash
   pip install stable-baselines3 pygame gym numpy scikit-image open3d
   ```

3. Run the environment check notebook or train directly:
   ```python
   from carla_gym_wrapper.carla_env import CarlaEnv
   env = CarlaEnv(params=params)
   obs = env.reset()
   ```

---

## Development Journey

The `trials/` directory contains the early experimental scripts from the initial development phase:

- **`GatheringData.py`** — First experiments connecting to CARLA and collecting sensor data.
- **`ReinforcementTrial.py`** — Initial RL training attempts before building the full Gym wrapper.
- **`SensorTrial.py`** — Testing LiDAR, camera, and collision sensor integration.
- **`VisualizingData.py`** — Visualizing point clouds and camera feeds from the simulator.

These scripts document the progression from raw CARLA API experimentation to the structured Gym-compatible environment.

---

## System Architecture

The full system (detailed in the thesis documentation) integrates several subsystems:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Autonomous Navigation | TD3 (DRL) in CARLA | Steering and throttle control |
| Perception | Semantic Segmentation Camera + LiDAR | Environment understanding |
| Image Segmentation | ResNet-DUC-HDC | Road/obstacle classification |
| Route Planning | Custom Waypoint Planner | Path following with hazard detection |
| Real-world Controller | ROS on Jetson TX2 | Hardware integration |
| Fire Detection | IoT Flame Sensors (ESP8266) | Trigger and locate fires |
| Mapping & Localization | SLAM + AMCL via ROS | Environment mapping |

---

## Results

- **OpenAI CarRacing** (prototyping): PPO achieved stable performance over 500k+ episodes, validating the training pipeline before moving to CARLA.
- **CARLA Simulator**: TD3 with the custom Gym wrapper demonstrated lane-following, traffic-aware navigation, and collision avoidance in Town02 with 20 surrounding vehicles.
- **ROS Integration**: Successfully tested with TurtleBot3 in Gazebo for mapping, navigation, and autonomous exploration before deploying on hardware.

---

## Team

| Name | Role |
|------|------|
| Shady Hisham Wagdy | Team Member |
| Mostafa Ayman Amin | Team Member |
| Mohamed Ashraf Fawzy | Team Member |
| Youssef Mohamed Barrima | Team Member |
| Youssef Mamdouh Darweesh | Team Member |

**Supervisor:** Dr. Mohamed Waleed Fakhr — AASTMT

---

## License

This project was developed as a B.Sc. graduation project at the Arab Academy for Science, Technology and Maritime Transport (AASTMT). The code is provided for educational and research purposes.

---

## Acknowledgments

- [CARLA Simulator](https://carla.org/) — Open-source autonomous driving simulator.
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — RL algorithm implementations.
- [OpenAI Gym](https://www.gymlibrary.dev/) — RL environment interface standard.
- Dr. Ibrahim Sobh — For guidance on RL and autonomous systems.
