import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, List, Dict
import os
import pygame


class SafetyAwareRobotEnv(gym.Env):
    """
    A 2D environment with a spherical robot navigating around obstacles.
    Enhanced with LIDAR-based obstacle detection.
    State space: [x, y, lidar_readings...]
    Action space: [vx, vy] (velocity commands)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        obstacles: Optional[List[Tuple[float, float, float]]] = None,
        max_episode_steps: int = 1000,
        n_lidar_rays: int = 16,
        lidar_max_range: float = 10.0,
    ):
        super().__init__()

        # Environment dimensions and constants
        self.max_pos = 6.0
        self.max_acc = 1.0
        self.current_vel = np.zeros(2)
        self.dt = 0.1
        self.robot_radius = 0.3

        # LIDAR configuration
        self.n_lidar_rays = n_lidar_rays
        self.lidar_max_range = lidar_max_range

        # Default obstacles if none provided: [(x, y, radius), ...]
        self.obstacles = obstacles or [
            (3.0, 3.0, 1.0),
            (-2.0, -2.0, 1.0),
            (4.0, -3.0, 0.8),
        ]

        # Enhanced state space: [x, y, lidar_readings...]
        obs_low = np.array([-self.max_pos, -self.max_pos] + [0.0] * n_lidar_rays)
        obs_high = np.array(
            [self.max_pos, self.max_pos] + [lidar_max_range] * n_lidar_rays
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        # Action space: [vx, vy]
        self.action_space = spaces.Box(
            low=-self.max_acc, high=self.max_acc, shape=(2,), dtype=np.float32
        )

        # Rendering setup
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.screen_width = 800
        self.screen_height = 800

        # Episode length limit
        self.max_episode_steps = max_episode_steps
        self.steps = 0

    def _get_lidar_readings(self) -> np.ndarray:
        """Simulate LIDAR readings by casting rays in different directions."""
        readings = []
        robot_pos = self.state[:2]

        for i in range(self.n_lidar_rays):
            angle = (2 * np.pi * i) / self.n_lidar_rays
            ray_dir = np.array([np.cos(angle), np.sin(angle)])

            # Find closest intersection with any obstacle
            min_distance = self.lidar_max_range

            for obs_x, obs_y, obs_r in self.obstacles:
                obs_pos = np.array([obs_x, obs_y])

                # Ray-circle intersection calculation
                a = np.dot(ray_dir, ray_dir)
                b = 2 * np.dot(ray_dir, robot_pos - obs_pos)
                c = np.dot(robot_pos - obs_pos, robot_pos - obs_pos) - obs_r**2

                discriminant = b**2 - 4 * a * c

                if discriminant >= 0:
                    # Ray intersects circle
                    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
                    t2 = (-b + np.sqrt(discriminant)) / (2 * a)

                    # Consider only intersections in front of the robot
                    if t1 > 0:
                        min_distance = min(min_distance, t1)
                    elif t2 > 0:
                        min_distance = min(min_distance, t2)

            readings.append(min_distance)

        return np.array(readings)

    def _get_observation(self) -> np.ndarray:
        """Combine robot state with LIDAR readings."""
        lidar_readings = self._get_lidar_readings()
        return np.concatenate([self.state, lidar_readings])

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.steps = 0

        # Initialize state randomly but away from obstacles
        while True:
            self.state = self.observation_space.sample()[:2]  # Only sample position
            if not self._is_collision(self.state):
                break

        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step."""
        self.steps += 1

        self.current_vel = action

        # New position = old position + action (velocity) * dt
        new_x = self.state[0] + self.current_vel[0] * self.dt
        new_y = self.state[1] + self.current_vel[1] * self.dt

        # Update state
        self.state = np.array([new_x, new_y])

        # Get full observation with LIDAR
        observation = self._get_observation()

        # Check termination conditions
        terminated = False
        truncated = False

        # Check if out of bounds
        if abs(self.state[0]) > self.max_pos or abs(self.state[1]) > self.max_pos:
            terminated = True

        # Check collision with obstacles
        if self._is_collision(self.state):
            terminated = True

        # Check episode length
        if self.steps >= self.max_episode_steps:
            truncated = True

        # Enhanced reward function using LIDAR readings
        if terminated:
            reward = 0.0
        else:
            reward = 1.0

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, {}

    def _is_collision(self, state: np.ndarray) -> bool:
        """Check if the robot collides with any obstacle or goes out of bounds."""
        robot_pos = state[:2]
        for obs_x, obs_y, obs_r in self.obstacles:
            distance = np.sqrt(
                (robot_pos[0] - obs_x) ** 2 + (robot_pos[1] - obs_y) ** 2
            )
            if distance < (self.robot_radius + obs_r):
                return True
        return False

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((255, 255, 255))

        # Convert coordinates from environment space to pixel space
        def convert_coord(x, y):
            screen_x = (x + self.max_pos) * self.screen_width / (2 * self.max_pos)
            screen_y = (self.max_pos - y) * self.screen_height / (2 * self.max_pos)
            return int(screen_x), int(screen_y)

        def convert_length(l):
            return int(l * self.screen_width / (2 * self.max_pos))

        # Draw obstacles
        for obs_x, obs_y, obs_r in self.obstacles:
            pos = convert_coord(obs_x, obs_y)
            radius = convert_length(obs_r)
            pygame.draw.circle(canvas, (200, 200, 200), pos, radius)

        # Draw robot
        robot_pos = convert_coord(self.state[0], self.state[1])
        robot_radius = convert_length(self.robot_radius)
        pygame.draw.circle(canvas, (255, 0, 0), robot_pos, robot_radius)

        # Draw LIDAR rays
        lidar_readings = self._get_lidar_readings()
        for i, distance in enumerate(lidar_readings):
            angle = (2 * np.pi * i) / self.n_lidar_rays
            end_x = self.state[0] + distance * np.cos(angle)
            end_y = self.state[1] + distance * np.sin(angle)
            end_pos = convert_coord(end_x, end_y)
            pygame.draw.line(canvas, (0, 255, 0), robot_pos, end_pos, 1)

        # Draw velocity vector
        vel_scale = 50
        pygame.draw.line(
            canvas,
            (0, 0, 255),
            robot_pos,
            (
                robot_pos[0] + int(self.current_vel[0] * vel_scale),
                robot_pos[1] - int(self.current_vel[1] * vel_scale),
            ),
            2,
        )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    # Create and test environment
    env = SafetyAwareRobotEnv(render_mode="human", n_lidar_rays=16)
    obs, info = env.reset()

    # Run for a few episodes
    for episode in range(3):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (terminated or truncated):
            action = env.action_space.sample()  # Random actions
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            print(f"Step: {steps}, Action: {action}, Reward: {reward:.2f}")

        print(
            f"Episode {episode + 1}: Steps: {steps}, Total Reward: {total_reward:.2f}"
        )

    env.close()
