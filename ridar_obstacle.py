import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import math
from datetime import datetime

class InertiaRacerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(InertiaRacerEnv, self).__init__()
        
        # --- 설정 ---
        self.SCREEN_SIZE = 800
        self.AGENT_RADIUS = 10
        self.TARGET_RADIUS = 15
        self.OBSTACLE_RADIUS = 30 
        
        self.MAX_SPEED = 200.0
        self.ACCEL_POWER = 2.0
        self.FRICTION = 0.92
        
        self.NUM_OBSTACLES = 5
        
        # [LIDAR 강화 유지]
        # 자동 회피 로직이 있더라도, LIDAR 해상도가 높으면 에이전트가 벽과의 거리를 미세조정하기 좋습니다.
        self.RAY_NUM = 32  # 15 -> 32
        self.RAY_ANGLES = np.linspace(-135, 135, self.RAY_NUM) 
        self.BASE_RAY_LENGTH = 200.0

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        obs_size = 5 + self.RAY_NUM
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # [상태 변수 복구] 자동 회피를 위한 변수들
        self.active_waypoint = None
        self.last_accel = np.array([0.0, 0.0])
        self.prev_action = np.array([0.0, 0.0])
        self.avoidance_timer = 0 
        self.last_blocking_obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([self.SCREEN_SIZE/2, self.SCREEN_SIZE/2], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.target = self._spawn_entity(padding=50)
        
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            self.obstacles.append(self._spawn_entity(padding=50, check_overlap=True))
        
        self.score = 0
        self.steps = 0
        self.max_steps = 1000
        self.prev_action = np.array([0.0, 0.0])
        
        # 회피 로직 초기화
        self.avoidance_timer = 0
        self.last_blocking_obs = None
        self.active_waypoint = self.target
        
        return self._get_obs(), {}

    def _spawn_entity(self, padding=50, check_overlap=False):
        while True:
            pos = np.random.uniform(padding, self.SCREEN_SIZE - padding, size=2).astype(np.float32)
            if not check_overlap: return pos
            
            safe = True
            if np.linalg.norm(pos - self.pos) < 150: safe = False
            if np.linalg.norm(pos - self.target) < 100: safe = False
            for obs in self.obstacles:
                if np.linalg.norm(pos - obs) < 80: safe = False
            if safe: return pos

    def _get_rays(self):
        speed = np.linalg.norm(self.vel)
        heading = math.atan2(self.vel[1], self.vel[0]) if speed > 1.0 else 0.0
        
        current_ray_length = self.BASE_RAY_LENGTH + (speed * 5.0) # 속도 비례 탐지 거리

        ray_readings = []
        self.current_rays_rendering = [] 

        for angle_deg in self.RAY_ANGLES:
            angle_rad = heading + math.radians(angle_deg)
            ray_dir = np.array([math.cos(angle_rad), math.sin(angle_rad)])
            
            min_dist = current_ray_length
            
            for obs in self.obstacles:
                to_obs = obs - self.pos
                proj = np.dot(to_obs, ray_dir)
                if proj > 0:
                    perp_dist = np.linalg.norm(to_obs - (ray_dir * proj))
                    if perp_dist < self.OBSTACLE_RADIUS:
                        hit_dist = proj - math.sqrt(self.OBSTACLE_RADIUS**2 - perp_dist**2)
                        if 0 < hit_dist < min_dist:
                            min_dist = hit_dist
            
            reading = min_dist / current_ray_length
            ray_readings.append(reading)
            self.current_rays_rendering.append((ray_dir, min_dist, reading))
            
        return np.array(ray_readings, dtype=np.float32)

    # [복구됨] 전방 장애물 감지 로직
    def _get_blocking_obstacle(self):
        check_vectors = []
        to_target = self.target - self.pos
        dist_target = np.linalg.norm(to_target)
        if dist_target > 0:
            check_vectors.append((to_target / dist_target, dist_target))
            
        speed = np.linalg.norm(self.vel)
        if speed > 10.0:
            prediction_dist = speed * 1.5 
            check_vectors.append((self.vel / speed, prediction_dist))

        closest_obs = None
        min_dist_to_obs = float('inf')

        for obs in self.obstacles:
            is_blocking = False
            for direction, max_dist in check_vectors:
                to_obs = obs - self.pos
                proj = np.dot(to_obs, direction)
                if proj <= 0 or proj >= max_dist: continue
                
                closest_point_on_line = direction * proj
                perp_dist = np.linalg.norm(to_obs - closest_point_on_line)
                
                # 충돌 예측 범위
                collision_threshold = self.OBSTACLE_RADIUS + self.AGENT_RADIUS + 20.0 
                if perp_dist < collision_threshold:
                    is_blocking = True
                    break 
            
            if is_blocking:
                dist_to_obs = np.linalg.norm(obs - self.pos)
                if dist_to_obs < min_dist_to_obs:
                    min_dist_to_obs = dist_to_obs
                    closest_obs = obs
                    
        return closest_obs

    # [복구됨] 우회 경로 계산 로직
    def _get_detour_point(self, blocking_obs):
        to_obs = blocking_obs - self.pos
        detour_offset = (self.OBSTACLE_RADIUS + self.AGENT_RADIUS) * 4.5 # 우회 반경 넉넉하게
        
        perp_vec = np.array([-to_obs[1], to_obs[0]])
        perp_vec = (perp_vec / (np.linalg.norm(perp_vec) + 1e-6)) * detour_offset
        
        to_real_target = self.target - self.pos
        target_dir = to_real_target / (np.linalg.norm(to_real_target) + 1e-6)
        forward_bias = target_dir * 70.0 
        
        waypoint1 = blocking_obs + perp_vec + forward_bias
        waypoint2 = blocking_obs - perp_vec + forward_bias
        
        ref_vec = self.target - self.pos
        if np.dot(ref_vec, waypoint1 - self.pos) > np.dot(ref_vec, waypoint2 - self.pos):
            return waypoint1
        else:
            return waypoint2

    def _get_obs(self, sensor_data=None):
        if sensor_data is None: sensor_data = self._get_rays()
        
        # 1. 장애물 감지 및 우회점 계산 (Auto-Logic)
        blocking_obs = self._get_blocking_obstacle()
        
        if blocking_obs is not None:
            self.last_blocking_obs = blocking_obs
            self.avoidance_timer = 20 
        
        if blocking_obs is None:
             self.avoidance_timer = 0
             self.last_blocking_obs = None

        if self.avoidance_timer > 0 and self.last_blocking_obs is not None:
            self.active_waypoint = self._get_detour_point(self.last_blocking_obs)
            self.avoidance_timer -= 1
        else:
            self.active_waypoint = self.target

        # 2. Observation 구성
        # 에이전트는 '계산된 우회점(active_waypoint)'을 목표로 인식하지만,
        # 동시에 'LIDAR(sensor_data)'를 통해 실제 물리적 거리를 파악합니다.
        to_objective = self.active_waypoint - self.pos
        dist_objective = np.linalg.norm(to_objective)
        scale = self.SCREEN_SIZE
        
        obs = np.concatenate([
            self.vel / self.MAX_SPEED,
            to_objective / scale,
            [dist_objective / scale],
            sensor_data # 32개의 강화된 LIDAR
        ])
        return obs.astype(np.float32)

    def step(self, action):
        self.steps += 1
        
        accel = np.array(action) * self.ACCEL_POWER
        self.last_accel = accel
        
        self.vel += accel
        speed = np.linalg.norm(self.vel)
        if speed > self.MAX_SPEED:
            self.vel = (self.vel / speed) * self.MAX_SPEED
        self.vel *= self.FRICTION
        self.pos += self.vel
        
        reward = -0.03
        terminated = False
        
        # 벽 페널티
        if self.pos[0] < 0 or self.pos[0] > self.SCREEN_SIZE or self.pos[1] < 0 or self.pos[1] > self.SCREEN_SIZE:
            self.pos = np.clip(self.pos, 0, self.SCREEN_SIZE)
            self.vel *= 0.5
            reward -= 5.0

        # LIDAR 데이터 갱신
        sensor_data = self._get_rays()
        
        # [LIDAR 반영 1] 근접 페널티 (Proximity Penalty)
        # 자동 회피 로직이 있더라도, 물리적으로 너무 가까워지면 벌점을 줍니다.
        # 이렇게 하면 '우회 경로'를 따르면서도 장애물에 스치지 않도록 '미세 조정'을 배웁니다.
        min_lidar_val = np.min(sensor_data)
        SAFE_THRESHOLD = 0.35
        if min_lidar_val < SAFE_THRESHOLD:
            reward -= (SAFE_THRESHOLD - min_lidar_val) * 2.0

        # 장애물 충돌
        for obs in self.obstacles:
            dist = np.linalg.norm(self.pos - obs)
            if dist < (self.AGENT_RADIUS + self.OBSTACLE_RADIUS):
                terminated = True
                reward -= 70.0
                return self._get_obs(sensor_data), reward, terminated, False, {}

        # 방향 보상 (Waypoint 추종)
        to_objective = self.active_waypoint - self.pos
        dist_objective = np.linalg.norm(to_objective)
        
        if speed < 1.0:
            reward -= 0.2
        
        if speed > 5.0:
            # 여기서는 '진짜 목표'가 아니라 '계산된 우회점'을 잘 따라가는지 평가합니다.
            cosine = np.dot(self.vel, to_objective) / (speed * dist_objective + 1e-8)
            reward += cosine * 0.15 
            
            # 안전하고 방향 맞으면 가속 보상
            if cosine > 0.8 and min_lidar_val > SAFE_THRESHOLD:
                reward += (speed / self.MAX_SPEED) * 0.1

            # 회피 모드일 때 추가점수
            if self.avoidance_timer > 0:
                reward += cosine * 0.5

        # 목표 달성
        dist_target = np.linalg.norm(self.target - self.pos)
        if dist_target < (self.AGENT_RADIUS + self.TARGET_RADIUS):
            reward += 30.0 
            self.score += 1
            self.target = self._spawn_entity(padding=50, check_overlap=True)
            self.active_waypoint = self.target # 목표 달성 시 웨이포인트 초기화
        
        action_diff = np.linalg.norm(np.array(action) - self.prev_action)
        reward -= action_diff * 0.1 
        self.prev_action = np.array(action)

        truncated = self.steps >= self.max_steps
        return self._get_obs(sensor_data), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
                self.clock = pygame.time.Clock()
                self.font = pygame.font.SysFont("Consolas", 18)
            
            self.screen.fill((20, 20, 20))
            
            # LIDAR 그리기
            if hasattr(self, 'current_rays_rendering'):
                for ray_dir, dist, reading in self.current_rays_rendering:
                    c_val = int(255 * reading)
                    c_val = max(0, min(255, c_val))
                    # 가까우면 빨강(위험), 멀면 초록
                    start = self.pos.astype(int)
                    end = (self.pos + ray_dir * dist).astype(int)
                    pygame.draw.line(self.screen, (255-c_val, c_val, 0), start, end, 1)

            # 장애물
            for obs in self.obstacles:
                pygame.draw.circle(self.screen, (100, 100, 100), obs.astype(int), self.OBSTACLE_RADIUS)
                
            # 목표물
            pygame.draw.circle(self.screen, (255, 50, 50), self.target.astype(int), self.TARGET_RADIUS)
            # 에이전트
            pygame.draw.circle(self.screen, (255, 255, 255), self.pos.astype(int), self.AGENT_RADIUS)
            
            # [시각화] 자동 회피 경로 (파란 선)
            if self.active_waypoint is not None:
                pygame.draw.circle(self.screen, (0, 100, 255), self.active_waypoint.astype(int), 5)
                pygame.draw.line(self.screen, (0, 100, 255), self.pos.astype(int), self.active_waypoint.astype(int), 2)

            # UI 텍스트
            speed = np.linalg.norm(self.vel)
            info_texts = [
                f"Score: {self.score}",
                f"Speed: {speed:.1f}",
                f"Mode: {'Avoiding' if self.avoidance_timer > 0 else 'Tracking'}"
            ]
            
            start_y = 30 
            for i, text in enumerate(info_texts):
                ts = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(ts, (10, start_y + i * 20))
            
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        if self.screen is not None: pygame.quit()


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_PATH = f"final/{current_time}_swingby"
    
    env = InertiaRacerEnv()
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    print(">>> Training (Hybrid: Auto-Waypoints + Enhanced LIDAR)...")
    
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        learning_rate=0.0003, 
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    )
    
    model.learn(total_timesteps=300000)
    model.save(MODEL_PATH)
    
    print(">>> Testing...")
    test_env = InertiaRacerEnv(render_mode="human")
    test_vec_env = DummyVecEnv([lambda: test_env])
    test_vec_env = VecFrameStack(test_vec_env, n_stack=4)
    
    model = PPO.load(MODEL_PATH)
    obs = test_vec_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = test_vec_env.step(action)
        test_env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: exit()