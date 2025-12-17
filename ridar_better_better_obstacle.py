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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(InertiaRacerEnv, self).__init__()
        
        # --- 설정 ---
        self.SCREEN_SIZE = 800
        self.AGENT_RADIUS = 10
        self.TARGET_RADIUS = 15
        self.OBSTACLE_RADIUS = 30 
        
        self.MAX_SPEED = 150.0
        self.ACCEL_POWER = 2.0  # [고정]
        self.FRICTION = 0.92    # [고정]
        
        self.NUM_OBSTACLES = 5 # 장애물 개수 적절히 조정
        
        # LIDAR
        self.RAY_NUM = 15 
        self.RAY_ANGLES = np.linspace(-90, 90, self.RAY_NUM) 
        self.BASE_RAY_LENGTH = 200.0

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        obs_size = 5 + self.RAY_NUM
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # 상태 변수
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
        self.avoidance_timer = 0
        self.last_blocking_obs = None
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
        
        current_ray_length = self.BASE_RAY_LENGTH + (speed * 6.0)

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

    def _get_blocking_obstacle(self):
        # 검사할 두 가지 경로 벡터
        check_vectors = []
        
        # 1. 목표물로 가는 벡터 (의도)
        to_target = self.target - self.pos
        dist_target = np.linalg.norm(to_target)
        if dist_target > 0:
            check_vectors.append((to_target / dist_target, dist_target))
            
        # 2. 현재 속도로 미끄러지는 벡터 (현실 - 관성 예측)
        speed = np.linalg.norm(self.vel)
        if speed > 10.0: # 속도가 어느 정도 있을 때만 예측
            # 1.5초 뒤의 위치까지 미리 검사 (속도가 빠를수록 멀리 봄)
            prediction_dist = speed * 1.5 
            check_vectors.append((self.vel / speed, prediction_dist))

        closest_obs = None
        min_dist_to_obs = float('inf')

        for obs in self.obstacles:
            is_blocking = False
            
            for direction, max_dist in check_vectors:
                to_obs = obs - self.pos
                
                # 벡터 내적 (Projection)으로 직선 거리 계산
                proj = np.dot(to_obs, direction)
                
                # 장애물이 내 뒤에 있거나(proj < 0), 검사 범위보다 멀면(proj > max_dist) 패스
                if proj <= 0 or proj >= max_dist:
                    continue
                
                # 직선 경로와 장애물 중심 사이의 수직 거리(Perpendicular Distance)
                closest_point_on_line = direction * proj
                perp_dist = np.linalg.norm(to_obs - closest_point_on_line)
                
                collision_threshold = self.OBSTACLE_RADIUS + self.AGENT_RADIUS + 15.0 
                
                if perp_dist < collision_threshold:
                    is_blocking = True
                    break # 하나의 경로라도 막히면 이 장애물은 위험함
            
            if is_blocking:
                dist_to_obs = np.linalg.norm(obs - self.pos)
                if dist_to_obs < min_dist_to_obs:
                    min_dist_to_obs = dist_to_obs
                    closest_obs = obs
                    
        return closest_obs

    def _get_detour_point(self, blocking_obs):
        # [수정 1] 역주행 방지 및 명확한 경로 선정
        to_obs = blocking_obs - self.pos
        
        # 1. 우회 반경 설정
        detour_offset = (self.OBSTACLE_RADIUS + self.AGENT_RADIUS) * 4.0
        
        # 장애물 기준 수직 벡터 생성
        perp_vec = np.array([-to_obs[1], to_obs[0]])
        perp_vec = (perp_vec / (np.linalg.norm(perp_vec) + 1e-6)) * detour_offset
        
        # 2. 전진 바이어스 (Forward Bias)
        to_real_target = self.target - self.pos
        target_dir = to_real_target / (np.linalg.norm(to_real_target) + 1e-6)
        
        # 목표물 방향으로 약간 당겨줌 (너무 많이 당기면 장애물과 겹치니 적당히 60)
        forward_bias = target_dir * 60.0 
        
        waypoint1 = blocking_obs + perp_vec + forward_bias
        waypoint2 = blocking_obs - perp_vec + forward_bias
        
        # 3. [핵심 수정] 기준 벡터를 무조건 '목표물 방향'으로 고정
        # 기존: self.vel (내가 미끄러지면 뒤쪽을 선택해버림 -> 역주행 원인)
        # 수정: self.target - self.pos (무조건 목표물과 가까운 쪽 선택)
        ref_vec = self.target - self.pos
        
        # 두 후보 지점 중, 목표물과 더 가까운(내적이 큰) 곳을 선택
        if np.dot(ref_vec, waypoint1 - self.pos) > np.dot(ref_vec, waypoint2 - self.pos):
            return waypoint1
        else:
            return waypoint2

    def _get_obs(self, sensor_data=None):
        if sensor_data is None: sensor_data = self._get_rays()
        
        # 현재 경로를 막고 있는 장애물 확인
        blocking_obs = self._get_blocking_obstacle()
        
        # [로직 1] 장애물 감지 시 회피 모드 진입
        if blocking_obs is not None:
            self.last_blocking_obs = blocking_obs
            self.avoidance_timer = 20 # 20프레임 동안 회피 유지
        
        # [로직 2] 조기 종료(Early Exit) - 목표물 지나침 방지
        # 타이머가 남아있더라도, 지금 당장 장애물이 없다면(blocking_obs is None)
        # 즉시 회피 모드를 풀고 목표물로 직행
        if blocking_obs is None:
             self.avoidance_timer = 0
             self.last_blocking_obs = None

        # [로직 3] 목표 지점(Waypoint) 결정
        if self.avoidance_timer > 0 and self.last_blocking_obs is not None:
            # 회피 중: 계산된 우회점을 바라봄
            self.active_waypoint = self._get_detour_point(self.last_blocking_obs)
            self.avoidance_timer -= 1
        else:
            # 평시: 진짜 목표물을 바라봄
            self.active_waypoint = self.target

        to_objective = self.active_waypoint - self.pos
        dist_objective = np.linalg.norm(to_objective)
        scale = self.SCREEN_SIZE
        
        obs = np.concatenate([
            self.vel / self.MAX_SPEED,
            to_objective / scale,
            [dist_objective / scale],
            sensor_data
        ])
        return obs.astype(np.float32)

    def step(self, action):
        self.steps += 1
        
        # 물리 적용 (가속 2.0, 마찰 0.92 고정)
        accel = np.array(action) * self.ACCEL_POWER
        self.last_accel = accel
        
        self.vel += accel
        speed = np.linalg.norm(self.vel)
        if speed > self.MAX_SPEED:
            self.vel = (self.vel / speed) * self.MAX_SPEED
        self.vel *= self.FRICTION
        self.pos += self.vel
        
        reward = -0.05 
        terminated = False
        
        # 벽 반발장
        WALL_MARGIN = 40.0 
        wall_penalty = 0.0
        
        if self.pos[0] < WALL_MARGIN:
            wall_penalty += (WALL_MARGIN - self.pos[0]) / WALL_MARGIN
        elif self.pos[0] > self.SCREEN_SIZE - WALL_MARGIN:
            wall_penalty += (self.pos[0] - (self.SCREEN_SIZE - WALL_MARGIN)) / WALL_MARGIN
            
        if self.pos[1] < WALL_MARGIN:
            wall_penalty += (WALL_MARGIN - self.pos[1]) / WALL_MARGIN
        elif self.pos[1] > self.SCREEN_SIZE - WALL_MARGIN:
            wall_penalty += (self.pos[1] - (self.SCREEN_SIZE - WALL_MARGIN)) / WALL_MARGIN
            
        if wall_penalty > 0:
            reward -= wall_penalty 
            
        if self.pos[0] < 0 or self.pos[0] > self.SCREEN_SIZE or self.pos[1] < 0 or self.pos[1] > self.SCREEN_SIZE:
            self.pos = np.clip(self.pos, 0, self.SCREEN_SIZE)
            self.vel *= 0.5
            reward -= 5.0

        # TTC
        sensor_data = self._get_rays()
        min_dist_normalized = np.min(sensor_data)
        
        speed_ratio = speed / self.MAX_SPEED
        urgency = (speed_ratio ** 2) / (min_dist_normalized + 0.05)
        
        TTC_THRESHOLD = 2.5
        if urgency > TTC_THRESHOLD:
            reward -= speed_ratio * 1.5 
            reward -= (urgency - TTC_THRESHOLD) * 0.5

        # 장애물 충돌
        for obs in self.obstacles:
            dist = np.linalg.norm(self.pos - obs)
            if dist < (self.AGENT_RADIUS + self.OBSTACLE_RADIUS):
                terminated = True
                reward -= 30.0
                return self._get_obs(sensor_data), reward, terminated, False, {}

        # 방향 보상
        to_objective = self.active_waypoint - self.pos
        dist_objective = np.linalg.norm(to_objective)
        
        if speed > 5.0:
            cosine = np.dot(self.vel, to_objective) / (speed * dist_objective + 1e-8)
            reward += cosine * 0.1 
            
            if cosine > 0.8 and urgency < TTC_THRESHOLD:
                reward += speed_ratio * 0.1

        # 목표 달성
        dist_target = np.linalg.norm(self.target - self.pos)
        if dist_target < (self.AGENT_RADIUS + self.TARGET_RADIUS):
            reward += 30.0 
            self.score += 1
            self.target = self._spawn_entity(padding=50, check_overlap=True)
        
        # 행동 일관성
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
            
            # 벽 가이드
            pygame.draw.rect(self.screen, (50, 0, 0), (0, 0, 60, self.SCREEN_SIZE))
            pygame.draw.rect(self.screen, (50, 0, 0), (self.SCREEN_SIZE-60, 0, 60, self.SCREEN_SIZE))
            pygame.draw.rect(self.screen, (50, 0, 0), (0, 0, self.SCREEN_SIZE, 60))
            pygame.draw.rect(self.screen, (50, 0, 0), (0, self.SCREEN_SIZE-60, self.SCREEN_SIZE, 60))
            
            if hasattr(self, 'current_rays_rendering'):
                for ray_dir, dist, reading in self.current_rays_rendering:
                    c_val = int(255 * reading)
                    c_val = max(0, min(255, c_val))
                    start = self.pos.astype(int)
                    end = (self.pos + ray_dir * dist).astype(int)
                    pygame.draw.line(self.screen, (255-c_val, c_val, 0), start, end, 1)

            for obs in self.obstacles:
                pygame.draw.circle(self.screen, (100, 100, 100), obs.astype(int), self.OBSTACLE_RADIUS)
                
            pygame.draw.circle(self.screen, (255, 50, 50), self.target.astype(int), self.TARGET_RADIUS)
            pygame.draw.circle(self.screen, (255, 255, 255), self.pos.astype(int), self.AGENT_RADIUS)
            
            if self.active_waypoint is not None and not np.array_equal(self.active_waypoint, self.target):
                 # 가상 목표 파란색
                 pygame.draw.circle(self.screen, (0, 100, 255), self.active_waypoint.astype(int), 5)
                 pygame.draw.line(self.screen, (0, 100, 255), self.pos.astype(int), self.active_waypoint.astype(int), 1)

            speed = np.linalg.norm(self.vel)
            info_texts = [
                f"Score: {self.score}",
                f"Speed: {speed:.1f}",
                f"Accel: ({self.last_accel[0]:.1f}, {self.last_accel[1]:.1f})"
            ]
            
            start_y = 50 
            for i, text in enumerate(info_texts):
                ts = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(ts, (10, start_y + i * 20))
            
            pygame.display.flip()
            self.clock.tick(20)

    def close(self):
        if self.screen is not None: pygame.quit()


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%D%H%M%S")
    MODEL_PATH = f"exp/smart_path_last_hope_{current_time}"
    
    env = InertiaRacerEnv()
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    print(">>> Training (Fixing Orbiting & Wall Hugging)...")
    
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
    
    model.learn(total_timesteps=500000)
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