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
        self.OBSTACLE_RADIUS = 30 # 장애물 크기 인식 범위를 좀 더 넉넉하게
        
        self.MAX_SPEED = 150.0
        self.ACCEL_POWER = 2.0 # 회피 기동을 위해 가속력 상향
        self.FRICTION = 0.90   # 마찰력을 높여서(0.92->0.90) 제동을 쉽게 만듦
        
        self.NUM_OBSTACLES = 3
        
        # LIDAR 설정
        self.RAY_NUM = 15
        self.RAY_ANGLES = np.linspace(-90, 90, self.RAY_NUM) 
        self.BASE_RAY_LENGTH = 250.0

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 관측 공간
        obs_size = 5 + self.RAY_NUM
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

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
        return self._get_obs(), {}

    def _spawn_entity(self, padding=50, check_overlap=False):
        while True:
            pos = np.random.uniform(padding, self.SCREEN_SIZE - padding, size=2).astype(np.float32)
            if not check_overlap: return pos
            
            safe = True
            # 생성 조건 강화
            if np.linalg.norm(pos - self.pos) < 150: safe = False
            if np.linalg.norm(pos - self.target) < 100: safe = False
            for obs in self.obstacles:
                if np.linalg.norm(pos - obs) < 80: safe = False
            if safe: return pos

    def _get_rays(self):
        speed = np.linalg.norm(self.vel)
        heading = math.atan2(self.vel[1], self.vel[0]) if speed > 1.0 else 0.0
        current_ray_length = self.BASE_RAY_LENGTH + (speed * 1.0) # 속도 비례 길이

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

    def _get_obs(self):
        sensor_data = self._get_rays()
        to_target = self.target - self.pos
        dist_target = np.linalg.norm(to_target)
        scale = self.SCREEN_SIZE
        
        obs = np.concatenate([
            self.vel / self.MAX_SPEED,
            to_target / scale,
            [dist_target / scale],
            sensor_data
        ])
        return obs.astype(np.float32)

    # [핵심 추가] 목표물까지 가는 길이 막혔는지 확인하는 함수
    def _check_path_blocked(self):
        to_target = self.target - self.pos
        dist_target = np.linalg.norm(to_target)
        if dist_target == 0: return False
        
        dir_target = to_target / dist_target
        
        # 목표물까지 선을 그었을 때 장애물과 겹치는지 확인 (Raycast)
        for obs in self.obstacles:
            to_obs = obs - self.pos
            proj = np.dot(to_obs, dir_target)
            
            # 장애물이 목표물보다 앞에 있고, 투영 거리가 가까우면
            if 0 < proj < dist_target:
                perp_dist = np.linalg.norm(to_obs - (dir_target * proj))
                # 장애물 반지름 + 여유공간(20) 안쪽이면 막힌 것으로 간주
                if perp_dist < (self.OBSTACLE_RADIUS + self.AGENT_RADIUS + 10):
                    return True
        return False

    def step(self, action):
        self.steps += 1
        
        # 1. 물리 적용
        accel = np.array(action) * self.ACCEL_POWER
        self.vel += accel
        speed = np.linalg.norm(self.vel)
        if speed > self.MAX_SPEED:
            self.vel = (self.vel / speed) * self.MAX_SPEED
        self.vel *= self.FRICTION
        self.pos += self.vel
        
        reward = -0.01 # 시간 페널티
        terminated = False
        
        # 2. 벽 충돌 처리
        if self.pos[0] < 0 or self.pos[0] > self.SCREEN_SIZE or self.pos[1] < 0 or self.pos[1] > self.SCREEN_SIZE:
            self.pos = np.clip(self.pos, 0, self.SCREEN_SIZE)
            self.vel *= 0.5
            reward -= 5.0 # 벽도 강하게 처벌

        # 3. 장애물 처리
        min_obs_dist = float('inf')
        for obs in self.obstacles:
            dist = np.linalg.norm(self.pos - obs)
            min_obs_dist = min(min_obs_dist, dist)
            
            if dist < (self.AGENT_RADIUS + self.OBSTACLE_RADIUS):
                terminated = True
                reward -= 100.0 # 충돌 시 강력한 페널티
                return self._get_obs(), reward, terminated, False, {}
            
            # 근접 공포 (Repulsion)
            if dist < 120.0:
                reward -= (120.0 - dist) / 120.0 * 0.5

        # 4. [핵심 로직] 경로 상태에 따른 보상 스위칭
        path_blocked = self._check_path_blocked()
        
        to_target = self.target - self.pos
        dist_target = np.linalg.norm(to_target)
        
        if speed > 10.0:
            # 내 진행 방향이 타겟과 일치하는가?
            cosine = np.dot(self.vel, to_target) / (speed * dist_target + 1e-8)
            
            if not path_blocked:
                # [상황 A] 길이 뚫려있음 -> 기존처럼 목표물 방향 보상 지급
                if cosine > 0:
                    reward += cosine * 1.0 
                    reward += (1.0 - dist_target/self.SCREEN_SIZE) * 0.2
            else:
                # [상황 B] 길이 막힘 -> 목표물 쳐다보면 오히려 감점일 수 있음!
                # 이때는 "속도 유지"와 "탐색"에 점수를 줌
                # 목표물을 향해 정면으로 가려고 하면(cosine 높음) 감점 (빙빙 돌기 방지)
                if cosine > 0.8: 
                    reward -= 0.5 
                else:
                    # 장애물을 피해서 움직이고 있으면 보너스
                    reward += 0.2
        else:
            # 멈춰있으면 큰 감점 (빙빙 돌다가 멈추는 것 방지)
            reward -= 0.2

        # 5. 목표물 획득
        if dist_target < (self.AGENT_RADIUS + self.TARGET_RADIUS):
            reward += 50.0
            self.score += 1
            self.target = self._spawn_entity(padding=50, check_overlap=True)

        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
                self.clock = pygame.time.Clock()
            
            self.screen.fill((20, 20, 20))
            
            # 경로 막힘 여부 시각화 (디버깅용)
            blocked = self._check_path_blocked()
            line_col = (255, 0, 0) if blocked else (0, 255, 0)
            pygame.draw.line(self.screen, line_col, self.pos.astype(int), self.target.astype(int), 1)

            # 라이다 그리기
            if hasattr(self, 'current_rays_rendering'):
                for ray_dir, dist, reading in self.current_rays_rendering:
                    color = (int(255 * (1-reading)), int(255 * reading), 0)
                    start = self.pos
                    end = self.pos + ray_dir * dist
                    pygame.draw.line(self.screen, color, start.astype(int), end.astype(int), 1)

            # 장애물
            for obs in self.obstacles:
                pygame.draw.circle(self.screen, (100, 100, 100), obs.astype(int), self.OBSTACLE_RADIUS)
                
            pygame.draw.circle(self.screen, (255, 50, 50), self.target.astype(int), self.TARGET_RADIUS)
            pygame.draw.circle(self.screen, (255, 255, 255), self.pos.astype(int), self.AGENT_RADIUS)
            
            # 점수 표시
            font = pygame.font.SysFont("Arial", 20)
            msg = f"Path Blocked: {blocked}"
            self.screen.blit(font.render(msg, True, line_col), (10, 10))
            
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        if self.screen is not None: pygame.quit()

if __name__ == "__main__":
    # 학습 및 실행 코드는 기존과 동일하게 VecFrameStack 사용 권장
    current_time = datetime.now().strftime("%H%M%S")
    MODEL_PATH = f"exp/smart_path_{current_time}"
    
    env = InertiaRacerEnv()
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=4) # 프레임 스태킹 유지
    
    print(">>> Training (Path Logic Added)...")
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, ent_coef=0.01)
    model.learn(total_timesteps=400000)
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