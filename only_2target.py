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
        
        self.MAX_SPEED = 189.0
        self.ACCEL_POWER = 2.0
        self.FRICTION = 0.92
        
        self.NUM_OBSTACLES = 5
        
        # LIDAR (32개로 유지하여 센서 정보는 충분히 줌)
        self.RAY_NUM = 32
        self.RAY_ANGLES = np.linspace(-135, 135, self.RAY_NUM) 
        self.BASE_RAY_LENGTH = 200.0

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # [Observation Space] - Momentum 모델 핵심
        # 스윙바이 로직은 없지만, Next Target 정보는 봅니다.
        # [내속도(2), 현재목표(2), 현재거리(1), 다음목표(2), 다음거리(1), LIDAR(32)] = 총 40개
        obs_size = 8 + self.RAY_NUM
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.prev_action = np.array([0.0, 0.0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([self.SCREEN_SIZE/2, self.SCREEN_SIZE/2], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        
        self.obstacles = [] 
        # 타겟 2개 생성 (Next Target 활용)
        self.target = self._spawn_entity(padding=50)
        self.next_target = self._spawn_entity(padding=50, check_overlap=True, extra_point=self.target)
        
        for _ in range(self.NUM_OBSTACLES):
            self.obstacles.append(self._spawn_entity(padding=50, check_overlap=True))
        
        self.score = 0
        self.steps = 0
        self.max_steps = 1000
        self.prev_action = np.array([0.0, 0.0])
        
        return self._get_obs(), {}

    def _spawn_entity(self, padding=50, check_overlap=False, extra_point=None):
        while True:
            pos = np.random.uniform(padding, self.SCREEN_SIZE - padding, size=2).astype(np.float32)
            if not check_overlap: return pos
            
            safe = True
            if np.linalg.norm(pos - self.pos) < 150: safe = False
            if hasattr(self, 'target') and np.linalg.norm(pos - self.target) < 100: safe = False
            if extra_point is not None and np.linalg.norm(pos - extra_point) < 100: safe = False
            
            for obs in self.obstacles:
                if np.linalg.norm(pos - obs) < 80: safe = False
            if safe: return pos

    def _get_rays(self):
        speed = np.linalg.norm(self.vel)
        heading = math.atan2(self.vel[1], self.vel[0]) if speed > 1.0 else 0.0
        current_ray_length = self.BASE_RAY_LENGTH + (speed * 5.0)

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

    def _get_obs(self, sensor_data=None):
        if sensor_data is None: sensor_data = self._get_rays()
        
        # [우회 로직 없음] 
        # 에이전트는 우회점(Waypoint)이 아닌 실제 타겟(Target)과 다음 타겟(Next Target) 정보를 받음
        to_target = self.target - self.pos
        dist_target = np.linalg.norm(to_target)
        
        to_next = self.next_target - self.pos
        dist_next = np.linalg.norm(to_next)
        
        scale = self.SCREEN_SIZE
        
        obs = np.concatenate([
            self.vel / self.MAX_SPEED,
            to_target / scale,
            [dist_target / scale],
            to_next / scale,      # 핵심: 다음 타겟 벡터
            [dist_next / scale],  # 핵심: 다음 타겟 거리
            sensor_data
        ])
        return obs.astype(np.float32)

    def step(self, action):
        self.steps += 1
        accel = np.array(action) * self.ACCEL_POWER
        self.vel += accel
        speed = np.linalg.norm(self.vel)
        if speed > self.MAX_SPEED:
            self.vel = (self.vel / speed) * self.MAX_SPEED
        self.vel *= self.FRICTION
        self.pos += self.vel
        
        reward = -0.05
        terminated = False
        
        if self.pos[0] < 0 or self.pos[0] > self.SCREEN_SIZE or self.pos[1] < 0 or self.pos[1] > self.SCREEN_SIZE:
            self.pos = np.clip(self.pos, 0, self.SCREEN_SIZE)
            self.vel *= 0.5
            reward -= 5.0

        sensor_data = self._get_rays()
        
        for obs in self.obstacles:
            dist = np.linalg.norm(self.pos - obs)
            if dist < (self.AGENT_RADIUS + self.OBSTACLE_RADIUS):
                terminated = True
                reward -= 50.0
                return self._get_obs(sensor_data), reward, terminated, False, {}

        to_target = self.target - self.pos
        dist_target = np.linalg.norm(to_target)
        
        # 방향 보상 (현재 타겟 향해서)
        if speed > 1.0:
            cosine = np.dot(self.vel, to_target) / (speed * dist_target + 1e-8)
            reward += cosine * 0.1
        
        # [목표 달성 로직]
        if dist_target < (self.AGENT_RADIUS + self.TARGET_RADIUS):
            reward += 30.0 
            self.score += 1
            
            # [관성 보너스 - Momentum Reward]
            # 우회 로직은 없지만, 보상은 "다음 타겟"을 고려해서 줌 -> 강화학습이 스스로 배우도록 유도
            to_next = self.next_target - self.pos
            dist_next = np.linalg.norm(to_next)
            
            if speed > 5.0 and dist_next > 0:
                vel_dir = self.vel / speed
                next_dir = to_next / dist_next
                alignment = np.dot(vel_dir, next_dir)
                
                # 다음 타겟 방향으로 미리 몸을 틀어놨다면 큰 점수
                if alignment > 0:
                    reward += alignment * (speed / self.MAX_SPEED) * 20.0
            
            # 타겟 갱신
            self.target = self.next_target
            self.next_target = self._spawn_entity(padding=50, check_overlap=True, extra_point=self.target)
        
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
            
            for ray_dir, dist, reading in self.current_rays_rendering:
                c_val = int(255 * reading)
                c_val = max(0, min(255, c_val))
                start = self.pos.astype(int)
                end = (self.pos + ray_dir * dist).astype(int)
                pygame.draw.line(self.screen, (255-c_val, c_val, 0), start, end, 1)

            for obs in self.obstacles:
                pygame.draw.circle(self.screen, (100, 100, 100), obs.astype(int), self.OBSTACLE_RADIUS)
            
            # [시각화 차이] Next Target(Ghost) 표시 (Momentum 모델임을 알 수 있음)
            pygame.draw.circle(self.screen, (0, 0, 150), self.next_target.astype(int), self.TARGET_RADIUS, 2)
            pygame.draw.line(self.screen, (100, 100, 100), self.target.astype(int), self.next_target.astype(int), 1)

            pygame.draw.circle(self.screen, (255, 50, 50), self.target.astype(int), self.TARGET_RADIUS)
            pygame.draw.circle(self.screen, (255, 255, 255), self.pos.astype(int), self.AGENT_RADIUS)
            
            speed = np.linalg.norm(self.vel)
            info_texts = [f"Score: {self.score}", f"Speed: {speed:.1f}", "Mode: MOMENTUM (Next-Target Only)"]
            for i, text in enumerate(info_texts):
                ts = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(ts, (10, 30 + i * 20))
            
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        if self.screen is not None: pygame.quit()

if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_PATH = f"final/{current_time}_2target"
    
    env = InertiaRacerEnv()
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    print(">>> Training Momentum Only (No Swing-by, Yes Next-Target)...")
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, ent_coef=0.01)
    model.learn(total_timesteps=300000)
    model.save(MODEL_PATH)
    
    print(">>> Testing Momentum...")
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