import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import math
from collections import deque

# =============================================================================
# [환경 클래스 정의]
# =============================================================================
class InertiaRacerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(InertiaRacerEnv, self).__init__()
        
        # --- 설정 ---
        self.SCREEN_SIZE = 800
        self.AGENT_RADIUS = 10
        self.TARGET_RADIUS = 15
        self.OBSTACLE_RADIUS = 30 
        self.MAX_SPEED = 20.0
        self.ACCEL_POWER = 2.0
        self.FRICTION = 0.92
        self.NUM_OBSTACLES = 5
        self.RAY_NUM = 32 
        self.RAY_ANGLES = np.linspace(-135, 135, self.RAY_NUM) 
        self.BASE_RAY_LENGTH = 200.0

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.level = 4 

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_size = 5 + self.RAY_NUM
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # 상태 변수 초기화
        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.active_waypoint = None
        self.last_accel = np.array([0.0, 0.0])
        self.prev_action = np.array([0.0, 0.0])
        
        # 회피 관련 변수
        self.avoidance_timer = 0 
        self.last_blocking_obs = None
        
        self.score = 0.0
        
        # 그래프 데이터 저장소 (최근 300프레임)
        self.reward_history = deque(maxlen=300)

    def set_level(self, level):
        self.level = level

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([self.SCREEN_SIZE/2, self.SCREEN_SIZE/2], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.obstacles = []
        
        # 레벨별 장애물 배치
        if self.level == 1:
            for _ in range(2): self.obstacles.append(self._spawn_entity(padding=50, check_overlap=True))
        elif self.level == 2:
            wall_x = self.SCREEN_SIZE / 2 + 100
            for i in range(5):
                y_pos = 100 + i * 140
                self.obstacles.append(np.array([wall_x + np.random.uniform(-20, 20), y_pos], dtype=np.float32))
        elif self.level == 3:
            for _ in range(4):
                base = self._spawn_entity(padding=80, check_overlap=True)
                self.obstacles.append(base)
                angle = np.random.uniform(0, 2*np.pi)
                offset = np.array([math.cos(angle), math.sin(angle)]) * (self.OBSTACLE_RADIUS*2 + 10)
                self.obstacles.append(np.clip(base + offset, 50, self.SCREEN_SIZE-50))
        elif self.level >= 4:
            for _ in range(self.NUM_OBSTACLES): self.obstacles.append(self._spawn_entity(padding=50, check_overlap=True))

        self.target = self._spawn_entity(padding=50, check_overlap=True)
        self.active_waypoint = self.target 
        
        self.score = 0.0
        self.steps = 0
        self.max_steps = 1000
        self.prev_action = np.array([0.0, 0.0])
        self.last_accel = np.array([0.0, 0.0])
        self.avoidance_timer = 0
        self.last_blocking_obs = None
        
        # 그래프 초기화
        self.reward_history.clear()
        
        return self._get_obs(), {}

    def _spawn_entity(self, padding=50, check_overlap=False):
        while True:
            pos = np.random.uniform(padding, self.SCREEN_SIZE - padding, size=2).astype(np.float32)
            if not check_overlap: return pos
            safe = True
            if np.linalg.norm(pos - self.pos) < 150: safe = False
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
                        if 0 < hit_dist < min_dist: min_dist = hit_dist
            ray_readings.append(min_dist / current_ray_length)
            self.current_rays_rendering.append((ray_dir, min_dist, min_dist/current_ray_length))
        return np.array(ray_readings, dtype=np.float32)

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

    def _get_detour_point(self, blocking_obs):
        to_obs = blocking_obs - self.pos
        detour_offset = (self.OBSTACLE_RADIUS + self.AGENT_RADIUS) * 4.5 
        
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

        to_objective = self.active_waypoint - self.pos 
        
        obs = np.concatenate([
            self.vel / self.MAX_SPEED,
            to_objective / self.SCREEN_SIZE,
            [np.linalg.norm(to_objective) / self.SCREEN_SIZE],
            sensor_data 
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

        sensor_data = self._get_rays()
        
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
                
                # 충돌 시 그래프 저장 (클리핑)
                graph_val = max(reward, -5.0)
                self.reward_history.append(graph_val)
                self.score += reward
                return self._get_obs(sensor_data), reward, terminated, False, {}

        # 안전장치
        if self.active_waypoint is None: self.active_waypoint = self.target

        to_objective = self.active_waypoint - self.pos
        dist_objective = np.linalg.norm(to_objective)
        
        if speed < 1.0:
            reward -= 0.5
        
        if speed > 5.0:
            cosine = np.dot(self.vel, to_objective) / (speed * dist_objective + 1e-8)
            reward += cosine * 0.15 
            
            if cosine > 0.8 and min_lidar_val > SAFE_THRESHOLD:
                reward += (speed / self.MAX_SPEED) * 0.1

            if self.avoidance_timer > 0:
                reward += cosine * 0.7
        
        # 그래프용 보상 값 저장 (목표 달성 보너스 추가 전)
        reward_for_graph = reward

        dist_target = np.linalg.norm(self.target - self.pos)
        if dist_target < (self.AGENT_RADIUS + self.TARGET_RADIUS):
            reward += 30.0 
            self.score += 1 # 점수는 +1
            self.target = self._spawn_entity(padding=50, check_overlap=True)
            self.active_waypoint = self.target
            self.avoidance_timer = 0
            self.last_blocking_obs = None 
            # *그래프에는 30점을 더하지 않음*
        
        action_diff = np.linalg.norm(np.array(action) - self.prev_action)
        reward -= action_diff * 0.1 
        self.prev_action = np.array(action)

        truncated = self.steps >= self.max_steps
        
        # 그래프 데이터 저장
        self.reward_history.append(max(reward_for_graph, -5.0))
        
        return self._get_obs(sensor_data), reward, terminated, truncated, {}

    # -------------------------------------------------------------------------
    # [수정됨] 그래프 그리기 함수 (TypeError 수정)
    # -------------------------------------------------------------------------
    def _draw_reward_graph(self, screen):
        if len(self.reward_history) < 2: return

        # 그래프 박스 설정 (우측 하단)
        box_w, box_h = 250, 100
        box_x = self.SCREEN_SIZE - box_w - 10
        box_y = self.SCREEN_SIZE - box_h - 10
        
        # 반투명 배경
        s = pygame.Surface((box_w, box_h))
        s.set_alpha(150)
        s.fill((0, 0, 0))
        screen.blit(s, (box_x, box_y))
        
        # 테두리
        pygame.draw.rect(screen, (255, 255, 255), (box_x, box_y, box_w, box_h), 1)
        
        # 중심선 (0.0)
        mid_y = box_y + box_h / 2
        pygame.draw.line(screen, (100, 100, 100), (box_x, mid_y), (box_x + box_w, mid_y), 1)
        
        # 데이터 그리기
        scale_y = 40.0 
        points = []
        for i, val in enumerate(self.reward_history):
            # [중요] numpy float 타입을 python float으로 명시적 형변환
            safe_val = float(val) if not np.isnan(val) else 0.0
            
            px = box_x + (i / len(self.reward_history)) * box_w
            
            # 값 클램핑
            clamped_val = max(-1.2, min(1.2, safe_val))
            py = mid_y - (clamped_val * scale_y)
            
            # [중요] 좌표도 float()으로 감싸서 순수 숫자로 만듦
            points.append((float(px), float(py)))
            
        if len(points) > 1:
            pygame.draw.lines(screen, (0, 255, 0), False, points, 2)
            
        # 라벨
        label = self.font.render("Reward Flow (No Target Bonus)", True, (200, 200, 200))
        screen.blit(label, (box_x + 5, box_y + 5))

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Consolas", 18)
        
        self.screen.fill((20, 20, 20))
        
        # # LIDAR
        # if hasattr(self, 'current_rays_rendering'):
        #     for ray_dir, dist, reading in self.current_rays_rendering:
        #         c_val = int(255 * reading)
        #         c_val = max(0, min(255, c_val))
        #         start = self.pos.astype(int)
        #         end = (self.pos + ray_dir * dist).astype(int)
        #         pygame.draw.line(self.screen, (255-c_val, c_val, 0), start, end, 1)

        # 장애물
        for obs in self.obstacles:
            pygame.draw.circle(self.screen, (100, 100, 100), obs.astype(int), self.OBSTACLE_RADIUS)
        
        # 목표물
        pygame.draw.circle(self.screen, (255, 50, 50), self.target.astype(int), self.TARGET_RADIUS)
        
        # 회피 경로(WayPoint) 시각화
        if self.active_waypoint is not None:
             pygame.draw.circle(self.screen, (0, 100, 255), self.active_waypoint.astype(int), 6)
             pygame.draw.line(self.screen, (0, 100, 255), self.pos.astype(int), self.active_waypoint.astype(int), 2)
        
        # 에이전트
        pygame.draw.circle(self.screen, (255, 255, 255), self.pos.astype(int), self.AGENT_RADIUS)
        
        # UI 텍스트
        speed = np.linalg.norm(self.vel)
        info_texts = [
            f"Score: {self.score}", 
            f"Speed: {speed:.1f}", 
            f"Mode: {'Avoiding' if self.avoidance_timer > 0 else 'Tracking'}",
            f"Level: {self.level}"
        ]
        
        start_y = 30 
        for i, text in enumerate(info_texts):
            ts = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(ts, (10, start_y + i * 20))
        
        # 그래프 그리기 호출
        self._draw_reward_graph(self.screen)
        
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None: pygame.quit()

# =============================================================================
# [실행부]
# =============================================================================
if __name__ == "__main__":
    
    MODEL_PATH = r"20251218_163230_curriculum_150k.zip"  
    
    if not os.path.exists(MODEL_PATH) and not os.path.exists(MODEL_PATH + ".zip"):
        print(f"\n[Error] '{MODEL_PATH}' 파일을 찾을 수 없습니다.")
        print("현재 폴더 파일 목록:", os.listdir("."))
        exit()

    print(f">>> Loading Model: {MODEL_PATH}")

    env = InertiaRacerEnv(render_mode="human")
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=4) 

    model = PPO.load(MODEL_PATH)

    while True:
        for level in range(2,5):
            print(f">>> Playing Level {level}")
            env.set_level(level)
            obs = vec_env.reset()
            
            for _ in range(1200):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                env.render()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        exit()
            
            print(f"Level {level} Complete. Next level in 2 seconds...")
            pygame.time.wait(2000)
