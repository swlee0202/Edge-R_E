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
        self.ACCEL_POWER = 2.0  
        self.FRICTION = 0.92    
        
        self.NUM_OBSTACLES = 5
        
        # LIDAR
        self.RAY_NUM = 15 
        self.RAY_ANGLES = np.linspace(-90, 90, self.RAY_NUM) 
        self.BASE_RAY_LENGTH = 200.0

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # --- Observation Space 변경 ---
        # [내속도(2), 현재목표벡터(2), 현재목표거리(1), 다음목표벡터(2), 다음목표거리(1), 라이다(15)]
        # 총 23개 입력
        obs_size = 8 + self.RAY_NUM
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # 상태 변수
        self.active_waypoint = None # 현재 쫓아가야 할 좌표 (Current Goal)
        self.future_waypoint = None # 그 다음에 쫓아가야 할 좌표 (Next Goal)
        
        self.last_accel = np.array([0.0, 0.0])
        self.prev_action = np.array([0.0, 0.0])
        
        # 스마트 우회 관련
        self.avoidance_timer = 0 
        self.last_blocking_obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([self.SCREEN_SIZE/2, self.SCREEN_SIZE/2], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        
        # [수정됨] 에러 원인: 장애물 리스트가 정의되기 전에 _spawn_entity가 호출되어 에러 발생
        # 해결: 빈 리스트를 먼저 생성해 둡니다.
        self.obstacles = [] 
        
        # 목표물 2개 생성 (이제 self.obstacles가 빈 리스트로 존재하므로 에러 안 남)
        self.target = self._spawn_entity(padding=50)
        self.next_target = self._spawn_entity(padding=50, check_overlap=True, extra_point=self.target)
        
        # 장애물 실제 생성 및 추가
        for _ in range(self.NUM_OBSTACLES):
            self.obstacles.append(self._spawn_entity(padding=50, check_overlap=True))
        
        self.score = 0
        self.steps = 0
        self.max_steps = 1500 
        self.prev_action = np.array([0.0, 0.0])
        self.avoidance_timer = 0
        self.last_blocking_obs = None
        
        return self._get_obs(), {}

    def _spawn_entity(self, padding=50, check_overlap=False, extra_point=None):
        while True:
            pos = np.random.uniform(padding, self.SCREEN_SIZE - padding, size=2).astype(np.float32)
            if not check_overlap: return pos
            
            safe = True
            # 플레이어 주변 방지
            if np.linalg.norm(pos - self.pos) < 150: safe = False
            # 현재 타겟 주변 방지 (target이 있다면)
            if hasattr(self, 'target') and np.linalg.norm(pos - self.target) < 100: safe = False
            # 추가 포인트(next_target 등) 주변 방지
            if extra_point is not None and np.linalg.norm(pos - extra_point) < 100: safe = False
            
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
        # 현재 타겟을 향한 경로상의 장애물 확인
        to_target = self.target - self.pos
        dist_target = np.linalg.norm(to_target)
        if dist_target == 0: return None
        
        dir_target = to_target / dist_target
        closest_blocking_obs = None
        min_dist_to_obs = float('inf')

        for obs in self.obstacles:
            to_obs = obs - self.pos
            proj = np.dot(to_obs, dir_target)
            
            if 0 < proj < dist_target:
                perp_dist = np.linalg.norm(to_obs - (dir_target * proj))
                # 여유 공간 
                safety_margin = self.OBSTACLE_RADIUS + self.AGENT_RADIUS + 40.0
                
                if perp_dist < safety_margin:
                    dist_to_obs = np.linalg.norm(to_obs)
                    if dist_to_obs < min_dist_to_obs:
                        min_dist_to_obs = dist_to_obs
                        closest_blocking_obs = obs
        return closest_blocking_obs

    def _get_detour_point(self, blocking_obs):
        to_obs = blocking_obs - self.pos
        detour_offset = (self.OBSTACLE_RADIUS + self.AGENT_RADIUS) * 3.0
        
        perp_vec = np.array([-to_obs[1], to_obs[0]])
        perp_vec = (perp_vec / (np.linalg.norm(perp_vec) + 1e-6)) * detour_offset
        
        # 목표물 방향으로 살짝 당겨줌 (전진성 확보)
        to_real_target = self.target - self.pos
        target_dir = to_real_target / (np.linalg.norm(to_real_target) + 1e-6)
        forward_bias = target_dir * 60.0
        
        waypoint1 = blocking_obs + perp_vec + forward_bias
        waypoint2 = blocking_obs - perp_vec + forward_bias
        
        # 목표물과 더 가까운 쪽 선택
        if np.dot(to_real_target, waypoint1 - self.pos) > np.dot(to_real_target, waypoint2 - self.pos):
            return waypoint1
        else:
            return waypoint2

    def _get_obs(self, sensor_data=None):
        if sensor_data is None: sensor_data = self._get_rays()
        
        # 1. 장애물 감지 및 우회 로직
        blocking_obs = self._get_blocking_obstacle()
        
        if blocking_obs is not None:
            self.last_blocking_obs = blocking_obs
            self.avoidance_timer = 20
        
        if blocking_obs is None:
             self.avoidance_timer = 0
             self.last_blocking_obs = None

        # --- [핵심 로직] 목표 설정 ---
        # "스마트 우회 경로가 있는 경우: Current=Detour, Next=Target"
        # "스마트 우회 경로가 없는 경우: Current=Target, Next=NextTarget"
        
        if self.avoidance_timer > 0 and self.last_blocking_obs is not None:
            # [우회 모드]
            detour_point = self._get_detour_point(self.last_blocking_obs)
            self.active_waypoint = detour_point # 지금 목표: 우회점
            self.future_waypoint = self.target  # 다음 목표: 원래 타겟
        else:
            # [일반 모드]
            self.active_waypoint = self.target      # 지금 목표: 타겟
            self.future_waypoint = self.next_target # 다음 목표: 다음 타겟

        # 관측 벡터 생성
        scale = self.SCREEN_SIZE
        
        # Current Goal 정보
        to_current = self.active_waypoint - self.pos
        dist_current = np.linalg.norm(to_current)
        
        # Next Goal 정보
        to_next = self.future_waypoint - self.pos
        dist_next = np.linalg.norm(to_next)

        obs = np.concatenate([
            self.vel / self.MAX_SPEED,     # 내 속도
            to_current / scale,            # 현재 목표 벡터
            [dist_current / scale],        # 현재 목표 거리
            to_next / scale,               # 다음 목표 벡터 (미리 보라고 줌)
            [dist_next / scale],           # 다음 목표 거리
            sensor_data                    # LIDAR
        ])
        return obs.astype(np.float32)

    def step(self, action):
        self.steps += 1
        
        # 물리 적용
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
        
        # 벽 처리
        WALL_MARGIN = 40.0 
        wall_penalty = 0.0
        if self.pos[0] < WALL_MARGIN: wall_penalty += (WALL_MARGIN - self.pos[0]) / WALL_MARGIN
        elif self.pos[0] > self.SCREEN_SIZE - WALL_MARGIN: wall_penalty += (self.pos[0] - (self.SCREEN_SIZE - WALL_MARGIN)) / WALL_MARGIN
        if self.pos[1] < WALL_MARGIN: wall_penalty += (WALL_MARGIN - self.pos[1]) / WALL_MARGIN
        elif self.pos[1] > self.SCREEN_SIZE - WALL_MARGIN: wall_penalty += (self.pos[1] - (self.SCREEN_SIZE - WALL_MARGIN)) / WALL_MARGIN
            
        if wall_penalty > 0: reward -= wall_penalty 
            
        if self.pos[0] < 0 or self.pos[0] > self.SCREEN_SIZE or self.pos[1] < 0 or self.pos[1] > self.SCREEN_SIZE:
            self.pos = np.clip(self.pos, 0, self.SCREEN_SIZE)
            self.vel *= 0.1
            reward -= 5.0

        # 센서 및 TTC(충돌 시간) 계산
        sensor_data = self._get_rays()
        min_dist_normalized = np.min(sensor_data)
        
        speed_ratio = speed / self.MAX_SPEED
        urgency = (speed_ratio ** 2) / (min_dist_normalized + 0.05)
        
        TTC_THRESHOLD = 2.5
        if urgency > TTC_THRESHOLD:
            reward -= speed_ratio * 1.5 
            reward -= (urgency - TTC_THRESHOLD) * 0.5

        # 장애물 충돌 확인
        for obs in self.obstacles:
            dist = np.linalg.norm(self.pos - obs)
            if dist < (self.AGENT_RADIUS + self.OBSTACLE_RADIUS):
                terminated = True
                reward -= 30.0
                return self._get_obs(sensor_data), reward, terminated, False, {}

        # --- 방향 보상 (Shaping) ---
        # active_waypoint(현재 목표)를 향해 잘 가고 있는지
        to_current = self.active_waypoint - self.pos
        dist_current = np.linalg.norm(to_current)
        
        if speed < 1.0:
            reward -= 0.1
        
        if speed > 5.0:
            # 코사인 유사도
            cosine = np.dot(self.vel, to_current) / (speed * dist_current + 1e-8)
            reward += cosine * 0.1 
            
            # 위험하지 않으면 속도 보상
            if cosine > 0.8 and urgency < TTC_THRESHOLD:
                if self.avoidance_timer > 0:
                    reward += speed_ratio * 0.2
                reward += speed_ratio * 0.1

        # --- 목표 달성 로직 ---
        # 실제 타겟(self.target)에 도달했는지 확인
        dist_target = np.linalg.norm(self.target - self.pos)
        
        if dist_target < (self.AGENT_RADIUS + self.TARGET_RADIUS):
            # [기본 보상]
            reward += 50.0 
            self.score += 1
            
            # [관성 보너스 - Alignment Bonus]
            # "방금 타겟을 먹었는데, 내 속도가 이미 다음 타겟(self.next_target)을 향하고 있는가?"
            to_next = self.next_target - self.pos
            dist_next = np.linalg.norm(to_next)
            
            if speed > 10.0 and dist_next > 0:
                vel_dir = self.vel / speed
                next_dir = to_next / dist_next
                alignment = np.dot(vel_dir, next_dir) # -1 ~ 1
                
                if alignment > 0:
                    # 정렬이 잘 되어있고 속도가 빠를수록 큰 보너스
                    bonus = alignment * speed_ratio * 20.0
                    reward += bonus
            
            # [목표 교체]
            # Current -> 삭제, Next -> Current, New -> Next
            self.target = self.next_target
            self.next_target = self._spawn_entity(padding=50, check_overlap=True, extra_point=self.target)
        
        # 우회점(Waypoint) 도달 시 보상은 없음 (그냥 지나가는 길일 뿐)
        # 하지만 너무 가까워지면 우회 모드를 일찍 꺼주어 자연스럽게 다음으로 넘어가게 함
        if self.avoidance_timer > 0 and np.linalg.norm(self.active_waypoint - self.pos) < 30:
            self.avoidance_timer = 0
        
        # 행동 일관성 (떨림 방지)
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
            
            # LIDAR 렌더링
            if hasattr(self, 'current_rays_rendering'):
                for ray_dir, dist, reading in self.current_rays_rendering:
                    c_val = int(255 * reading)
                    c_val = max(0, min(255, c_val))
                    start = self.pos.astype(int)
                    end = (self.pos + ray_dir * dist).astype(int)
                    pygame.draw.line(self.screen, (255-c_val, c_val, 0), start, end, 1)

            # 장애물
            for obs in self.obstacles:
                pygame.draw.circle(self.screen, (100, 100, 100), obs.astype(int), self.OBSTACLE_RADIUS)
                
            # [목표물 렌더링]
            # 1. Next Target (Ghost) - 파란색, 반투명 느낌
            pygame.draw.circle(self.screen, (0, 0, 150), self.next_target.astype(int), self.TARGET_RADIUS, 2)
            
            # 2. Current Target (Real) - 빨간색
            pygame.draw.circle(self.screen, (255, 50, 50), self.target.astype(int), self.TARGET_RADIUS)
            
            # 3. 경로 연결선 (Current -> Next)
            pygame.draw.line(self.screen, (100, 100, 100), self.target.astype(int), self.next_target.astype(int), 1)

            # 4. Agent
            pygame.draw.circle(self.screen, (255, 255, 255), self.pos.astype(int), self.AGENT_RADIUS)
            
            # 5. [스마트 뷰] 현재 AI가 보고 있는 목표(Active Waypoint) 시각화
            if self.active_waypoint is not None:
                # 우회 중이라면 Cyan 색상
                color = (0, 255, 255) if self.avoidance_timer > 0 else (255, 100, 100)
                # 에이전트 -> 현재 목표 선
                pygame.draw.line(self.screen, color, self.pos.astype(int), self.active_waypoint.astype(int), 2)
                # 목표 지점 표시
                if self.avoidance_timer > 0:
                    pygame.draw.circle(self.screen, (0, 255, 255), self.active_waypoint.astype(int), 8)

            speed = np.linalg.norm(self.vel)
            info_texts = [
                f"Score: {self.score}",
                f"Speed: {speed:.1f}",
                f"Mode: {'DETOUR' if self.avoidance_timer > 0 else 'DIRECT'}"
            ]
            
            start_y = 50 
            for i, text in enumerate(info_texts):
                ts = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(ts, (10, start_y + i * 20))
            
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        if self.screen is not None: pygame.quit()

if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    MODEL_PATH = f"final/{current_time}_swingby_2target"
    
    # 1. 학습
    print(">>> Training with Momentum Awareness...")
    env = InertiaRacerEnv()
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
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
    
    model.learn(total_timesteps=300000) # 충분한 학습 필요
    model.save(MODEL_PATH)
    
    # 2. 테스트
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