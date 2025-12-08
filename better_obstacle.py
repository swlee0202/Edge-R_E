import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
from datetime import datetime

# ==========================================
# 1. 커리큘럼 콜백 클래스 (난이도 조절자)
# ==========================================
class CurriculumCallback(BaseCallback):
    """
    학습 단계(Timestep)에 따라 환경의 난이도(Level)를 높이는 콜백
    """
    def __init__(self, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.current_level = 0

    def _on_step(self) -> bool:
        # 학습 진행 상황 확인 (총 30만 스텝 기준 예시)
        step = self.num_timesteps
        
        # 단계별 난이도 설정 조건
        # 0 ~ 8만: Level 0 (장애물 없음) - 이동법 학습
        # 8만 ~ 16만: Level 1 (장애물 1개) - 회피 기초
        # 16만 ~ : Level 2 (장애물 3개) - 심화
        new_level = 0
        if step > 200000:
            new_level = 1
        if step > 400000:
            new_level = 2
            
        # 레벨이 바뀌는 순간 환경에 적용
        if new_level != self.current_level:
            self.current_level = new_level
            print(f"\n[Curriculum] Level Up! Now Level {self.current_level} (Step: {step})")
            
            # 환경의 set_level 함수 호출 (Vectorized Environment 대응)
            self.training_env.env_method("set_level", self.current_level)
            
        return True

# ==========================================
# 2. 환경 클래스 (패딩 적용)
# ==========================================
class InertiaRacerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(InertiaRacerEnv, self).__init__()
        
        self.SCREEN_SIZE = 800
        self.AGENT_RADIUS = 10
        self.TARGET_RADIUS = 15
        self.MAX_SPEED = 150.0
        self.ACCEL_POWER = 1.0
        self.FRICTION = 0.92
        
        self.OBSTACLE_RADIUS = 20
        self.MAX_OBSTACLES = 3 # 최대 장애물 개수 (고정)
        self.current_level = 0 # 현재 난이도 레벨

        # [설정] 안전 거리 (이 거리 안으로 들어오면 불안해함)
        self.SAFE_MARGIN = 60.0
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # [중요] 관측 공간 크기 고정: 6(기본) + (최대 장애물 수 * 2)
        # 장애물이 없어도 항상 이 크기를 유지해야 에러가 안 납니다.
        obs_size = 6 + (self.MAX_OBSTACLES * 4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        # 내부 변수
        self.obstacles = [] # 현재 활성화된 장애물 좌표 리스트

    def set_level(self, level):
        """커리큘럼 콜백에서 호출하는 난이도 설정 함수"""
        self.current_level = level
        # 다음 reset() 때 적용됨

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.pos = np.array([self.SCREEN_SIZE / 2, self.SCREEN_SIZE / 2], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.target_A = self._spawn_target()
        self.target_B = self._spawn_target()
        
        # 레벨에 따른 장애물 개수 결정
        num_obstacles_to_spawn = 0
        if self.current_level == 1:
            num_obstacles_to_spawn = 1
        elif self.current_level >= 2:
            num_obstacles_to_spawn = self.MAX_OBSTACLES
            
        self.obstacles = self._spawn_obstacles(num_obstacles_to_spawn)

        self.score = 0
        self.steps = 0
        self.max_steps = 1000

        return self._get_obs(), {}

    def _spawn_target(self):
        padding = 50
        return np.random.uniform(padding, self.SCREEN_SIZE - padding, size=2).astype(np.float32)

    def _spawn_obstacles(self, count):
        obstacles = []
        for _ in range(count):
            while True:
                pos = np.random.uniform(50, self.SCREEN_SIZE - 50, size=2).astype(np.float32)
                # 안전거리 확보
                safe_dist = 80.0
                if (np.linalg.norm(pos - self.pos) > safe_dist and
                    np.linalg.norm(pos - self.target_A) > safe_dist and
                    np.linalg.norm(pos - self.target_B) > safe_dist):
                    
                    overlap = False
                    for existing in obstacles:
                        if np.linalg.norm(pos - existing) < self.OBSTACLE_RADIUS * 2:
                            overlap = True
                            break
                    if not overlap:
                        obstacles.append(pos)
                        break
        return list(obstacles) # 리스트로 변환

    def _get_obs(self):
        scale = self.SCREEN_SIZE
        
        future_pos = self.pos + (self.vel * 30.0)

        base_obs = [
            self.vel[0] / self.MAX_SPEED,
            self.vel[1] / self.MAX_SPEED,
            (self.target_A[0] - self.pos[0]) / scale,
            (self.target_A[1] - self.pos[1]) / scale,
            (self.target_B[0] - self.pos[0]) / scale,
            (self.target_B[1] - self.pos[1]) / scale
        ]
        
        # [핵심] 장애물 정보 패딩 (Padding)
        # 실제 존재하는 장애물 좌표를 넣고, 남는 자리는 '화면 밖' 좌표로 채움
        obstacle_obs = []
        
        # 1. 존재하는 장애물 넣기
        for obs_pos in self.obstacles:
            rel_x = (obs_pos[0] - self.pos[0]) / scale
            rel_y = (obs_pos[1] - self.pos[1]) / scale
            fut_rel_x = (obs_pos[0] - future_pos[0]) / scale
            fut_rel_y = (obs_pos[1] - future_pos[1]) / scale
            obstacle_obs.extend([rel_x, rel_y, fut_rel_x, fut_rel_y])            

        # 2. 남는 공간 채우기 (Dummy Values)
        # MAX_OBSTACLES 개수를 채울 때까지 더미 데이터 추가
        remaining_slots = self.MAX_OBSTACLES - len(self.obstacles)
        for _ in range(remaining_slots):
            # 상대 좌표 (2.0, 2.0)은 화면 밖 아주 먼 곳을 의미 (0~1 범위 밖)
            # 신경망은 "이 값은 무시해도 되는구나"라고 학습함
            obstacle_obs.extend([2.0, 2.0, 2.0, 2.0]) 
            
        return np.array(base_obs + obstacle_obs, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        
        # 1. 물리 엔진 (가속도 -> 속도 -> 위치)
        accel = np.array(action, dtype=np.float32) * self.ACCEL_POWER
        self.vel += accel
        
        speed = np.linalg.norm(self.vel)
        if speed > self.MAX_SPEED:
            self.vel = (self.vel / speed) * self.MAX_SPEED
            
        self.vel *= self.FRICTION
        self.pos += self.vel

        # 기본 보상 초기화
        reward = -0.01 
        terminated = False
        
        # -----------------------------------------------------------
        # [핵심 로직] 장애물 감지 및 회피 유도 (Raycasting 느낌)
        # -----------------------------------------------------------
        current_speed = np.linalg.norm(self.vel)
        is_blocked = False # 장애물이 내 앞길을 막고 있는가?

        for obs_pos in self.obstacles:
            # 벡터 계산: 나 -> 장애물
            to_obs = obs_pos - self.pos
            dist_obs = np.linalg.norm(to_obs)
            
            # 1. 충돌 처리 (죽음)
            if dist_obs < (self.AGENT_RADIUS + self.OBSTACLE_RADIUS):
                terminated = True
                reward = -50.0
                return self._get_obs(), reward, terminated, False, {}

            # 2. [수정됨] 회피 유도 로직
            # "거리가 가까운데(150px 이내), 내 속도 방향이 장애물을 향하고 있는가?"
            if dist_obs < 150.0 and current_speed > 1.0:
                # 내 속도 벡터(단위)와 장애물 방향 벡터(단위)의 내적
                # 값이 1.0에 가까우면 정면충돌 코스, 0이면 수직(회피), -1이면 도망
                obs_dir = to_obs / dist_obs
                vel_dir = self.vel / current_speed
                collision_alignment = np.dot(vel_dir, obs_dir)
                
                # 정면 30도 내외(cos(30) ≈ 0.86)로 장애물이 있다면 "막혔다"고 판단
                if collision_alignment > 0.8:
                    is_blocked = True
                    
                    # (A) "핸들 꺾어!" 페널티
                    # 장애물 정면으로 빨리 갈수록 감점을 크게 줌
                    # 예: 속도 100으로 정면 돌진 시 -> -1.0 * 1.0 * 0.5 = -0.5점 매 프레임
                    penalty = collision_alignment * (current_speed / self.MAX_SPEED) * 0.5
                    reward -= penalty
                    
                    # 안전거리(SAFE_MARGIN) 침범 시 추가 공포 페널티
                    if dist_obs < self.SAFE_MARGIN:
                         reward -= 1.0 # 아주 강력하게 "여긴 아니야"라고 신호

        # -----------------------------------------------------------
        # 벽 충돌 처리
        if self.pos[0] < 0 or self.pos[0] > self.SCREEN_SIZE:
            self.vel[0] = 0; self.pos[0] = np.clip(self.pos[0], 0, self.SCREEN_SIZE); reward -= 0.1
        if self.pos[1] < 0 or self.pos[1] > self.SCREEN_SIZE:
            self.vel[1] = 0; self.pos[1] = np.clip(self.pos[1], 0, self.SCREEN_SIZE); reward -= 0.1
        
        # 멈춤 방지 (최소 속도 유지)
        if current_speed < 10.0: 
            reward -= 0.05

        # -----------------------------------------------------------
        # 3. 목표물 방향 보상 (조건부 지급!)
        # -----------------------------------------------------------
        to_target = self.target_A - self.pos
        dist_to_target = np.linalg.norm(to_target)
        
        # [중요] 장애물이 앞을 막고 있다면(is_blocked), 목표물 접근 점수를 0으로 만듦!
        # 이렇게 하면 "목표물로 가는 이득"이 사라지고 "장애물 페널티"만 남으므로
        # 에이전트는 살기 위해 핸들을 꺾게 됩니다.
        
        if not is_blocked and current_speed > 0.1:
            # 장애물이 없을 때만 평소처럼 방향 점수를 줌
            cosine_sim = np.dot(self.vel, to_target) / (current_speed * dist_to_target + 1e-8)
            if cosine_sim > 0:
                dist_factor = np.clip(dist_to_target / 300.0, 0.0, 1.0)
                speed_ratio = current_speed / 30
                speed_score = (dist_factor * speed_ratio) + ((1.0 - dist_factor) * (1.0 - speed_ratio))
                reward += cosine_sim * speed_score * 0.1
            else:
                reward += cosine_sim * 0.05
        elif is_blocked:
             # 막혔는데 목표물 보상을 주면 들이받음. 
             # 여기선 오히려 목표물 방향보다는 "현재 속도 유지"에 대한 약한 페널티를 줄 수도 있음.
             pass 

        # 4. 목표물 획득 판정
        if dist_to_target < (self.AGENT_RADIUS + self.TARGET_RADIUS):
            reward += 35.0
            self.score += 1
            
            # 관성 보너스 (Alignment Bonus)
            vec_A_to_B = self.target_B - self.target_A
            dist_A_to_B = np.linalg.norm(vec_A_to_B)
            
            if dist_A_to_B > 0 and current_speed > 0.1:
                dir_A_to_B = vec_A_to_B / dist_A_to_B
                dir_vel = self.vel / current_speed
                alignment = np.dot(dir_vel, dir_A_to_B)
                
                if alignment > 0:
                    dist_factor = np.clip(dist_A_to_B / 400.0, 0.0, 1.0)
                    speed_ratio = current_speed / self.MAX_SPEED
                    speed_score = (dist_factor * speed_ratio) + ((1.0 - dist_factor) * (1.0 - speed_ratio))
                    reward += alignment * speed_score * 40.0

            self.target_A = self.target_B
            self.target_B = self._spawn_target()

        truncated = self.steps >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
                pygame.display.set_caption("Inertia Racer: Safety First")
                self.clock = pygame.time.Clock()

            self.screen.fill((30, 30, 30))

            for obs_pos in self.obstacles:
                # 안전 거리 (점선 느낌의 얇은 원)
                pygame.draw.circle(self.screen, (50, 50, 50), obs_pos.astype(int), int(self.SAFE_MARGIN), 1)
                # 실제 장애물
                pygame.draw.circle(self.screen, (100, 100, 100), obs_pos.astype(int), self.OBSTACLE_RADIUS)
                pygame.draw.circle(self.screen, (200, 50, 50), obs_pos.astype(int), self.OBSTACLE_RADIUS, 2)

            pygame.draw.circle(self.screen, (100, 100, 255), self.target_B.astype(int), self.TARGET_RADIUS, 2)
            pygame.draw.line(self.screen, (100, 100, 255), self.target_A.astype(int), self.target_B.astype(int), 1)
            pygame.draw.circle(self.screen, (255, 50, 50), self.target_A.astype(int), self.TARGET_RADIUS)
            pygame.draw.circle(self.screen, (255, 255, 255), self.pos.astype(int), self.AGENT_RADIUS)
            
            # 미래 위치 예측선 그리기 (디버깅용)
            future_pos = self.pos + (self.vel * 30.0)
            pygame.draw.line(self.screen, (255, 255, 0), self.pos.astype(int), future_pos.astype(int), 1)
            pygame.draw.circle(self.screen, (255, 255, 0), future_pos.astype(int), 3)

            # UI
            font = pygame.font.SysFont("Arial", 20)
            self.screen.blit(font.render(f"Score: {self.score}", True, (255, 255, 255)), (10, 10))
            self.screen.blit(font.render(f"Level: {self.current_level}", True, (255, 200, 50)), (10, 35))
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None: pygame.quit()

# ==========================================
# 3. 메인 실행 (학습 + 커리큘럼 적용)
# ==========================================
if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    MODEL_PATH = f"exp/model_curriculum_{current_time}.zip"
    
    # 총 학습 스텝 (커리큘럼 단계를 모두 소화할 만큼 넉넉히)
    TOTAL_TIMESTEPS = 700000 
    
    if not os.path.exists(MODEL_PATH):
        print(f">>> 커리큘럼 학습 시작 (총 {TOTAL_TIMESTEPS} Steps)")
        print(">>> Level 0: 0 ~ 80k (기본 주행)")
        print(">>> Level 1: 80k ~ 160k (장애물 1개)")
        print(">>> Level 2: 160k ~ (장애물 3개)")
        
        train_env = InertiaRacerEnv(render_mode=None)
        
        # 커리큘럼 콜백 생성
        curriculum_callback = CurriculumCallback()
        
        model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=0.0003)
        
        # 학습 시 callback 인자에 넣어줍니다.
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=curriculum_callback)
        
        model.save(MODEL_PATH)
        print(">>> 학습 완료!")
        train_env.close()

    # --- 테스트 ---
    print(">>> 시각화 (최종 난이도 Level 2로 테스트)")
    env = InertiaRacerEnv(render_mode="human")
    # 테스트 할 때는 최고 난이도로 설정
    env.set_level(2) 
    
    model = PPO.load(MODEL_PATH)
    
    obs, _ = env.reset()
    running = True
    while running:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()