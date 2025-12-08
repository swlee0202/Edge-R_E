import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3 import PPO
import os

# ==========================================
# 1. 환경 정의 (Physics & Game Logic)
# ==========================================
class InertiaRacerEnv(gym.Env):
    """
    관성 주행 환경
    - 목표: A를 줍고, 바로 B로 향해야 함 (멈추지 않고 지나치기)
    - 상태: 내 속도, A와의 거리, B와의 거리
    - 행동: X, Y축 가속도 조절
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    #p_obs_gain=40.633, p_safety_margin=1.8837, p_lookahead=3.616, p_shaping=2.4
    def __init__(self, render_mode=None, p_obs_gain=64.6305, p_safety_margin=1.9193, p_lookahead=199.8128, p_shaping=3.4954):
        super(InertiaRacerEnv, self).__init__()
        
        # --- 설정 상수 ---
        self.p_obs_gain = p_obs_gain
        self.p_safety_margin = p_safety_margin
        self.p_lookahead = p_lookahead
        self.p_shaping = p_shaping

        self.SCREEN_SIZE = 800
        self.AGENT_RADIUS = 10
        self.TARGET_RADIUS = 15
        self.MAX_SPEED = 150.0
        self.ACCEL_POWER = 1.0
        self.FRICTION = 0.92  # 공기 저항 (1.0이면 저항 없음)
        self.obstacle_radius = 30.0

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # --- Action Space: [가속도X, 가속도Y] (-1.0 ~ 1.0) ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # --- Observation Space: 정규화된 관측값 (크기 6) ---
        # [내Vx, 내Vy, A_RelX, A_RelY, B_RelX, B_RelY]
        # 값의 범위는 대략 -inf ~ inf 지만 학습을 위해 정규화된 값을 주는 게 좋음
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 에이전트 초기화 (화면 중앙)
        self.pos = np.array([self.SCREEN_SIZE / 2, self.SCREEN_SIZE / 2], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)

        # 2. 목표물 초기화 (A와 B 생성)
        self.target_A = self._spawn_target()
        self.target_B = self._spawn_target()
        self.prev_dist_to_A = np.linalg.norm(self.pos - self.target_A)

        while True:
            self.obstacle_pos = np.random.uniform(100, self.SCREEN_SIZE-100, size=2).astype(np.float32)
            if np.linalg.norm(self.obstacle_pos - self.pos) > 100:
                break

        self.score = 0
        self.steps = 0
        self.max_steps = 1000 # 한 에피소드 최대 길이

        return self._get_obs(), {}

    def _spawn_target(self):
        # 화면 가장자리를 제외한 랜덤 위치 반환
        padding = 50
        return np.random.uniform(padding, self.SCREEN_SIZE - padding, size=2).astype(np.float32)

    def _get_obs(self):
        # 신경망에 넣어줄 데이터 (상대 좌표로 변환하여 학습 효율 증대)
        # 모든 값을 대략 -1 ~ 1 사이로 스케일링
        scale = self.SCREEN_SIZE
        vec_to_obs = self.obstacle_pos - self.pos
        dist_to_obs = np.linalg.norm(vec_to_obs) + 1e-5
        closing_speed = np.dot(self.vel, vec_to_obs / dist_to_obs) / self.MAX_SPEED
        obs = np.array([
            self.vel[0] / self.MAX_SPEED,     # 속도 X
            self.vel[1] / self.MAX_SPEED,     # 속도 Y
            (self.target_A[0] - self.pos[0]) / scale, # A와의 거리 X
            (self.target_A[1] - self.pos[1]) / scale, # A와의 거리 Y
            (self.target_B[0] - self.pos[0]) / scale, # B와의 거리 X
            (self.target_B[1] - self.pos[1]) / scale,  # B와의 거리 Y
            (self.obstacle_pos[0] - self.pos[0]) / scale, #장애물과의 거리 X
            (self.obstacle_pos[1] - self.pos[1]) / scale, #장애물과의 거리 Y
            closing_speed,
        ], dtype=np.float32)
        return obs
   
    def step(self, action):
        self.steps += 1
        
        # 1. 물리 엔진
        accel = np.array(action, dtype=np.float32) * self.ACCEL_POWER
        self.vel += accel
        speed = np.linalg.norm(self.vel)
        if speed > self.MAX_SPEED:
            self.vel = (self.vel / speed) * self.MAX_SPEED
        self.vel *= self.FRICTION
        self.pos += self.vel

        # -----------------------------------------------------
        # 2. 보상 로직
        # -----------------------------------------------------
        reward = 0.0
        terminated = False
        
        dist_to_A = np.linalg.norm(self.pos - self.target_A)
        to_obs_vec = self.obstacle_pos - self.pos 
        dist_to_obs = np.linalg.norm(to_obs_vec)
        
        # [파라미터 1] 위험 감지 거리 (대폭 상향)
        # 관성이 있으므로 150px은 너무 짧습니다. 300px 이상 봐야 미리 피합니다.
        # 속도에 비례해서 더 멀리 보게 설정 (Dynamic Lookahead)
        detection_radius = self.obstacle_radius + self.AGENT_RADIUS + self.p_lookahead + (speed * 10.0)
        is_in_danger = dist_to_obs < detection_radius

        # 거리 쉐이핑 (위험할 땐 0.5배로 줄여서 생존 우선순위 높임)
        current_shaping_weight = self.p_shaping 
        if is_in_danger:
            current_shaping_weight = 0.5
        
        reward += (self.prev_dist_to_A - dist_to_A) * current_shaping_weight
        self.prev_dist_to_A = dist_to_A

        # -------------------------------------------------------------------
        # [핵심] 기하학적 정밀 충돌 판정 (Geometric Collision Cone)
        # -------------------------------------------------------------------
        if is_in_danger and speed > 0.5:
            # 1. 현재 진행 방향과 장애물 간의 각도 (Theta Current)
            to_obs_dir = to_obs_vec / (dist_to_obs + 1e-5)
            vel_dir = self.vel / speed
            
            # 내적값(-1~1)을 각도(0~PI 라디안)로 변환
            dot_prod = np.clip(np.dot(vel_dir, to_obs_dir), -1.0, 1.0)
            current_angle = np.arccos(dot_prod) # 0이면 정면, PI면 반대
            
            # 2. 회피에 필요한 최소 각도 (Theta Limit) 계산
            # 반지름 합(플레이어+장애물)이 차지하는 각도 계산
            combined_radius = self.obstacle_radius + self.AGENT_RADIUS
            
            # arcsin 입력값 안전장치 (거리가 반지름보다 작으면 이미 겹친 것)
            ratio = np.clip(combined_radius / (dist_to_obs + 1e-5), 0, 1.0)
            limit_angle = np.arcsin(ratio)
            
            # [중요] 여유 마진 (Safety Margin)
            # 딱 맞춰서 피하면 긁힙니다. 1.5배 정도 더 여유 있게 피하게 만듭니다.
            safe_angle_threshold = limit_angle * self.p_safety_margin
            
            # 3. 충돌 코스 판정
            # "현재 각도가 안전 각도보다 작으면" -> 충돌 코스임
            if current_angle < safe_angle_threshold:
                
                # [파라미터 2] 공포 계수 (여전히 강력하게)
                current_obs_gain = self.p_obs_gain
                
                # (1) 각도 위반 정도 (얼마나 정면인가?)
                # 0(정면)일수록 큼, 경계선일수록 작음
                angle_penetration = (safe_angle_threshold - current_angle) / safe_angle_threshold
                
                # (2) 거리 위협 정도 (가까울수록 급격히 커짐)
                dist_factor = (detection_radius - dist_to_obs) / detection_radius
                
                # (3) 속도 위협 정도 (빠르면 더 위험)
                speed_factor = speed / self.MAX_SPEED
                
                # 최종 페널티: 각도 * 거리 * 속도 * 가중치
                penalty = angle_penetration * dist_factor * speed_factor * current_obs_gain
                reward -= penalty
                
                # 4. [정면 충돌 방지] 브레이크 보상 유도
                # 충돌 코스인데 속도가 빠르면, "가속 페달을 밟는 것" 자체를 처벌
                # 반대로 감속하면 처벌 안 함 (브레이크 유도)
                accel_magnitude = np.linalg.norm(accel)
                if accel_magnitude > 0:
                    # 가속 방향이 장애물 쪽이면 2배 처벌
                    if np.dot(accel, to_obs_dir) > 0:
                        reward -= penalty * 1.0
                    else:
                        reward -= penalty * 0.2 # 회피 가속은 조금만 처벌

        # -------------------------------------------------------------------

        # 3. 충돌 처리 (즉사)
        if dist_to_obs < (self.AGENT_RADIUS + self.obstacle_radius):
            reward -= 500.0 # 타협 없는 죽음
            terminated = True
            return self._get_obs(), reward, terminated, False, {}
        
        # 4. 벽 및 시간 페널티
        reward -= 0.01 
        
        hit_wall = False
        for i in range(2):
            if self.pos[i] < 0: 
                self.pos[i] = 0; self.vel[i] *= -0.5; hit_wall = True
            if self.pos[i] > self.SCREEN_SIZE: 
                self.pos[i] = self.SCREEN_SIZE; self.vel[i] *= -0.5; hit_wall = True
        
        if hit_wall:
            reward -= 2.0

        # 5. 목표 획득
        if dist_to_A < (self.AGENT_RADIUS + self.TARGET_RADIUS):
            reward += 30.0
            self.score += 1
            
            # Alignment Bonus (속도감 유지)
            vec_A_to_B = self.target_B - self.target_A
            dist_A_to_B = np.linalg.norm(vec_A_to_B)
            
            if dist_A_to_B > 0 and speed > 0.1:
                dir_A_to_B = vec_A_to_B / dist_A_to_B
                vel_dir = self.vel / speed
                alignment = np.dot(vel_dir, dir_A_to_B)
                
                if alignment > 0:
                    reward += alignment * (speed / self.MAX_SPEED) * 20.0

            self.target_A = self.target_B
            self.target_B = self._spawn_target()
            self.prev_dist_to_A = np.linalg.norm(self.pos - self.target_A)
        
        truncated = self.steps >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}
    

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
                pygame.display.set_caption("Inertia Racer RL Visualization")
                self.clock = pygame.time.Clock()

            self.screen.fill((30, 30, 30)) # 배경

            pygame.draw.circle(self.screen, (100, 100, 100), self.obstacle_pos.astype(int), int(self.obstacle_radius))

            # --- 그리기 도우미 함수 ---
            def draw_vec(start, vec, color, scale=10):
                end = start + vec * scale
                pygame.draw.line(self.screen, color, start.astype(int), end.astype(int), 3)
            # ------------------------

            # 1. 다음 목표물 B
            pygame.draw.circle(self.screen, (100, 100, 255), self.target_B.astype(int), self.TARGET_RADIUS, 2)
            pygame.draw.line(self.screen, (100, 100, 255), self.target_A.astype(int), self.target_B.astype(int), 1)

            # 2. 현재 목표물 A
            pygame.draw.circle(self.screen, (255, 50, 50), self.target_A.astype(int), self.TARGET_RADIUS)

            # 3. 에이전트
            pygame.draw.circle(self.screen, (255, 255, 255), self.pos.astype(int), self.AGENT_RADIUS)

            # 4. 벡터 시각화
            draw_vec(self.pos, self.vel, (0, 255, 0), scale=10) # 속도 (초록)
            
            # --- UI 정보 표시 ---
            font = pygame.font.SysFont("Arial", 20)
            
            # (1) 점수 표시
            score_surf = font.render(f"Score: {self.score}", True, (255, 255, 255))
            self.screen.blit(score_surf, (10, 10))
            
            # (2) [추가됨] 현재 속력 표시
            # np.linalg.norm(self.vel)은 벡터의 길이(속력)를 구합니다.
            current_speed = np.linalg.norm(self.vel)
            
            # 속도가 빠르면(10 이상) 노란색, 아니면 하늘색으로 표시
            color = (255, 255, 0) if current_speed > 10.0 else (0, 255, 255)
            
            speed_surf = font.render(f"Speed: {current_speed:.2f}", True, color)
            self.screen.blit(speed_surf, (10, 35)) # 점수 바로 아래(y=35)에 배치
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()

# ==========================================
# 2. 메인 실행 코드 (학습 및 테스트)
# ==========================================
if __name__ == "__main__":
    MODEL_PATH = "model_prod_minv3_30.zip"
    
    # 환경 생성
    env = InertiaRacerEnv(render_mode="human") # 학습 시엔 "rgb_array" 권장하나, 보는 맛을 위해 human

    # --- A. 학습 (모델이 없으면 학습 시작) ---
    if not os.path.exists(MODEL_PATH):
        print(">>> 새로운 모델 학습을 시작합니다 (약 50,000 steps)...")
        print(">>> 학습 중에는 화면이 뜨지 않거나 검게 보일 수 있습니다.")
        
        # 시각화 없이 빠르게 학습하기 위해 더미 환경 생성
        train_env = InertiaRacerEnv(render_mode=None)
        
        # PPO 모델 생성
        model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=0.0003)
        
        # 학습 실행 (약 3~5분 소요, steps를 늘리면 더 똑똑해짐)
        model.learn(total_timesteps=100000)
        
        # 저장 (현재는 안함)
        model.save(MODEL_PATH)
        print(">>> 학습 완료 및 저장됨!")
        train_env.close()

    # --- B. 테스트 및 시각화 ---
    print(">>> 학습된 모델을 불러와 시각화합니다.")
    
    # 모델 로드
    model = PPO.load(MODEL_PATH)
    
    # 테스트 루프
    obs, _ = env.reset()
    running = True
    while running:
        # AI가 행동 결정
        action, _ = model.predict(obs, deterministic=True)
        
        # 환경에 적용
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # 화면 그리기
        env.render()
        
        # Pygame 종료 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
