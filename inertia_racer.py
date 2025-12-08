import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3 import PPO
import os
from datetime import datetime

class InertiaRacerEnv(gym.Env):
    """
    관성 주행 환경
    - 목표: A를 줍고, 바로 B로 향해야 함 (멈추지 않고 지나치기)
    - 상태: 내 속도, A와의 거리, B와의 거리
    - 행동: X, Y축 가속도 조절
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(InertiaRacerEnv, self).__init__()
        
        # --- 설정 상수 ---
        self.SCREEN_SIZE = 800
        self.AGENT_RADIUS = 10
        self.TARGET_RADIUS = 15
        self.MAX_SPEED = 150.0
        self.ACCEL_POWER = 1.0
        self.FRICTION = 0.92  # 공기 저항 (1.0이면 저항 없음)
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # --- Action Space: [가속도X, 가속도Y] (-1.0 ~ 1.0) ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # --- Observation Space: 정규화된 관측값 (크기 6) ---
        # [내Vx, 내Vy, A_RelX, A_RelY, B_RelX, B_RelY]
        # 값의 범위는 대략 -inf ~ inf 지만 학습을 위해 정규화된 값을 주는 게 좋음
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 에이전트 초기화 (화면 중앙)
        self.pos = np.array([self.SCREEN_SIZE / 2, self.SCREEN_SIZE / 2], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)

        # 2. 목표물 초기화 (A와 B 생성)
        self.target_A = self._spawn_target()
        self.target_B = self._spawn_target()
        
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
        
        obs = np.array([
            self.vel[0] / self.MAX_SPEED,     # 속도 X
            self.vel[1] / self.MAX_SPEED,     # 속도 Y
            (self.target_A[0] - self.pos[0]) / scale, # A와의 거리 X
            (self.target_A[1] - self.pos[1]) / scale, # A와의 거리 Y
            (self.target_B[0] - self.pos[0]) / scale, # B와의 거리 X
            (self.target_B[1] - self.pos[1]) / scale  # B와의 거리 Y
        ], dtype=np.float32)
        return obs

    def step(self, action):
        self.steps += 1
        
        # 1. 물리 엔진 적용 (가속도 -> 속도 -> 위치)
        accel = np.array(action, dtype=np.float32) * self.ACCEL_POWER
        self.vel += accel
        
        # 속도 제한 (Max Speed)
        speed = np.linalg.norm(self.vel)
        if speed > self.MAX_SPEED:
            self.vel = (self.vel / speed) * self.MAX_SPEED
            
        # 마찰력 및 위치 업데이트
        self.vel *= self.FRICTION
        self.pos += self.vel

        # 2. 벽 충돌 처리 (화면 밖으로 나가면 튕김 or 멈춤)
        # 여기서는 멈추게 하고 페널티를 줌 (학습 가속화)
        reward = -0.01 # 기본 시간 페널티 (빨리 움직여라)
        to_target = self.target_A - self.pos
        dist_to_target = np.linalg.norm(to_target)
        current_speed = np.linalg.norm(self.vel)
        if current_speed<0.05:
            reward-=40
            
            
        # --- 방향 페널티 로직 (속도 고려 버전) ---
        if current_speed > 0.1: # 멈춰있지 않을 때만 계산
            # 1. 코사인 유사도 (방향): 1.0(정면) ~ -1.0(반대)
            cosine_sim = np.dot(self.vel, to_target) / (current_speed * dist_to_target + 1e-8)
            
            # 2. 방향이 맞을 때만(+), 거리와 속도 궁합을 봅니다
            if cosine_sim > 0:
                # (1) 거리 계수 (Distance Factor)
                # 300px 이상이면 "멀다(1.0)", 가까우면 "가깝다(0.0)"
                # 300.0은 제동이 필요한 거리 기준 (튜닝 가능)
                dist_factor = np.clip(dist_to_target / 300.0, 0.0, 1.0)
                
                # (2) 속도 계수 (Speed Ratio)
                speed_ratio = current_speed / 30
                
                # (3) 상황별 점수 (Mix)
                # - 멀다(1.0) -> 빠를수록(1.0) 점수 높음
                # - 가깝다(0.0) -> 느릴수록(0.0) 점수 높음 (1 - speed_ratio)
                speed_score = (dist_factor * speed_ratio) + ((1.0 - dist_factor) * (1.0 - speed_ratio))
                
                # 최종 보상: 방향(1.0) * 속도적절성(1.0) * 가중치(0.1)
                # 방향도 맞고, 거리에 맞는 속도라면 점수를 줌
                reward += cosine_sim * speed_score * 0.1
            
            else:
                # 방향이 틀렸으면(-), 그냥 감점 (속도 고려 없이 단순하게)
                # 잘못된 방향으로 빨리 가는 건 최악이므로 페널티를 줌
                reward += cosine_sim * 0.05
        
        if self.pos[0] < 0 or self.pos[0] > self.SCREEN_SIZE:
            self.vel[0] = 0
            self.pos[0] = np.clip(self.pos[0], 0, self.SCREEN_SIZE)
            reward -= 0.1 # 벽 박으면 감점
            
        if self.pos[1] < 0 or self.pos[1] > self.SCREEN_SIZE:
            self.vel[1] = 0
            self.pos[1] = np.clip(self.pos[1], 0, self.SCREEN_SIZE)
            reward -= 0.1

        # 3. 목표물 획득 판정
        dist_to_A = np.linalg.norm(self.pos - self.target_A)
        
        if dist_to_A < (self.AGENT_RADIUS + self.TARGET_RADIUS):
            # --- [기본 보상] ---
            reward += 35.0 # 획득 점수
            self.score += 1
            
            # --- [추가된 부분: 관성 보너스 (Alignment Bonus)] ---
            # "A를 먹는 순간, 내 속도가 이미 B를 향하고 있는가?"
            
            # 1. A에서 B로 가는 방향 벡터 계산
            vec_A_to_B = self.target_B - self.target_A
            dist_A_to_B = np.linalg.norm(vec_A_to_B)
            
            # 2. 내 속도 방향 벡터 계산
            current_speed = np.linalg.norm(self.vel)
            
            if dist_A_to_B > 0 and current_speed > 0.1:
                # 정규화 (크기를 1로 만듦)
                dir_A_to_B = vec_A_to_B / dist_A_to_B
                dir_vel = self.vel / current_speed
                
                # 3. 내적(Dot Product) 계산
                alignment = np.dot(dir_vel, dir_A_to_B)
                
                # 4. [수정됨] 거리 기반 가변 보너스 지급
                if alignment > 0:
                    # (1) 거리 계수 (Distance Factor) 계산
                    # 300px 이상이면 "멀다(1.0)", 0px에 가까우면 "가깝다(0.0)"
                    # 300.0은 튜닝 가능한 상수 (화면 크기 800 기준 적절한 제동 거리)
                    dist_factor = np.clip(dist_A_to_B / 400.0, 0.0, 1.0)
                    
                    # (2) 속도 계수 (Speed Ratio) 계산 (0.0 ~ 1.0)
                    speed_ratio = current_speed / self.MAX_SPEED
                    
                    # (3) 상황별 점수 계산 (Mix)
                    # - 멀 때(dist 1.0): 빠를수록(speed 1.0) 점수 높음
                    # - 가까울 때(dist 0.0): 느릴수록(speed 0.0) 점수 높음 (1 - speed_ratio)
                    speed_score = (dist_factor * speed_ratio) + ((1.0 - dist_factor) * (1.0 - speed_ratio))
                    
                    # 최종 보너스: (방향 정확도) * (거리별 적절 속도) * 가중치
                    # 예: 방향 완벽(1.0) * 속도 적절(1.0) * 15.0 = 15점 추가
                    bonus = alignment * speed_score * 40.0
                    reward += bonus

            # -------------------------------------------------------

            # 목표물 교체 로직 (보너스 계산 후에 해야 함!)
            self.target_A = self.target_B
            self.target_B = self._spawn_target()
            
        # else:
        #     # 쉐이핑 보상: 목표물에 다가갈수록 약간의 점수 (학습 초반용)
        #     # 속도 벡터와 목표물 방향 벡터의 내적(Dot Product)을 사용
        #     to_target = self.target_A - self.pos
        #     to_target /= (np.linalg.norm(to_target) + 1e-5) # 단위벡터
        #     velocity_towards_target = np.dot(self.vel, to_target)

        #     if velocity_towards_target > 0:
        #         reward += 0.005 * (velocity_towards_target / self.MAX_SPEED)

        # 4. 종료 조건
        terminated = False
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
            if current_speed<0.05: color=(255,0,0)
            
            speed_surf = font.render(f"Speed: {current_speed:.2f}", True, color)
            self.screen.blit(speed_surf, (10, 35)) # 점수 바로 아래(y=35)에 배치
            
            step_color = (200, 200, 200)
            if self.steps > self.max_steps * 0.9:
                step_color = (255, 100, 100)
                
            step_surf = font.render(f"Step: {self.steps}/{self.max_steps}", True, step_color)
            self.screen.blit(step_surf, (10, 60)) # 속도 아래(y=60)에 배치
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            

    def close(self):
        if self.screen is not None:
            pygame.quit()

# ==========================================
# 2. 메인 실행 코드 (학습 및 테스트)
# ==========================================
if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    MODEL_PATH = f"exp/model_{current_time}.zip"
    
    # 환경 생성
    env = InertiaRacerEnv(render_mode="human") # 학습 시엔 "rgb_array" 권장하나, 보는 맛을 위해 human

    # --- A. 학습 (모델이 없으면 학습 시작) ---
    if not os.path.exists(MODEL_PATH):
        print(">>> 새로운 모델 학습을 시작합니다 (약 50,000 steps)...")
        print(">>> 학습 중에는 화면이 뜨지 않거나 검게 보일 수 있습니다.")
        
        # 시각화 없이 빠르게 학습하기 위해 더미 환경 생성
        train_env = InertiaRacerEnv(render_mode="human")
        
        # PPO 모델 생성
        model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=0.0003)
        
        # 학습 실행 (약 3~5분 소요, steps를 늘리면 더 똑똑해짐)
        model.learn(total_timesteps=300000)
        
        # 저장
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