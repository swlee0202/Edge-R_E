import gymnasium as gym
from gymnasium import spaces
import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# ==========================================
# 1. 파라미터화된 환경 클래스
# ==========================================
class InertiaRacerEnv(gym.Env):
    def __init__(self, p_obs_gain=60.0, p_safety_margin=1.5, p_lookahead=250.0, p_shaping=2.0):
        super(InertiaRacerEnv, self).__init__()
        
        # --- 하이퍼 파라미터 저장 ---
        self.p_obs_gain = p_obs_gain
        self.p_safety_margin = p_safety_margin
        self.p_lookahead = p_lookahead
        self.p_shaping = p_shaping

        # --- 환경 설정 상수 ---
        self.SCREEN_SIZE = 800
        self.AGENT_RADIUS = 10
        self.TARGET_RADIUS = 15
        self.MAX_SPEED = 15.0
        self.ACCEL_POWER = 0.5
        self.FRICTION = 0.98
        self.obstacle_radius = 30.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 관측 공간: [Vx, Vy, Tx, Ty, NextTx, NextTy, ObsX, ObsY] (8개)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([self.SCREEN_SIZE / 2, self.SCREEN_SIZE / 2], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        
        self.target_A = self._spawn_target()
        self.target_B = self._spawn_target()
        
        while True:
            self.obstacle_pos = np.random.uniform(100, self.SCREEN_SIZE - 100, size=2).astype(np.float32)
            if np.linalg.norm(self.obstacle_pos - self.pos) > 200:
                break
                
        self.prev_dist_to_A = np.linalg.norm(self.pos - self.target_A)
        self.score = 0
        self.steps = 0
        self.max_steps = 1000
        return self._get_obs(), {}

    def _spawn_target(self):
        return np.random.uniform(50, self.SCREEN_SIZE - 50, size=2).astype(np.float32)

    def _get_obs(self):
        scale = self.SCREEN_SIZE
        obs = np.array([
            self.vel[0] / self.MAX_SPEED,
            self.vel[1] / self.MAX_SPEED,
            (self.target_A[0] - self.pos[0]) / scale,
            (self.target_A[1] - self.pos[1]) / scale,
            (self.target_B[0] - self.pos[0]) / scale,
            (self.target_B[1] - self.pos[1]) / scale,
            (self.obstacle_pos[0] - self.pos[0]) / scale,
            (self.obstacle_pos[1] - self.pos[1]) / scale
        ], dtype=np.float32)
        return obs

    # ----------------------------------------------------------------
    # 사용자님이 제공한 step 함수 (상수 -> 변수 교체)
    # ----------------------------------------------------------------
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
        
        # [파라미터 적용] 위험 감지 거리 (Dynamic Lookahead)
        # p_lookahead 사용
        detection_radius = self.obstacle_radius + self.AGENT_RADIUS + self.p_lookahead + (speed * 10.0)
        is_in_danger = dist_to_obs < detection_radius

        # 거리 쉐이핑
        # p_shaping 사용
        current_shaping_weight = self.p_shaping 
        if is_in_danger:
            current_shaping_weight = 0.5 
            
        reward += (self.prev_dist_to_A - dist_to_A) * current_shaping_weight
        self.prev_dist_to_A = dist_to_A

        # -------------------------------------------------------------------
        # [핵심] 기하학적 정밀 충돌 판정
        # -------------------------------------------------------------------
        if is_in_danger and speed > 0.5:
            to_obs_dir = to_obs_vec / (dist_to_obs + 1e-5)
            vel_dir = self.vel / speed
            
            dot_prod = np.clip(np.dot(vel_dir, to_obs_dir), -1.0, 1.0)
            current_angle = np.arccos(dot_prod)
            
            combined_radius = self.obstacle_radius + self.AGENT_RADIUS
            ratio = np.clip(combined_radius / (dist_to_obs + 1e-5), 0, 1.0)
            limit_angle = np.arcsin(ratio)
            
            # [파라미터 적용] 여유 마진 (Safety Margin)
            # p_safety_margin 사용
            safe_angle_threshold = limit_angle * self.p_safety_margin
            
            if current_angle < safe_angle_threshold:
                
                # [파라미터 적용] 공포 계수
                # p_obs_gain 사용
                current_obs_gain = self.p_obs_gain
                
                angle_penetration = (safe_angle_threshold - current_angle) / safe_angle_threshold
                dist_factor = (detection_radius - dist_to_obs) / detection_radius
                speed_factor = speed / self.MAX_SPEED
                
                penalty = angle_penetration * dist_factor * speed_factor * current_obs_gain
                reward -= penalty
                
                accel_magnitude = np.linalg.norm(accel)
                if accel_magnitude > 0:
                    if np.dot(accel, to_obs_dir) > 0:
                        reward -= penalty * 1.0
                    else:
                        reward -= penalty * 0.2

        # -------------------------------------------------------------------

        # 3. 충돌 처리 (즉사)
        if dist_to_obs < (self.AGENT_RADIUS + self.obstacle_radius):
            reward -= 500.0 # 학습용 큰 페널티 (평가 로직과는 별개)
            terminated = True
            return self._get_obs(), reward, terminated, False, {}
        
        # 4. 벽 및 시간 페널티
        reward -= 0.01 
        hit_wall = False
        for i in range(2):
            if self.pos[i] < 0: self.pos[i] = 0; self.vel[i] *= -0.5; hit_wall = True
            if self.pos[i] > self.SCREEN_SIZE: self.pos[i] = self.SCREEN_SIZE; self.vel[i] *= -0.5; hit_wall = True
        if hit_wall: reward -= 2.0

        # 5. 목표 획득
        if dist_to_A < (self.AGENT_RADIUS + self.TARGET_RADIUS):
            reward += 30.0
            self.score += 1
            
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


# ==========================================
# 2. Optuna 튜닝 로직
# ==========================================

def evaluate_metrics(model, env_params, n_episodes=5):
    """
    [평가 기준]
    - 장애물 충돌: -10점 (즉시 종료)
    - 생존 및 목표 획득: 획득한 점수(score)
    """
    env = InertiaRacerEnv(**env_params)
    total_score = 0
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        hit_obstacle = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # 학습 코드에서 충돌 시 -500을 주므로, -400 이하면 충돌로 간주
            if reward < -400:
                hit_obstacle = True
            
            done = terminated or truncated
        
        # 평가 점수 계산
        if hit_obstacle:
            total_score += -100.0
        else:
            total_score += env.score

    return total_score / n_episodes

def objective(trial):
    # 1. 튜닝할 파라미터 제안 (범위 설정)
    params = {
        'p_obs_gain': trial.suggest_float('p_obs_gain', 30.0, 100.0),      # 공포심 (낮으면 무시, 높으면 동결)
        'p_safety_margin': trial.suggest_float('p_safety_margin', 1.2, 2.5), # 회피 여유 (좁으면 스침, 넓으면 너무 돎)
        'p_lookahead': trial.suggest_float('p_lookahead', 150.0, 400.0),     # 감지 거리
        'p_shaping': trial.suggest_float('p_shaping', 1.0, 5.0)              # 목표 탐욕
    }
    
    # 2. 환경 생성
    # 튜닝 속도를 위해 병렬 환경(4개) 사용
    train_env = make_vec_env(lambda: InertiaRacerEnv(**params), n_envs=4)
    
    # 3. 모델 학습
    # 튜닝용이므로 빠르고 가볍게 (Steps를 줄임)
    model = PPO("MlpPolicy", train_env, verbose=0, learning_rate=3e-4)
    
    try:
        # 30,000 스텝 정도면 경향성 파악 가능
        model.learn(total_timesteps=100000)
    except Exception as e:
        print(f"Training failed: {e}")
        return -100 # 실패 시 최저점
    
    # 4. 평가
    score = evaluate_metrics(model, params, n_episodes=10)
    
    train_env.close()
    return score

if __name__ == "__main__":
    print(">>> 튜닝 시작 (목표: 충돌 없이 점수 많이 먹기)")
    
    study = optuna.create_study(direction="maximize")
    # 30회 정도 시도 (시간이 오래 걸리면 줄이세요)
    study.optimize(objective, n_trials=50)
    
    print("\n-------------------------------------------")
    print(">>> Best Parameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")
    print(f">>> Best Score: {study.best_value}")
    print("-------------------------------------------")