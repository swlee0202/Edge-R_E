import pygame
import numpy as np
import random

# ==========================================
# 1. AI 환경과 100% 동일한 상수 설정
# ==========================================
SCREEN_SIZE = 800
AGENT_RADIUS = 10
TARGET_RADIUS = 15
MAX_SPEED = 10.0
ACCEL_POWER = 0.5  # AI Action이 -1~1일 때 곱해지는 힘
FRICTION = 0.94    # 공기 저항
FPS = 60           # AI 학습 기준 프레임
MAX_STEPS = 1000   # AI 에피소드 길이 (약 16.66초)

# 색상 정의
COLOR_BG = (30, 30, 30)
COLOR_PLAYER = (255, 255, 255)
COLOR_TARGET_A = (255, 50, 50)   # 현재 목표
COLOR_TARGET_B = (100, 100, 255) # 다음 목표
COLOR_VELOCITY = (0, 255, 0)     
COLOR_ACCEL = (0, 200, 255)      

def spawn_target():
    padding = 50
    return np.array([
        random.uniform(padding, SCREEN_SIZE - padding),
        random.uniform(padding, SCREEN_SIZE - padding)
    ])

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Human vs AI: 1000 Steps Challenge")
    clock = pygame.time.Clock()
    
    font = pygame.font.SysFont("Arial", 24)
    big_font = pygame.font.SysFont("Arial", 60)

    # --- 게임 초기화 함수 ---
    def reset_game():
        pos = np.array([SCREEN_SIZE/2, SCREEN_SIZE/2])
        vel = np.array([0.0, 0.0])
        target_A = spawn_target()
        target_B = spawn_target()
        score = 0
        current_step = 0 # 시간이 아니라 스텝(프레임) 수를 셉니다
        return pos, vel, target_A, target_B, score, current_step

    pos, vel, target_A, target_B, score, current_step = reset_game()
    game_over = False

    running = True
    while running:
        # 1. 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # 재시작
            if game_over and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pos, vel, target_A, target_B, score, current_step = reset_game()
                    game_over = False

        # 2. 게임 로직
        accel_input = np.array([0.0, 0.0])
        
        if not game_over:
            # 스텝 증가
            current_step += 1
            if current_step >= MAX_STEPS:
                game_over = True

            # 키보드 입력 (WASD)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]: accel_input[1] -= 1
            if keys[pygame.K_s]: accel_input[1] += 1
            if keys[pygame.K_a]: accel_input[0] -= 1
            if keys[pygame.K_d]: accel_input[0] += 1

            # 가속도 정규화
            if np.linalg.norm(accel_input) > 0:
                accel_input = accel_input / np.linalg.norm(accel_input)
            
            # --- [물리 엔진] AI 환경과 동일 ---
            # 1. 가속
            vel += accel_input * ACCEL_POWER
            # 2. 속도 제한
            speed = np.linalg.norm(vel)
            if speed > MAX_SPEED:
                vel = (vel / speed) * MAX_SPEED
            # 3. 마찰 및 이동
            vel *= FRICTION
            pos += vel
            # 4. 벽 충돌 (튕기지 않고 멈춤, AI 학습 코드와 동일)
            pos[0] = np.clip(pos[0], 0, SCREEN_SIZE)
            pos[1] = np.clip(pos[1], 0, SCREEN_SIZE)
            if pos[0] == 0 or pos[0] == SCREEN_SIZE: vel[0] = 0
            if pos[1] == 0 or pos[1] == SCREEN_SIZE: vel[1] = 0

            # 목표물 획득
            dist_to_target = np.linalg.norm(pos - target_A)
            if dist_to_target < (AGENT_RADIUS + TARGET_RADIUS):
                score += 1
                target_A = target_B
                target_B = spawn_target()

        # 3. 화면 그리기
        screen.fill(COLOR_BG)

        # 목표물 및 플레이어
        pygame.draw.circle(screen, COLOR_TARGET_B, target_B.astype(int), TARGET_RADIUS, 2)
        pygame.draw.line(screen, COLOR_TARGET_B, target_A.astype(int), target_B.astype(int), 1)
        pygame.draw.circle(screen, COLOR_TARGET_A, target_A.astype(int), TARGET_RADIUS)
        pygame.draw.circle(screen, COLOR_PLAYER, pos.astype(int), AGENT_RADIUS)

        # 벡터 시각화
        end_pos_vel = pos + vel * 10
        pygame.draw.line(screen, COLOR_VELOCITY, pos.astype(int), end_pos_vel.astype(int), 3)
        
        if np.linalg.norm(accel_input) > 0:
            end_pos_acc = pos + accel_input * 30
            pygame.draw.line(screen, COLOR_ACCEL, pos.astype(int), end_pos_acc.astype(int), 2)

        # UI 정보
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        
        # 남은 스텝 및 시간 계산
        steps_left = MAX_STEPS - current_step
        time_left_sec = steps_left / FPS
        
        # 시간이 적으면 빨간색 경고
        timer_color = (255, 255, 255)
        if time_left_sec < 3.0: 
            timer_color = (255, 50, 50)

        timer_text = font.render(f"Time: {time_left_sec:.2f}s ({steps_left} steps)", True, timer_color)

        screen.blit(score_text, (20, 20))
        screen.blit(timer_text, (20, 50))

        # 게임 오버 화면
        if game_over:
            over_bg = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE))
            over_bg.set_alpha(150)
            over_bg.fill((0, 0, 0))
            screen.blit(over_bg, (0, 0))
            
            # AI와 비교 메시지
            ai_par = "Wait..."
            if score>=20: ai_par = "you are ai!"
            elif score>=16: ai_par = "god!"
            elif score>=12: ai_par = "good!"
            elif score >= 8: ai_par = "Human!"
            elif score >= 6: ai_par = "Noob"
            else: ai_par = "Try Again"

            end_msg1 = big_font.render("TIME UP!", True, (255, 50, 50))
            end_msg2 = big_font.render(f"Score: {score}", True, (255, 255, 255))
            end_msg3 = font.render(f"Rank: {ai_par}", True, (100, 255, 100))
            restart_msg = font.render("Press SPACE to Restart", True, (200, 200, 200))
            
            screen.blit(end_msg1, (SCREEN_SIZE//2 - end_msg1.get_width()//2, SCREEN_SIZE//2 - 100))
            screen.blit(end_msg2, (SCREEN_SIZE//2 - end_msg2.get_width()//2, SCREEN_SIZE//2 - 20))
            screen.blit(end_msg3, (SCREEN_SIZE//2 - end_msg3.get_width()//2, SCREEN_SIZE//2 + 30))
            screen.blit(restart_msg, (SCREEN_SIZE//2 - restart_msg.get_width()//2, SCREEN_SIZE//2 + 100))

        pygame.display.flip()
        clock.tick(FPS) # 60 FPS 고정 (AI 학습 속도와 동기화)

    pygame.quit()

if __name__ == "__main__":
    main()