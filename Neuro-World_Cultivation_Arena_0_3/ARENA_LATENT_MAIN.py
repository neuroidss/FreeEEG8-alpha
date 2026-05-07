#!/usr/bin/env python3
import time, cv2, torch, numpy as np, pygame
from PIL import Image

from latent_input import UnifiedInputController
from vision_logic import QwenVision
from render_logic import NeuroRender
from arena_latent_logic import ArenaLatentJudge

GW, GH = 512, 384

def main():
    DEBUG_MODE = False # ПОСТАВЬ True ДЛЯ ТЕСТА ЭЭГ БЕЗ ДЕВАЙСА
    print("🔥 UNIFIED NEURAL INTERFACE ARENA (FIXED) 🔥")
    
    pygame.init()
    render = NeuroRender(mode="lcm", compile_unet=False)
    vision = QwenVision(model="qwen3.5:2b")
    
    # --- НАСТРОЙКА ВВОДА ---
    BCI_AVAILABLE = False
    try:
        from bci_logic import FreeEEG
        if FreeEEG.is_available(): BCI_AVAILABLE = True
    except ImportError: pass

    # Игрок 1: Человек
    p1_input = UnifiedInputController(seed=123, use_bci=BCI_AVAILABLE, is_bot=False, debug=DEBUG_MODE)
    
    # Игрок 2: Бот (добавлен флаг is_bot=True)
    p2_input = UnifiedInputController(seed=999, use_bci=True, is_bot=True)

    user_profile = { "name": "Cyber Paladin", "inventory":["Giant Blue Lightning Hammer", "Swarm of Golden Butterflies"]}
    bot_profile = { "name": "Volcano Demonlord", "inventory":["Massive Red Magma Tornado", "Thick Obsidian Wall Barrier"]}
    
    arena = ArenaLatentJudge(render, vision, user_profile, bot_profile)
    
    # Генерируем честные 10D цели (чтобы и геймпад, и клава, и бот могли дотянуться)
    np.random.seed(1024)
    p1_10d_targets = np.random.uniform(-1, 1, (len(user_profile['inventory']), 10)).astype(np.float32)
    p1_28d_targets = np.dot(p1_10d_targets, p1_input.projection_matrix)
    p1_28d_targets /= np.linalg.norm(p1_28d_targets, axis=1, keepdims=True)
    
    np.random.seed(2048)
    p2_10d_targets = np.random.uniform(-1, 1, (len(bot_profile['inventory']), 10)).astype(np.float32)
    p2_28d_targets = np.dot(p2_10d_targets, p2_input.projection_matrix)
    p2_28d_targets /= np.linalg.norm(p2_28d_targets, axis=1, keepdims=True)

    arena.set_fair_secrets(torch.tensor(p1_28d_targets, dtype=render.dtype, device=render.device),
                           torch.tensor(p2_28d_targets, dtype=render.dtype, device=render.device))
    
    WIN_W, WIN_H = 1280, 480 
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.DOUBLEBUF | pygame.HWSURFACE)
    
    pil_p1 = Image.fromarray(np.random.randint(50, 100, (GH, GW, 3), dtype=np.uint8))
    pil_p2 = Image.fromarray(np.random.randint(50, 100, (GH, GW, 3), dtype=np.uint8))
    
    last_judge_time = time.time()
    t_start = time.time()
    frame_count = 0
    running = True

    # Цель для 10 осей бота
    bot_target_10d = np.zeros(10)

    while running:
        current_time = time.time() - t_start
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False
            p1_input.handle_event(event)

        # 1. ОБНОВЛЕНИЕ ИГРОКА
        p1_input.update(current_time)
        
        # 2. ЛОГИКА БОТА
        if time.time() % 4.0 < 0.05:
            if arena.p2_state == "IDLE": 
                t_idx = np.random.randint(0, len(arena.p2['inventory']))
            else: 
                t_idx = arena.bot_brain_think(arena.p2_incoming)
            
            # Бот выбирает 10-мерную цель (как будто зажимает кнопки)
            bot_target_10d = p2_10d_targets[t_idx] * 1.2
        
        # Бот плавно двигает свои виртуальные стики
        # Мы напрямую меняем axes_10d перед вызовом update()
        p2_input.axes_10d = p2_input.axes_10d * 0.9 + bot_target_10d * 0.1
        # Вызываем update, чтобы проецировать эти оси в 28D
        p2_input.update(current_time)

        # 3. СУДЬЯ
        if time.time() - last_judge_time > 5.0 and arena.p1_hp > 0 and arena.p2_hp > 0:
            arena.request_judgment(pil_p1.copy(), pil_p2.copy())
            last_judge_time = time.time()

        # 4. РЕНДЕР
        if arena.p1_hp > 0 and arena.p2_hp > 0:
            with torch.no_grad():
                if frame_count % 2 == 0:
                    emb_p1 = arena.get_p1_embedding(p1_input.ciplv_vec, p1_input.persistence)
                    pil_p1 = render.generate(prompt_embeds=emb_p1, image=pil_p1, strength=0.55)
                else:
                    emb_p2 = arena.get_p2_embedding(p2_input.ciplv_vec, p2_input.persistence)
                    pil_p2 = render.generate(prompt_embeds=emb_p2, image=pil_p2, strength=0.55)
        frame_count += 1

        # 5. UI
        canvas = np.full((WIN_H, WIN_W, 3), (15, 15, 18), np.uint8)
        half_w = WIN_W // 2
        scale = min(half_w / GW, WIN_H / GH)
        nw, nh = int(GW * scale), int(GH * scale)
        dx, dy = (half_w - nw) // 2, (WIN_H - nh) // 2
        
        canvas[dy:dy+nh, dx:dx+nw] = cv2.resize(cv2.cvtColor(np.array(pil_p1), cv2.COLOR_RGB2BGR), (nw, nh))
        canvas[dy:dy+nh, half_w+dx:half_w+dx+nw] = cv2.resize(cv2.cvtColor(np.array(pil_p2), cv2.COLOR_RGB2BGR), (nw, nh))
        
        # Рисуем UI и передаем 10 осей обоих игроков для дебага
        arena.draw_ui(canvas, WIN_W, WIN_H, p1_input.axes_10d, p2_input.axes_10d)
        
        surf = pygame.image.frombuffer(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).tobytes(), (WIN_W, WIN_H), 'RGB')
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
