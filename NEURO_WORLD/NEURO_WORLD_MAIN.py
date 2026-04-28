#!/usr/bin/env python3
import time, threading, cv2, torch, numpy as np, pygame, os, argparse
from PIL import Image
from bci_logic import CrystalBCI
from vision_logic import QwenVision
from render_logic import NeuroRender

# Базовое разрешение (4:3)
GW, GH = 512, 384

def parse_args():
    parser = argparse.ArgumentParser(description="NEURO-WORLD: Final Perfect Edition")
    parser.add_argument("--render", type=str, choices=["turbo", "lcm"], default="turbo")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--bci-sens", type=float, default=1.0)
    parser.add_argument("--pad-sens", type=float, default=1.0)
    parser.add_argument("--bci-mode", type=str, choices=["sticks", "hypersphere"], default="hypersphere")
    parser.add_argument("--init-prompt", type=str, default="psytrance visuals")
    parser.add_argument("--style-suffix", type=str, default=", highly detailed, 8k, vivid")
    parser.add_argument("--bg-color", type=str, choices=["black", "white"], default="black")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--llm", type=str, default="qwen3.5:0.8b")
    parser.add_argument("--llm-prompt", type=str, default="Describe the image in 2 words.")
    return parser.parse_args()

def create_mapping_matrix(latent_dim):
    torch.manual_seed(42)
    matrix = torch.empty(28, latent_dim, dtype=torch.float32)
    torch.nn.init.orthogonal_(matrix)
    return matrix * 2.0

def main():
    args = parse_args()
    bci = CrystalBCI()
    vision = QwenVision(model=args.llm, prompt=args.llm_prompt)
    vision.last_answer = args.init_prompt
    render = NeuroRender(mode=args.render, compile_unet=args.compile)
    
    # Инициализация Pygame и монитора
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    pygame.init(); pygame.joystick.init()
    scr_info = pygame.display.Info()
    MONITOR_W, MONITOR_H = scr_info.current_w, scr_info.current_h
    
    joy = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
    if joy: joy.init()

    # Матрица маппинга BCI -> Латенты
    M_DIM = render.latent_dim
    MAPPING_MATRIX = create_mapping_matrix(M_DIM).to(device=render.device, dtype=render.dtype)
    smoothed_shift = torch.zeros((1, M_DIM), device=render.device, dtype=render.dtype)

    current_pil = Image.fromarray(np.random.randint(50, 150, (GH, GW, 3), dtype=np.uint8))
    WINDOW_NAME = "NEURO-WORLD"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    B_COLOR = (0, 0, 0) if args.bg_color == "black" else (255, 255, 255)

    def vision_thread():
        while True:
            vision.ask(current_pil.copy())
            time.sleep(2.0)
    threading.Thread(target=vision_thread, daemon=True).start()

    while True:
        # --- 1. ОПРЕДЕЛЕНИЕ РАЗМЕРОВ ЭКРАНА ---
        is_fs = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
        if is_fs:
            win_w, win_h = MONITOR_W, MONITOR_H
        else:
            rect = cv2.getWindowImageRect(WINDOW_NAME)
            win_w, win_h = (rect[2], rect[3]) if (rect and rect[2] > 0) else (1280, 720)

        # --- 2. ОБРАБОТКА BCI И ГЕЙМПАДА ---
        bci.update_sim()
        pygame.event.pump()
        def dz(v): return v if abs(v) > 0.15 else 0.0
        axes = {"lx":0.0, "ly":0.0, "rx":0.0, "ry":0.0}
        if joy:
            axes = {"lx": dz(joy.get_axis(0)), "ly": dz(joy.get_axis(1)), 
                    "rx": dz(joy.get_axis(3)) if joy.get_numaxes()>3 else 0,
                    "ry": dz(joy.get_axis(4)) if joy.get_numaxes()>4 else 0}

        strength = 0.35
        shift_magnitude = 0.0
        prompt_embeds_tensor = None

        # Логика Гиперсферы (динамика векторов)
        if args.bci_mode == "hypersphere":
            bci_tensor = torch.tensor(bci.ciplv_vec * args.bci_sens, dtype=render.dtype, device=render.device).unsqueeze(0)
            target_shift = torch.matmul(bci_tensor, MAPPING_MATRIX)
            decay = 0.92 if bci.is_connected else 0.75
            smoothed_shift = smoothed_shift * decay + target_shift * (1.0 - decay)
            shift_magnitude = float(torch.norm(smoothed_shift).cpu())
            if shift_magnitude > 0.1:
                strength = min(0.85, 0.35 + (shift_magnitude * 0.1))
        
        # Подготовка эмбеддингов
        full_prompt = vision.last_answer + args.style_suffix
        base_embed = render.encode_prompt(full_prompt)
        prompt_embeds_tensor = torch.nan_to_num(base_embed + (smoothed_shift if args.bci_mode == "hypersphere" else 0))

        # --- 3. ФИЗИКА КАМЕРЫ (WARP) ---
        final_vx = axes["lx"] * args.pad_sens
        final_vy = axes["ly"] * args.pad_sens
        final_tq = axes["rx"] * args.pad_sens
        
        if args.bci_mode == "sticks":
            final_vx += bci.vx * args.bci_sens
            final_vy += bci.vy * args.bci_sens
            final_tq += bci.tq * args.bci_sens
            if abs(final_vx) > 0.02 or abs(final_vy) > 0.02: strength = 0.55

        boost = 1.0 + bci.persistence * np.clip(args.bci_sens, 0, 1)
        M = cv2.getRotationMatrix2D((GW//2, GH//2), final_tq * 15.0 * boost, 1.0 - (final_vy * 0.08 * boost))
        M[0, 2] += (-final_vx * 25 * boost)
        M[1, 2] += axes["ry"] * 30 * args.pad_sens
        
        warped_np = cv2.warpAffine(np.array(current_pil), M, (GW, GH), borderMode=cv2.BORDER_REFLECT_101)

        # --- 4. РЕНДЕР И ЦВЕТОКОРРЕКЦИЯ ---
        with torch.no_grad():
            res = render.generate(prompt_embeds=prompt_embeds_tensor, image=Image.fromarray(warped_np), strength=strength)
        
        current_pil = render.match_palette(current_pil, res)
        img_bgr = cv2.cvtColor(np.array(current_pil), cv2.COLOR_RGB2BGR)

        # --- 5. ФИНАЛЬНАЯ СБОРКА (БЕЗ БЕЛЫХ ПОЛОС) ---
        canvas = np.full((win_h, win_w, 3), B_COLOR, dtype=np.uint8)
        scale = min(win_w / GW, win_h / GH)
        nw, nh = int(GW * scale), int(GH * scale)
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dx, dy = (win_w - nw) // 2, (win_h - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized

        # --- 6. UI ---
        if args.debug:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (win_w, 160), (0,0,0), -1)
            canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)
            
            y_pos = 35
            cv2.putText(canvas, f"THOUGHT: {vision.last_answer.upper()}", (20, y_pos), 0, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            y_pos += 35
            cv2.putText(canvas, f"BCI SHIFT: {shift_magnitude:.4f} | STRENGTH: {strength:.2f}", (20, y_pos), 0, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            y_pos += 35
            cv2.putText(canvas, f"CANVAS: {win_w}x{win_h} | IMAGE: {nw}x{nh}", (20, y_pos), 0, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Статус-точка
        st_color = (0, 255, 0) if bci.is_connected else (0, 0, 255)
        cv2.circle(canvas, (dx + nw - 20, dy + 20), 8, st_color, -1)

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('f'):
            is_fs_now = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, 
                                  cv2.WINDOW_NORMAL if is_fs_now else cv2.WINDOW_FULLSCREEN)

    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
