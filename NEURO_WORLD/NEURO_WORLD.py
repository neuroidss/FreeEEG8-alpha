#!/usr/bin/env python3
import torch, numpy as np, cv2, pygame, time, gc, math
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, LCMScheduler, AutoencoderTiny

try:
    from bci_logic import CrystalBCI
    BCI_AVAILABLE = True
except ImportError:
    BCI_AVAILABLE = False

W, H = 512, 384
UI_W = 420 

class MockBCI:
    """ Симулятор: когерентности как скорости. Вектор sin(t) возвращает в 0. """
    def __init__(self):
        self.is_connected = False
        self.persistence = 0.0
        self.ciplv_vec = np.zeros(28, dtype=np.float32)
        self.scales = np.cos(np.arange(28) * 0.5)

    def update_sim_internal(self, current_time, duration):
        """ Внутренний метод для плавного маятника """
        freq = (2 * np.pi) / duration
        wave = np.sin(current_time * freq)
        self.ciplv_vec = wave * self.scales
        self.persistence = abs(wave)

    def update_sim(self):
        """ Заглушка для совместимости с интерфейсом CrystalBCI """
        pass

class StabilityLabBCI:
    def __init__(self, mode="lcm"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.pipe = None
        self.current_pil = self._make_noise_image()
        self.running = True
        self.paused = False
        self.reset_active = False 
        
        # Управление фазой 
        self.sim_start_time = time.time()
        self.sim_duration = 10.0
        
        self.bci = CrystalBCI() if BCI_AVAILABLE else MockBCI()
        
        # --- СТРОГО ТВОИ ПАРАМЕТРЫ V38 ПО УМОЛЧАНИЮ ---
        self.p = {
            "unet_passes": 3,      
            "noise_inject": 0.97,  
            "guidance": 1.20,      
            "g_channel_nerf": 1.0, 
            "contrast_fix": 0.0,
            "bci_apply": 0.01        
        }
        
        self.accumulated_drift = None 
        self.drift_mag = 0.0
        self.actual_passes = 0
        self.switch_model(mode)

    def switch_model(self, new_mode):
        print(f"🔄 Loading {new_mode.upper()}...")
        cfg_id = "stabilityai/sd-turbo" if new_mode == "turbo" else "SimianLuo/LCM_Dreamshaper_v7"
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            cfg_id, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False
        ).to(self.device)
        if new_mode == "lcm": 
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=self.dtype).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        
        dim = self.pipe.text_encoder.config.hidden_size
        self.accumulated_drift = torch.zeros((1, dim), device=self.device, dtype=self.dtype)
        torch.manual_seed(42)
        directions = torch.empty(28, dim)
        torch.nn.init.orthogonal_(directions) 
        self.direction_basis = directions.to(device=self.device, dtype=self.dtype)

    def _make_noise_image(self):
        return Image.fromarray(np.random.randint(60, 140, (H, W, 3), dtype=np.uint8))

    def _get_decoupled_steps(self, target_passes, strength):
        strength = max(0.02, min(1.0, strength))
        if strength >= 0.99: return int(target_passes)
        calc_steps = math.ceil(target_passes / strength)
        return min(50, max(int(target_passes) + 1, calc_steps))

    def reset_timer(self):
        """ Сброс фазы (времени) симулятора """
        self.sim_start_time = time.time()

    def surgery(self, current_f32, old_f32):
        res = current_f32.copy()
        if self.p["g_channel_nerf"] > 0:
            mu = np.mean(res, axis=(0,1))
            target_g = (mu[0] + mu[2]) / 2.0
            if mu[1] > target_g: res[:,:,1] -= (mu[1] - target_g) * self.p["g_channel_nerf"]
        return np.clip(res, 0, 255).astype(np.float32)

    def step(self):
        # 1. Обновляем BCI (Реальный или Мок)
        if hasattr(self.bci, 'update_sim_internal'):
            # Если это наш MockBCI — передаем время
            self.bci.update_sim_internal(time.time() - self.sim_start_time, self.sim_duration)
        else:
            # Если это реальный девайс — вызываем стандартный метод
            self.bci.update_sim()
        
        focus = self.bci.persistence * self.p["bci_apply"]

        # 2. Логика RESET и накопления
        if self.reset_active:
            self.accumulated_drift.zero_()
            effective_focus = 0.0
        else:
            effective_focus = focus
            # Когерентности работают как скорость
            velocities = torch.tensor(self.bci.ciplv_vec, device=self.device, dtype=self.dtype).unsqueeze(0)
            # Коэффициент 0.2 толкает эмбеддинг
            self.accumulated_drift += torch.matmul(velocities, self.direction_basis) * focus * 0.2
        
        # 3. Вытеснение шума
        current_noise = self.p["noise_inject"] * (1.0 - effective_focus)
        current_noise = max(0.05, min(1.0, current_noise))
        
        calc_steps = self._get_decoupled_steps(self.p["unet_passes"], current_noise)
        old_f32 = np.array(self.current_pil).astype(np.float32)

        with torch.inference_mode(), torch.autocast("cuda"):
            tokens = self.pipe.tokenizer("cyberpunk city buildings, night, glowing windows", 
                                        return_tensors="pt", padding="max_length", truncation=True).input_ids.to(self.device)
            base_embeds = self.pipe.text_encoder(tokens)[0]
            # Двигаем центр семантики по накопленному вектору
            prompt_embeds = base_embeds + self.accumulated_drift.unsqueeze(1)

            res_pil = self.pipe(
                prompt_embeds=prompt_embeds,
                image=Image.fromarray(old_f32.astype(np.uint8)).convert("RGB"),
                strength=float(current_noise), 
                num_inference_steps=int(calc_steps),
                guidance_scale=float(self.p["guidance"])
            ).images[0]

        new_f32 = np.array(res_pil).astype(np.float32)
        self.current_pil = Image.fromarray(self.surgery(new_f32, old_f32).astype(np.uint8))
        self.drift_mag = torch.norm(self.accumulated_drift).item()
        self.actual_passes = int(calc_steps * current_noise)

def main():
    lab = StabilityLabBCI(mode="lcm")
    pygame.init()
    screen = pygame.display.set_mode((W + UI_W, H))
    font = pygame.font.SysFont("monospace", 14, bold=True)
    small_font = pygame.font.SysFont("monospace", 12)
    dragging = False
    btn_rect = pygame.Rect(20, 10, 160, 35)

    while lab.running:
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: lab.running = False
            
            # Клавиатура
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: lab.reset_active = True
                if event.key == pygame.K_SPACE: lab.paused = not lab.paused
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_r:
                    lab.reset_timer() # Сброс фазы
                    lab.reset_active = False
            
            # Мышь
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_rect.collidepoint(mx, my): lab.reset_active = True
                elif mx < UI_W: dragging = True
            if event.type == pygame.MOUSEBUTTONUP:
                if lab.reset_active: lab.reset_timer() # Сброс фазы
                lab.reset_active = False
                dragging = False

        if dragging:
            p_keys = list(lab.p.keys())
            p_idx = (my - 125) // 35 
            if 0 <= p_idx < len(p_keys):
                k = p_keys[p_idx]
                val = np.clip((mx - 20) / 300, 0, 1)
                if k == "unet_passes": lab.p[k] = int(val * 3) + 1 
                elif k == "guidance": lab.p[k] = 1.0 + val * 4.0
                else: lab.p[k] = val

        if not lab.paused: lab.step()
        
        # --- UI ---
        img_py = pygame.image.frombuffer(np.array(lab.current_pil).tobytes(), (W, H), 'RGB')
        screen.blit(img_py, (UI_W, 0))
        pygame.draw.rect(screen, (15, 15, 18), (0, 0, UI_W, H))
        
        # RESET BUTTON
        btn_col = (255, 50, 50) if lab.reset_active else (100, 20, 20)
        pygame.draw.rect(screen, btn_col, btn_rect, border_radius=5)
        pygame.draw.rect(screen, (255,255,255), btn_rect, 1, border_radius=5)
        screen.blit(font.render("HOLD RESET (R)", True, (255, 255, 255)), (btn_rect.x+12, btn_rect.y+9))
        
        dist_col = (0, 255, 150) if lab.drift_mag < 0.05 else (0, 200, 255)
        screen.blit(small_font.render(f"DRIFT: {lab.drift_mag:.5f}", True, dist_col), (190, 22))

        # Визуальный маятник
        t_sim = time.time() - lab.sim_start_time
        wave = np.sin(t_sim * (2 * np.pi / lab.sim_duration))
        pygame.draw.rect(screen, (30, 30, 40), (20, 55, UI_W-40, 10))
        pygame.draw.rect(screen, (0, 150, 255), (20, 55, int((UI_W-40)*(0.5 + 0.5 * wave)), 10))
        screen.blit(small_font.render(f"WAVE SPEED (RESET AT 0): {wave:.2f}", True, (150, 200, 255)), (20, 67))
            
        # Бар замещения
        f_disp = 0.0 if lab.reset_active else (lab.bci.persistence * lab.p["bci_apply"])
        pygame.draw.rect(screen, (40, 40, 40), (20, 90, UI_W-40, 10))
        pygame.draw.rect(screen, (255, 200, 0), (20, 90, int((UI_W-40)*(1-f_disp)), 10)) 
        pygame.draw.rect(screen, (0, 180, 255), (20 + int((UI_W-40)*(1-f_disp)), 90, int((UI_W-40)*f_disp), 10))
        screen.blit(small_font.render(f"NOISE vs BCI DYNAMICS: {f_disp*100:.0f}%", True, (200,200,200)), (20, 80))

        for i, (k, v) in enumerate(lab.p.items()):
            y_pos = 125 + i * 35
            col = (255, 100, 255) if k == "unet_passes" else (255, 200, 0) if k == "noise_inject" else (0, 200, 255)
            screen.blit(font.render(f"{k.upper()[:15]:15}: {v:6.2f}", True, col), (20, y_pos))
            pygame.draw.rect(screen, (40, 40, 40), (220, y_pos+6, 150, 4))
            norm = (v-1)/3 if k=="unet_passes" else (v-1)/4 if k=="guidance" else v
            pygame.draw.rect(screen, col, (220, y_pos+6, int(150 * np.clip(norm, 0, 1)), 4))

        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__": main()
