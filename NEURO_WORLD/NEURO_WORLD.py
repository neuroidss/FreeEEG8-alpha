#!/usr/bin/env python3
import torch, numpy as np, cv2, pygame, time, gc, math
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, LCMScheduler, AutoencoderTiny

# Пытаемся импортировать реальную логику
try:
    from bci_logic import CrystalBCI
    BCI_AVAILABLE = True
except ImportError:
    BCI_AVAILABLE = False

W, H = 512, 384
UI_W = 420 

class MockBCI:
    """ Заглушка, если файл bci_logic.py вообще отсутствует """
    def __init__(self):
        self.is_connected = False
        self.persistence = 0.0
        self.ciplv_vec = np.zeros(28, dtype=np.float32)
    def update_sim(self): pass

class StabilityLabBCI:
    def __init__(self, mode="lcm"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.pipe = None
        self.current_pil = self._make_noise_image()
        self.running = True
        self.paused = False
        self.reset_active = False 
        
        # Управление временем для симулятора
        self.sim_start_time = time.time()
        self.sim_duration = 10.0
        self.sim_scales = np.cos(np.arange(28) * 0.7) * 2.0 # Направления симуляции
        
        # Инициализируем BCI (он сам будет пытаться подключиться в фоне)
        self.bci = CrystalBCI() if BCI_AVAILABLE else MockBCI()
        
        # --- ТВОИ РАБОЧИЕ ПАРАМЕТРЫ ---
        self.p = {
            "unet_passes": 1,      
            "noise_inject": 0.97,  
            "guidance": 1.20,      
            "g_channel_nerf": 1.0, 
            "bci_apply": 0.003,
            "elasticity": 1.0,   
            "focus_grip": 0.0    
        }
        
        self.accumulated_drift = None 
        self.drift_mag = 0.0
        self.actual_passes = 0
        self._load_engine(mode)

    def _load_engine(self, mode):
        print(f"🔄 Loading {mode.upper()}...")
        cfg_id = "stabilityai/sd-turbo" if mode == "turbo" else "SimianLuo/LCM_Dreamshaper_v7"
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            cfg_id, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False
        ).to(self.device)
        if mode == "lcm": 
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
        return min(50, max(int(target_passes) + 1, math.ceil(target_passes / strength)))

    def reset_logic(self):
        self.accumulated_drift.zero_()
        self.sim_start_time = time.time()

    def surgery(self, current_f32, old_f32):
        res = current_f32.copy()
        if self.p["g_channel_nerf"] > 0:
            mu = np.mean(res, axis=(0,1))
            target_g = (mu[0] + mu[2]) / 2.0
            if mu[1] > target_g: res[:,:,1] -= (mu[1] - target_g) * self.p["g_channel_nerf"]
        return np.clip(res, 0, 255).astype(np.float32)

    def step(self):
        # --- ЛОГИКА ПОДКЛЮЧЕНИЯ ---
        if self.bci.is_connected:
            self.bci.update_sim() # Обновляем реальный поток (названо update_sim для совместимости)
        else:
            # Если не подключен — крутим нашу математическую "Волну"
            t_sim = time.time() - self.sim_start_time
            freq = (2 * np.pi) / self.sim_duration
            wave = np.sin(t_sim * freq)
            self.bci.ciplv_vec = wave * self.sim_scales
            self.bci.persistence = abs(wave)
        
        focus = self.bci.persistence
        apply_force = focus * self.p["bci_apply"]

        if self.reset_active:
            self.reset_logic()
            effective_focus = 0.0
        else:
            effective_focus = apply_force
            velocities = torch.tensor(self.bci.ciplv_vec, device=self.device, dtype=self.dtype).unsqueeze(0)
            # Накопление пути (Investigation)
            self.accumulated_drift += torch.matmul(velocities, self.direction_basis) * apply_force * 5.0
            
            # Упругость (Homeostasis) - по умолчанию выключена (elasticity=0)
            pull_back = self.p["elasticity"] * (1.0 - (focus * self.p["focus_grip"]))
            if pull_back > 0:
                self.accumulated_drift *= (1.0 - pull_back * 0.1)

        # Вытеснение шума
        current_noise = self.p["noise_inject"] * (1.0 - effective_focus)
        current_noise = max(0.05, min(1.0, current_noise))
        
        calc_steps = self._get_decoupled_steps(self.p["unet_passes"], current_noise)
        old_f32 = np.array(self.current_pil).astype(np.float32)

        with torch.inference_mode(), torch.autocast("cuda"):
            tokens = self.pipe.tokenizer("cyberpunk city buildings, night, glowing windows", 
                                        return_tensors="pt", padding="max_length", truncation=True).input_ids.to(self.device)
            base_embeds = self.pipe.text_encoder(tokens)[0]
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
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: lab.reset_active = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_r: lab.reset_active = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_rect.collidepoint(mx, my): lab.reset_active = True
                elif mx < UI_W: dragging = True
            if event.type == pygame.MOUSEBUTTONUP:
                lab.reset_active = False
                dragging = False

        if dragging:
            p_keys = list(lab.p.keys())
            p_idx = (my - 130) // 35 
            if 0 <= p_idx < len(p_keys):
                k = p_keys[p_idx]
                val = np.clip((mx - 20) / 300, 0, 1)
                if k == "unet_passes": lab.p[k] = int(val * 4) + 1 
                elif k == "bci_apply": lab.p[k] = val * 0.1 # Лимит для точности
                else: lab.p[k] = val

        lab.step()
        
        # --- UI ---
        img_py = pygame.image.frombuffer(np.array(lab.current_pil).tobytes(), (W, H), 'RGB')
        screen.blit(img_py, (UI_W, 0))
        pygame.draw.rect(screen, (15, 15, 18), (0, 0, UI_W, H))
        
        # Статус BCI
        bci_status = "BCI: CONNECTED" if lab.bci.is_connected else "BCI: SEARCHING/SIM..."
        bci_col = (0, 255, 120) if lab.bci.is_connected else (255, 150, 0)
        screen.blit(small_font.render(bci_status, True, bci_col), (20, 50))
        
        # Кнопка
        btn_col = (255, 50, 50) if lab.reset_active else (100, 20, 20)
        pygame.draw.rect(screen, btn_col, btn_rect, border_radius=5)
        screen.blit(font.render("HOLD RESET (R)", True, (255, 255, 255)), (btn_rect.x+12, btn_rect.y+9))
        
        dist_col = (0, 255, 150) if lab.drift_mag < 0.05 else (0, 200, 255)
        screen.blit(small_font.render(f"DRIFT: {lab.drift_mag:.5f}", True, dist_col), (190, 22))

        # Визуализация фазы (для симулятора)
        if not lab.bci.is_connected:
            t_sim = time.time() - lab.sim_start_time
            wave = np.sin(t_sim * (2 * np.pi / lab.sim_duration))
            pygame.draw.rect(screen, (30, 30, 40), (20, 70, UI_W-40, 8))
            pygame.draw.rect(screen, (0, 150, 255), (20, 70, int((UI_W-40)*(0.5 + 0.5 * wave)), 8))
            
        # Замещение
        f_disp = lab.bci.persistence * lab.p["bci_apply"]
        pygame.draw.rect(screen, (40, 40, 40), (20, 100, UI_W-40, 12))
        pygame.draw.rect(screen, (255, 200, 0), (20, 100, int((UI_W-40)*(1-f_disp)), 12)) 
        pygame.draw.rect(screen, (0, 180, 255), (20 + int((UI_W-40)*(1-f_disp)), 100, int((UI_W-40)*f_disp), 12))
        screen.blit(small_font.render(f"NOISE vs BCI DYNAMICS: {f_disp*100:.2f}%", True, (200,200,200)), (20, 88))

        for i, (k, v) in enumerate(lab.p.items()):
            y_pos = 135 + i * 35
            col = (255, 100, 255) if k == "unet_passes" else (255, 200, 0) if k == "noise_inject" else (0, 200, 255)
            if k in ["elasticity", "focus_grip"]: col = (100, 255, 100)
            
            screen.blit(font.render(f"{k.upper()[:15]:15}: {v:6.4f}", True, col), (20, y_pos))
            pygame.draw.rect(screen, (40, 40, 40), (220, y_pos+6, 150, 4))
            norm = (v-1)/4 if k=="unet_passes" else v/0.1 if k=="bci_apply" else v
            pygame.draw.rect(screen, col, (220, y_pos+6, int(150 * np.clip(norm, 0, 1)), 4))

        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__": main()
