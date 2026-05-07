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
    def __init__(self):
        self.is_connected = False
        self.persistence = 0.0
        self.ciplv_vec = np.zeros(28, dtype=np.float32)
        self.scales = np.cos(np.arange(28) * 0.7) * 2.0

    def update_sim_internal(self, current_time, duration):
        freq = (2 * np.pi) / duration
        wave = np.sin(current_time * freq)
        self.ciplv_vec = wave * self.scales
        self.persistence = abs(wave)

class StabilityLabBCI:
    def __init__(self, mode="lcm"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.pipe = None
        self.current_pil = self._make_noise_image()
        self.running = True
        self.paused = False
        self.reset_active = False 
        self.fullscreen = False
        self.show_ui = True
        
        self.sim_start_time = time.time()
        self.sim_duration = 10.0
        
        self.bci = CrystalBCI() if BCI_AVAILABLE else MockBCI()
        
        # --- ТВОИ ПАРАМЕТРЫ ДЛЯ НЕЙРОФИДБЕКА ---
        self.p = {
            "unet_passes": 1,      
            "noise_inject": 0.97,  
            "guidance": 1.20,      
            "g_channel_nerf": 1.0, 
            "bci_apply": 0.003,
            "elasticity": 1.0,   
            "focus_grip": 0.0    
        }
        
        self.accumulated_drift = torch.zeros((1, 768), device=self.device, dtype=self.dtype) 
        self.drift_mag = 0.0
        self.actual_passes = 0
        self._load_engine(mode)

    def _load_engine(self, mode):
        print(f"🔄 Loading {mode.upper()}...")
        cfg_id = "stabilityai/sd-turbo" if mode == "turbo" else "SimianLuo/LCM_Dreamshaper_v7"
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            cfg_id, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False
        ).to(self.device)
        if mode == "lcm": self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
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

    def surgery(self, current_f32, old_f32):
        res = current_f32.copy()
        if self.p["g_channel_nerf"] > 0:
            mu = np.mean(res, axis=(0,1))
            target_g = (mu[0] + mu[2]) / 2.0
            if mu[1] > target_g: res[:,:,1] -= (mu[1] - target_g) * self.p["g_channel_nerf"]
        return np.clip(res, 0, 255).astype(np.float32)

    def step(self):
        if hasattr(self.bci, 'update_sim_internal'):
            self.bci.update_sim_internal(time.time() - self.sim_start_time, self.sim_duration)
        else: self.bci.update_sim()
        
        focus = self.bci.persistence
        apply_force = focus * self.p["bci_apply"]

        if self.reset_active:
            self.accumulated_drift.zero_()
            self.sim_start_time = time.time()
            effective_focus = 0.0
        else:
            effective_focus = apply_force
            velocities = torch.tensor(self.bci.ciplv_vec, device=self.device, dtype=self.dtype).unsqueeze(0)
            # Дрейф
            self.accumulated_drift += torch.matmul(velocities, self.direction_basis) * apply_force * 2.0
            # Упругость (Homeostasis)
            pull_back = self.p["elasticity"] * (1.0 - (focus * self.p["focus_grip"]))
            if pull_back > 0: self.accumulated_drift *= (1.0 - pull_back * 0.1)

        current_noise = self.p["noise_inject"] * (1.0 - (effective_focus * 10.0)) # Усилил влияние фокуса на шум для наглядности
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
                strength=float(current_noise), num_inference_steps=int(calc_steps), guidance_scale=float(self.p["guidance"])
            ).images[0]

        new_f32 = np.array(res_pil).astype(np.float32)
        self.current_pil = Image.fromarray(self.surgery(new_f32, old_f32).astype(np.uint8))
        self.drift_mag = torch.norm(self.accumulated_drift).item()
        self.actual_passes = int(calc_steps * current_noise)

def main():
    lab = StabilityLabBCI(mode="lcm")
    pygame.init()
    
    # Получаем инфо о мониторе
    scr_info = pygame.display.Info()
    MONITOR_W, MONITOR_H = scr_info.current_w, scr_info.current_h
    
    screen = pygame.display.set_mode((W + UI_W, H), pygame.RESIZABLE)
    pygame.display.set_caption("Stability Lab BCI - [F] Fullscreen, [Tab] UI")
    
    font = pygame.font.SysFont("monospace", 14, bold=True)
    small_font = pygame.font.SysFont("monospace", 12)
    dragging = False
    btn_rect = pygame.Rect(20, 10, 160, 35)

    while lab.running:
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: lab.running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    lab.fullscreen = not lab.fullscreen
                    if lab.fullscreen:
                        screen = pygame.display.set_mode((MONITOR_W, MONITOR_H), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode((W + UI_W, H), pygame.RESIZABLE)
                if event.key == pygame.K_TAB:
                    lab.show_ui = not lab.show_ui
                if event.key == pygame.K_r: lab.reset_active = True
                if event.key == pygame.K_SPACE: lab.paused = not lab.paused

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_r: lab.reset_active = False

            if event.type == pygame.MOUSEBUTTONDOWN and lab.show_ui and not lab.fullscreen:
                if btn_rect.collidepoint(mx, my): lab.reset_active = True
                elif mx < UI_W: dragging = True
            if event.type == pygame.MOUSEBUTTONUP:
                lab.reset_active = False
                dragging = False

        if dragging and not lab.fullscreen:
            p_keys = list(lab.p.keys())
            p_idx = (my - 135) // 35 
            if 0 <= p_idx < len(p_keys):
                k = p_keys[p_idx]
                val = np.clip((mx - 20) / 300, 0, 1)
                if k == "unet_passes": lab.p[k] = int(val * 4) + 1 
                elif k == "bci_apply": lab.p[k] = val * 0.05
                else: lab.p[k] = val

        if not lab.paused: lab.step()
        
        # --- ОТРИСОВКА ---
        img_np = np.array(lab.current_pil)
        img_py = pygame.image.frombuffer(img_np.tobytes(), (W, H), 'RGB')
        
        if lab.fullscreen:
            screen.fill((0, 0, 0))
            # Масштабируем картинку под экран с сохранением пропорций
            scale = min(MONITOR_W / W, MONITOR_H / H)
            new_w, new_h = int(W * scale), int(H * scale)
            scaled_img = pygame.transform.smoothscale(img_py, (new_w, new_h))
            screen.blit(scaled_img, ((MONITOR_W - new_w) // 2, (MONITOR_H - new_h) // 2))
            
            # В фулскрине рисуем только если UI включен (Tab)
            if lab.show_ui:
                txt = small_font.render(f"DRIFT: {lab.drift_mag:.4f} | BCI: {'CONNECTED' if lab.bci.is_connected else 'SIM'}", True, (0, 255, 0))
                screen.blit(txt, (20, MONITOR_H - 30))
        else:
            # Обычный режим
            screen.fill((15, 15, 18))
            screen.blit(img_py, (UI_W, 0))
            
            if lab.show_ui:
                # Кнопка и статус
                btn_col = (255, 50, 50) if lab.reset_active else (100, 20, 20)
                pygame.draw.rect(screen, btn_col, btn_rect, border_radius=5)
                screen.blit(font.render("HOLD RESET (R)", True, (255, 255, 255)), (btn_rect.x+12, btn_rect.y+9))
                screen.blit(small_font.render(f"DRIFT: {lab.drift_mag:.5f}", True, (0, 255, 150)), (190, 22))

                # Замещение
                f_disp = lab.bci.persistence * lab.p["bci_apply"] * 10.0 # Визуальный масштаб
                pygame.draw.rect(screen, (40, 40, 40), (20, 100, UI_W-40, 12))
                pygame.draw.rect(screen, (255, 200, 0), (20, 100, int((UI_W-40)*max(0, 1-f_disp)), 12)) 
                pygame.draw.rect(screen, (0, 180, 255), (20 + int((UI_W-40)*max(0, 1-f_disp)), 100, int((UI_W-40)*min(1, f_disp)), 12))

                for i, (k, v) in enumerate(lab.p.items()):
                    y_pos = 135 + i * 35
                    col = (255, 100, 255) if k == "unet_passes" else (255, 200, 0) if k == "noise_inject" else (0, 200, 255)
                    screen.blit(font.render(f"{k.upper()[:15]:15}: {v:6.4f}", True, col), (20, y_pos))
                    pygame.draw.rect(screen, (40, 40, 40), (220, y_pos+6, 150, 4))
                    norm = (v-1)/4 if k=="unet_passes" else v/0.05 if k=="bci_apply" else v
                    pygame.draw.rect(screen, col, (220, y_pos+6, int(150 * np.clip(norm, 0, 1)), 4))

        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__": main()
