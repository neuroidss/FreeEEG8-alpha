# engine_core.py
import time, torch, numpy as np, pygame, cv2, math
from PIL import Image

# Твои стабильные модули (без изменений)
from render_logic import NeuroRender
from latent_input import UnifiedInputController

class NeuroCultivationEngine:
    def __init__(self, render_mode, vision_obj, seed, physics_cfg):
        pygame.init()
        pygame.joystick.init()
        
        self.render = NeuroRender(mode=render_mode)
        self.vision = vision_obj # Используем передаваемый объект vision_logic
        self.input = UnifiedInputController(seed=seed, use_bci=True)
        self.cfg = physics_cfg
        self.drift = 0.0

    def _surgery(self, img_pil, old_f32):
        res = np.array(img_pil).astype(np.float32)
        p = self.cfg.get("color_p", 0.0)
        if p > 0:
            green = res[:, :, 1] * p
            res[:, :, 1] -= green
            res[:, :, 0] += green * 0.5; res[:, :, 2] += green * 0.5
        mu_t, std_t = cv2.meanStdDev(res); mu_s, std_s = cv2.meanStdDev(old_f32)
        inertia = self.cfg.get("inertia", 0.0)
        t_std = std_t * (1 - inertia) + std_s * inertia
        res = (res - mu_t.reshape(1,1,3)) * (t_std / (std_t + 1e-5)).reshape(1,1,3) + mu_t.reshape(1,1,3)
        return Image.fromarray(np.clip(res, 0, 255).astype(np.uint8))

    def run(self, void_t, pipeline, win_size, req_ok, v_int, ok_val):
        screen = pygame.display.set_mode(win_size)
        first_emb = pipeline[0]['b_emb']
        seed_noise = Image.fromarray(np.random.randint(40, 70, (384, 512, 3), dtype=np.uint8))
        current_pil = self.render.generate(prompt_embeds=first_emb, image=seed_noise, strength=1.0)
        
        idx, successes, last_v_t = 0, 0, time.time()
        u_on, b_on, v_msg = True, True, 0 # v_msg теперь число/статус

        while True:
            t = time.time()
            self.input.update(t)
            if idx >= len(pipeline): break
            
            step = pipeline[idx]
            target_t, secret_b, prompt_s, is_act, label = step['t'], step['b'], step['p'], step['a'], step['l']
            base_emb, slot = step['b_emb'], step['slot']

            # --- ФИЗИКА ---
            u_v = torch.tensor(np.nan_to_num(self.input.ciplv_vec), device=self.render.device, dtype=self.render.dtype)
            b_v = secret_b.to(self.render.device, self.render.dtype) * 1.5 if (is_act and b_on) else torch.zeros(28, device=self.render.device, dtype=self.render.dtype)
            active_sig = (u_v if u_on else 0) + b_v
            if torch.norm(active_sig) > 2.0: active_sig *= (2.0 / torch.norm(active_sig))

            if is_act:
                sim = torch.dot(torch.nn.functional.normalize(active_sig, dim=0, eps=1e-6), 
                                torch.nn.functional.normalize(secret_b.to(self.render.device, self.render.dtype), dim=0, eps=1e-6)).item()
                target_drift = max(0.0, sim) * min(1.0, torch.norm(active_sig).item() / 1.5)
            else: target_drift = 0.0

            if target_drift > 0.05:
                self.drift += (target_drift - self.drift) * self.cfg["impulse"]
            else:
                self.drift *= self.cfg["decay"]
            self.drift = max(0.0, min(1.0, self.drift))

            # --- РЕНДЕР ---
            final_emb = base_emb.clone()
            if is_act:
                final_emb[0, slot, :] = torch.lerp(void_t.to(self.render.device, self.render.dtype), 
                                                   target_t.to(self.render.device, self.render.dtype), self.drift)
            
            old_f32 = np.array(current_pil).astype(np.float32)
            is_moving = (self.drift > 0.1 and is_act)
            res_pil = self.render.pipe(
                prompt_embeds=final_emb.to(self.render.dtype), image=current_pil, 
                strength=float(self.cfg["s_mov"] if is_moving else self.cfg["s_rest"]), 
                num_inference_steps=(4 if is_moving else 2), 
                guidance_scale=1.2 + (successes / req_ok) * 1.5
            ).images[0]
            current_pil = self._surgery(res_pil, old_f32)

            # --- СУДЬЯ ---
            if t - last_v_t > v_int:
                last_v_t = t
                ans = self.vision.ask(current_pil, prompt_s).lower()
                if ok_val in ans:
                    successes += 1
                    if successes >= req_ok: 
                        idx += 1; successes = 0; self.drift = 0.0
                else: successes = 0

            # --- UI ---
            canvas = np.full((win_size[1], win_size[0], 3), (15, 15, 20), np.uint8)
            img_cv = cv2.resize(cv2.cvtColor(np.array(current_pil), cv2.COLOR_RGB2BGR), (512, 384))
            canvas[168:168+384, 384:384+512] = img_cv
            
            def txt(t, p, s, c): cv2.putText(canvas, str(t), p, 0, s, (0,0,0), 3, cv2.LINE_AA); cv2.putText(canvas, str(t), p, 0, s, c, 1, cv2.LINE_AA)
            txt(f"GOAL: {label}", (450, 40), 0.8, (0, 255, 255))
            txt(f"SUCCESS: {successes}/{req_ok}", (450, 100), 0.6, (0, 255, 100) if successes > 0 else (100, 100, 255))
            
            cv2.rectangle(canvas, (20, 130), (360, 145), (40, 40, 45), -1); cv2.rectangle(canvas, (20, 130), (20+int(340*self.drift), 145), (0, 200, 255), -1)
            cv2.rectangle(canvas, (20, 180), (360, 195), (40, 40, 45), -1); cv2.rectangle(canvas, (20, 180), (20+int(340*(successes/req_ok)), 195), (0, 255, 100), -1)
            
            # Компас
            c_p, r = (1080, 540), 120
            cv2.circle(canvas, c_p, r, (20, 20, 25), -1); cv2.circle(canvas, c_p, r, (100, 100, 100), 1)
            cur_np = active_sig.detach().cpu().numpy()
            for i in range(28):
                ang = (i / 28) * 2 * math.pi
                if is_act:
                    tv = np.clip(secret_b[i].item() * 3.0, -1, 1)
                    cv2.line(canvas, c_p, (int(c_p[0] + math.cos(ang)*r*tv), int(c_p[1] + math.sin(ang)*r*tv)), (255, 100, 0), 1)
                cv = np.clip(cur_np[i] * 1.5, -1, 1)
                cv2.line(canvas, c_p, (int(c_p[0] + math.cos(ang)*r*cv), int(c_p[1] + math.sin(ang)*r*cv)), (0, 255, 150), 2)

            pygame.display.get_surface().blit(pygame.image.frombuffer(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).tobytes(), win_size, 'RGB'), (0, 0))
            pygame.display.flip()
            for e in pygame.event.get():
                if e.type == pygame.QUIT: return
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_u: u_on = not u_on
                    if e.key == pygame.K_b: b_on = not b_on
