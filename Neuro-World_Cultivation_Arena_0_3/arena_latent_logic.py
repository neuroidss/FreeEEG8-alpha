import torch
import cv2
import threading
import requests
import numpy as np

class ArenaLatentJudge:
    def __init__(self, render, vision, p1_data, p2_data):
        self.render = render
        self.vision = vision
        self.p1 = p1_data
        self.p2 = p2_data
        
        self.p1_state = "IDLE"
        self.p2_state = "IDLE"
        self.p1_incoming = None
        self.p2_incoming = None
        self.p1_scars = "tattered clothes"
        self.p2_scars = "tattered clothes"
        
        self.p1_idle_prompt = f"{self.p1['name']} peacefully meditating, empty hands, detailed, solo"
        self.p2_idle_prompt = f"{self.p2['name']} peacefully meditating, empty hands, detailed, solo"
        
        self.p1_base_emb = self.render.encode_prompt(self.p1_idle_prompt)
        self.p2_base_emb = self.render.encode_prompt(self.p2_idle_prompt)
        _, self.seq_len, self.hidden_dim = self.p1_base_emb.shape
        
        # Секретные комбинации теперь загружаются снаружи (Честные 10D -> 28D)
        self.p1_secrets = None
        self.p2_secrets = None
        
        self.p1_accumulated_drift = 0.0
        self.p2_accumulated_drift = 0.0
        
        self.p1_current_sims = np.zeros(len(self.p1['inventory']))
        self.p2_current_sims = np.zeros(len(self.p2['inventory']))
        
        self.update_latent_topography(1)
        self.update_latent_topography(2)

        self.p1_hp = 100.0
        self.p2_hp = 100.0
        self.combat_log = ["LATENT SPACE INITIALIZED. SEEK YOUR ARTIFACTS."]
        self.is_judging = False
        self.winner = None

    def set_fair_secrets(self, p1_28d_tensor, p2_28d_tensor):
        """Принимает честные векторы, до которых игрок реально может дотянуться"""
        self.p1_secrets = p1_28d_tensor
        self.p2_secrets = p2_28d_tensor

    def _log(self, msg):
        self.combat_log.append(msg)
        if len(self.combat_log) > 6: self.combat_log.pop(0)

    def update_latent_topography(self, player_num):
        name = self.p1['name'] if player_num == 1 else self.p2['name']
        state = self.p1_state if player_num == 1 else self.p2_state
        incoming = self.p1_incoming if player_num == 1 else self.p2_incoming
        scars = self.p1_scars if player_num == 1 else self.p2_scars
        inventory = self.p1['inventory'] if player_num == 1 else self.p2['inventory']
        
        if state == "IDLE":
            base_prompt = f"{name} peacefully sitting in lotus position, meditating, empty hands, calm, {scars}, highly detailed"
            base_emb = self.render.encode_prompt(base_prompt)
            targets =[]
            for art in inventory:
                art_emb = self.render.encode_prompt(f"{name} standing up, violently attacking with {art}, glowing magic, {scars}")
                targets.append((art_emb - base_emb).squeeze(0))
        else:
            base_prompt = f"{name} violently blasted and knocked down by {incoming}, massive explosion, injured, falling, {scars}"
            base_emb = self.render.encode_prompt(base_prompt)
            targets =[]
            for art in inventory:
                art_emb = self.render.encode_prompt(f"{name} completely protected, blocking incoming {incoming} using a massive {art}, perfect defense, {scars}")
                targets.append((art_emb - base_emb).squeeze(0))
                
        targets_tensor = torch.stack(targets)
        if player_num == 1:
            self.p1_current_base = base_emb
            self.p1_current_targets = targets_tensor
        else:
            self.p2_current_base = base_emb
            self.p2_current_targets = targets_tensor

    def bot_brain_think(self, incoming_attack):
        inv_str = ", ".join(self.p2['inventory'])
        sys_prompt = "You are a martial arts master defending an attack. Choose ONE best defense from your list. Reply with ONLY the exact name."
        user_prompt = f"Incoming attack: '{incoming_attack}'. Your techniques: {inv_str}. Best counter?"
        try:
            payload = {"model": self.vision.model, "messages":[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}], "stream": False, "think": False}
            r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=3)
            if r.status_code == 200:
                ans = r.json().get('message', {}).get('content', '').strip().lower()
                for i, tech in enumerate(self.p2['inventory']):
                    if tech.lower() in ans: return i
        except: pass
        import random
        return random.randint(0, len(self.p2['inventory'])-1)

    def _calculate_embedding(self, vec_28d, secrets, targets, base, current_drift, focus):
        # Жесткая чистка входных данных
        v = np.nan_to_num(vec_28d, nan=0.0, posinf=1.5, neginf=-1.5)
        vec_tensor = torch.tensor(v, dtype=self.render.dtype, device=self.render.device)
        
        if focus < 0.05:
            similarity = torch.zeros(secrets.shape[0], dtype=self.render.dtype, device=self.render.device)
        else:
            norm_v = torch.nn.functional.normalize(vec_tensor, dim=0, eps=1e-6)
            similarity = torch.mv(secrets, norm_v) * focus
        
        # Чистка сходства перед UI
        similarity = torch.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=-1.0)
        
        weights = torch.clamp((similarity - 0.4) * 2.5, 0.0, 1.0)
        velocity = torch.tensordot(weights, targets, dims=([0], [0]))
        
        if isinstance(current_drift, float): new_drift = velocity * 0.2
        else: new_drift = current_drift * 0.85 + velocity * 0.15
            
        return base + new_drift.unsqueeze(0), new_drift, similarity

    def get_p1_embedding(self, vec_28d_numpy, focus):
        emb, drift, sims = self._calculate_embedding(vec_28d_numpy, self.p1_secrets, self.p1_current_targets, self.p1_current_base, self.p1_accumulated_drift, focus)
        self.p1_accumulated_drift = drift
        self.p1_current_sims = sims.detach().cpu().numpy()
        return emb

    def get_p2_embedding(self, vec_28d_numpy, focus):
        emb, drift, sims = self._calculate_embedding(vec_28d_numpy, self.p2_secrets, self.p2_current_targets, self.p2_current_base, self.p2_accumulated_drift, focus)
        self.p2_accumulated_drift = drift
        self.p2_current_sims = sims.detach().cpu().numpy()
        return emb

    def request_judgment(self, p1_img, p2_img):
        if self.is_judging or self.winner: return
        self.is_judging = True
        threading.Thread(target=self._dual_judge_thread, args=(p1_img, p2_img), daemon=True).start()

    def _ask_qwen_strict(self, img, choices):
        choices_str = "\n".join([f"- {c}" for c in choices])
        self.vision.sys_prompt = f"You are a strict referee. Reply with EXACTLY ONE option from this list that best describes the image:\nOPTIONS:\n{choices_str}"
        ans = self.vision.ask(img)
        if not ans: return "ERROR"
        ans_lower = ans.lower()
        for choice in choices:
            if choice.lower() in ans_lower: return choice
        return "IDLE"

    def _dual_judge_thread(self, p1_img, p2_img):
        if self.p1_state == "IDLE": choices_p1 =["IDLE (meditating)"] +[f"ATTACKING with {t}" for t in self.p1['inventory']]
        else: choices_p1 =["TAKING DAMAGE (knocked down)"] +[f"BLOCKING with {t}" for t in self.p1['inventory']]
        ans_p1 = self._ask_qwen_strict(p1_img, choices_p1)
        
        if self.p2_state == "IDLE": choices_p2 =["IDLE (meditating)"] +[f"ATTACKING with {t}" for t in self.p2['inventory']]
        else: choices_p2 = ["TAKING DAMAGE (knocked down)"] +[f"BLOCKING with {t}" for t in self.p2['inventory']]
        ans_p2 = self._ask_qwen_strict(p2_img, choices_p2)
        
        print(f"\n[QWEN] P1: '{ans_p1}' | P2: '{ans_p2}'")
        
        if "ATTACKING" in ans_p1:
            self._log(f"[P1] CASTS {ans_p1.split('with ')[-1].upper()}!")
            if "BLOCKING" not in ans_p2:
                self.p2_hp -= 35.0
                self.p2_scars += ", bloody"
                self._log(f"[P2] CRITICAL HIT FROM P1!")
            else: self._log(f"[P2] PERFECT BLOCK!")
            
        if "ATTACKING" in ans_p2:
            self._log(f"[P2] CASTS {ans_p2.split('with ')[-1].upper()}!")
            if "BLOCKING" not in ans_p1:
                self.p1_hp -= 35.0
                self.p1_scars += ", charred"
                self._log(f"[P1] CRITICAL HIT FROM P2!")
            else: self._log(f"[P1] PERFECT BLOCK!")

        old_p1_state, old_p2_state = self.p1_state, self.p2_state
        
        if "ATTACKING" in ans_p1:
            self.p2_state = "UNDER_ATTACK"
            self.p2_incoming = ans_p1.split('with ')[-1]
        else: self.p2_state = "IDLE"
            
        if "ATTACKING" in ans_p2:
            self.p1_state = "UNDER_ATTACK"
            self.p1_incoming = ans_p2.split('with ')[-1]
        else: self.p1_state = "IDLE"

        if old_p1_state != self.p1_state: self.update_latent_topography(1)
        if old_p2_state != self.p2_state: self.update_latent_topography(2)

        if self.p1_hp <= 0: self.winner = self.p2['name']
        if self.p2_hp <= 0: self.winner = self.p1['name']
        self.is_judging = False

    def draw_text_outline(self, img, text, pos, scale, color, thickness):
        cv2.putText(img, text, pos, 0, scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(img, text, pos, 0, scale, color, thickness, cv2.LINE_AA)

    def _draw_radar(self, canvas, start_x, start_y, inventory, sims, w_half):
        self.draw_text_outline(canvas, "RESONANCE RADAR:", (start_x, start_y), 0.5, (200, 200, 255), 1)
        for i, art in enumerate(inventory):
            y_pos = start_y + 20 + (i * 25)
            # ФИКС: val теперь гарантированно число
            val = float(np.clip(sims[i], 0, 1.5))
            pct = int(val * 100)
            
            is_active = val >= 0.4
            bar_color = (0, 255, 100) if is_active else (0, 150, 255)
            
            display_name = art.upper()[:16] + ".." if len(art) > 16 else art.upper()
            self.draw_text_outline(canvas, display_name, (start_x, y_pos + 12), 0.45, bar_color if is_active else (150,150,150), 1)
            
            bar_x = start_x + 160
            cv2.rectangle(canvas, (bar_x, y_pos), (bar_x + 150, y_pos + 14), (40, 40, 40), -1)
            # ФИКС: fill_w больше не взорвется
            fill_w = int(min(150, 150 * val))
            cv2.rectangle(canvas, (bar_x, y_pos), (bar_x + fill_w, y_pos + 14), bar_color, -1)
            cv2.line(canvas, (bar_x + int(150*0.4), y_pos), (bar_x + int(150*0.4), y_pos + 14), (255, 0, 0), 2)
            self.draw_text_outline(canvas, f"{pct}%", (bar_x + 155, y_pos + 12), 0.45, bar_color, 1)

    def _draw_axes_debug(self, canvas, start_x, start_y, axes_10d):
        """Отрисовка 10 осей геймпада (Дебаггер)"""
        labels =["LX", "LY", "RX", "RY", "TRG", "BMP", "Y/A", "B/X", "DP-X", "DP-Y"]
        self.draw_text_outline(canvas, "NEURAL AXES (10D):", (start_x, start_y), 0.5, (255, 150, 0), 1)
        
        for i in range(10):
            x_pos = start_x + (i * 32)
            val = axes_10d[i] # от -1 до 1
            
            # Рисуем вертикальный барчик от -1 до 1
            center_y = start_y + 40
            bar_h = int(abs(val) * 20)
            
            cv2.line(canvas, (x_pos+5, center_y-20), (x_pos+5, center_y+20), (50,50,50), 2)
            cv2.line(canvas, (x_pos, center_y), (x_pos+10, center_y), (100,100,100), 1) # Ноль
            
            color = (0, 255, 100) if val > 0 else (0, 100, 255)
            if val > 0:
                cv2.rectangle(canvas, (x_pos+3, center_y - bar_h), (x_pos+8, center_y), color, -1)
            else:
                cv2.rectangle(canvas, (x_pos+3, center_y), (x_pos+8, center_y + bar_h), color, -1)
                
            self.draw_text_outline(canvas, labels[i], (x_pos-2, center_y + 35), 0.35, (200,200,200), 1)

    def draw_ui(self, canvas, w, h, p1_axes, p2_axes):
        half_w = w // 2
        cv2.line(canvas, (half_w, 0), (half_w, h), (100, 100, 100), 2)

        cv2.rectangle(canvas, (20, 20), (int(20 + (half_w - 40) * (self.p1_hp/100)), 35), (0, 255, 100), -1)
        cv2.rectangle(canvas, (half_w + 20, 20), (int(half_w + 20 + (half_w - 40) * (self.p2_hp/100)), 35), (0, 50, 255), -1)

        c1 = (0, 255, 255) if self.p1_state == "IDLE" else (0, 50, 255)
        t1 = "MEDITATING" if self.p1_state == "IDLE" else f"DEFENDING {self.p1_incoming.upper()}"
        self.draw_text_outline(canvas, f"YOU: {self.p1['name']}[{t1}]", (20, 60), 0.6, c1, 2)

        c2 = (0, 255, 255) if self.p2_state == "IDLE" else (0, 50, 255)
        t2 = "MEDITATING" if self.p2_state == "IDLE" else f"DEFENDING {self.p2_incoming.upper()}"
        self.draw_text_outline(canvas, f"BOT: {self.p2['name']} [{t2}]", (half_w + 20, 60), 0.6, c2, 2)

        self._draw_radar(canvas, 20, 90, self.p1['inventory'], self.p1_current_sims, half_w)
        self._draw_radar(canvas, half_w + 20, 90, self.p2['inventory'], self.p2_current_sims, half_w)

        # Выводим Дебаггер Осей!
        self._draw_axes_debug(canvas, 20, 180, p1_axes)
        self._draw_axes_debug(canvas, half_w + 20, 180, p2_axes)

        y_pos = h - (len(self.combat_log) * 20) - 20
        for i, line in enumerate(self.combat_log):
            lc = (50, 255, 100) if "[P1]" in line else ((50, 100, 255) if "[P2]" in line else (200, 200, 200))
            ts = cv2.getTextSize(line, 0, 0.5, 2)[0]
            self.draw_text_outline(canvas, line, (half_w - ts[0]//2, y_pos + i*20), 0.5, lc, 2)

        if self.winner:
            self.draw_text_outline(canvas, f"VICTORY: {self.winner.upper()}", (half_w - 200, h//2), 1.2, (0, 255, 255), 3)
