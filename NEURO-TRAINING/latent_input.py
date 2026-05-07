import numpy as np
import pygame

class GamepadHandler:
    def __init__(self):
        self.joy = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
        if self.joy: self.joy.init()
        self.axes_10d = np.zeros(10, dtype=np.float32)
        self.trig_l_init, self.trig_r_init = False, False

    def _dz(self, val): return 0.0 if abs(val) < 0.15 else val
    def _read_trigger(self, axis_idx, is_left):
        if not self.joy or self.joy.get_numaxes() <= axis_idx: return 0.0
        val = self.joy.get_axis(axis_idx)
        if is_left:
            if val != 0.0: self.trig_l_init = True
            if not self.trig_l_init: val = -1.0
        else:
            if val != 0.0: self.trig_r_init = True
            if not self.trig_r_init: val = -1.0
        return (val + 1.0) / 2.0

    def update(self):
        if not self.joy: 
            self.axes_10d.fill(0); return
        pygame.event.pump()
        num_ax, num_btn = self.joy.get_numaxes(), self.joy.get_numbuttons()
        # Маппинг Linux Sony (DualShock/DualSense)
        self.axes_10d[0] = self._dz(self.joy.get_axis(0)) # LX
        self.axes_10d[1] = self._dz(self.joy.get_axis(1)) # LY
        self.axes_10d[2] = self._dz(self.joy.get_axis(3)) if num_ax > 3 else 0 # RX
        self.axes_10d[3] = self._dz(self.joy.get_axis(4)) if num_ax > 4 else 0 # RY
        self.axes_10d[4] = self._read_trigger(5, False) - self._read_trigger(2, True) # TRG
        self.axes_10d[5] = (float(self.joy.get_button(5)) if num_btn > 5 else 0) - (float(self.joy.get_button(4)) if num_btn > 4 else 0)
        self.axes_10d[6] = (float(self.joy.get_button(3)) if num_btn > 3 else 0) - (float(self.joy.get_button(0)) if num_btn > 0 else 0)
        self.axes_10d[7] = (float(self.joy.get_button(1)) if num_btn > 1 else 0) - (float(self.joy.get_button(2)) if num_btn > 2 else 0)
        hat = self.joy.get_hat(0) if self.joy.get_numhats() > 0 else (0, 0)
        self.axes_10d[8], self.axes_10d[9] = float(hat[0]), float(hat[1])

class MouseKeyboardHandler:
    def __init__(self):
        self.axes_10d = np.zeros(10, dtype=np.float32)
        self.mouse_wheel_y = 0.0

    def handle_event(self, event):
        if event.type == pygame.MOUSEWHEEL: self.mouse_wheel_y = np.clip(float(event.y), -1, 1)

    def update(self):
        keys = pygame.key.get_pressed()
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        lmb, _, rmb = pygame.mouse.get_pressed()
        self.axes_10d.fill(0)
        self.axes_10d[0] = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
        self.axes_10d[1] = float(keys[pygame.K_s]) - float(keys[pygame.K_w])
        self.axes_10d[2] = np.clip(mouse_dx / 15.0, -1, 1)
        self.axes_10d[3] = np.clip(-mouse_dy / 15.0, -1, 1)
        self.axes_10d[4] = float(rmb) - float(lmb)
        self.axes_10d[5] = self.mouse_wheel_y + (float(keys[pygame.K_e]) - float(keys[pygame.K_q]))
        self.axes_10d[6] = float(keys[pygame.K_r]) - float(keys[pygame.K_f])
        self.axes_10d[7] = float(keys[pygame.K_x]) - float(keys[pygame.K_z])
        self.axes_10d[8] = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        self.axes_10d[9] = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])
        self.mouse_wheel_y *= 0.5

class UnifiedInputController:
    def __init__(self, seed, use_bci=True, is_bot=False, debug=False):
        self.is_bot = is_bot
        self.debug = debug
        self.gamepad = GamepadHandler() if not is_bot else None
        self.mouse_kb = MouseKeyboardHandler() if not is_bot else None
        self.bci = None
        
        if use_bci:
            try:
                from bci_logic import FreeEEG
                if FreeEEG.is_available(): self.bci = FreeEEG()
            except: pass
            if self.bci is None:
                from NEURO_WORLD import MockBCI
                self.bci = MockBCI()

        self.axes_10d = np.zeros(10, dtype=np.float32)
        self.ciplv_vec = np.zeros(28, dtype=np.float32)
        self.persistence = 0.0 
        
        np.random.seed(seed)
        self.projection_matrix = np.random.randn(10, 28).astype(np.float32)
        self.projection_matrix /= (np.linalg.norm(self.projection_matrix, axis=0) + 1e-6)
        self.proj_inv = np.linalg.pinv(self.projection_matrix)

    def handle_event(self, event):
        if self.mouse_kb: self.mouse_kb.handle_event(event)

    def update(self, current_time):
        if self.is_bot:
            self.ciplv_vec = np.dot(self.axes_10d, self.projection_matrix)
            self.persistence = min(1.0, np.linalg.norm(self.axes_10d) / 1.5)
            return

        # 1. Механика
        m_10d = np.zeros(10, dtype=np.float32)
        if self.gamepad: self.gamepad.update(); m_10d += self.gamepad.axes_10d
        if self.mouse_kb: self.mouse_kb.update(); m_10d += self.mouse_kb.axes_10d
        
        # 2. Мозг
        b_10d = np.zeros(10, dtype=np.float32); b_p = 0.0
        if self.bci:
            is_mock = hasattr(self.bci, 'update_sim_internal')
            if is_mock:
                self.bci.update_sim_internal(current_time, 7.0)
                # ПОКАЗЫВАЕМ МОК ТОЛЬКО В ДЕБАГЕ
                if self.debug:
                    b_10d = np.dot(self.bci.ciplv_vec, self.proj_inv)
                    b_p = self.bci.persistence
            else:
                # РЕАЛЬНЫЙ FreeEEG
                self.bci.update_sim()
                # ПОКАЗЫВАЕМ ТОЛЬКО ЕСЛИ ЕСТЬ КОННЕКТ
                if getattr(self.bci, 'is_connected', False):
                    b_10d = np.dot(np.nan_to_num(self.bci.ciplv_vec), self.proj_inv)
                    b_p = self.bci.persistence

        self.axes_10d = np.nan_to_num(np.clip(m_10d + b_10d, -1.5, 1.5))
        self.ciplv_vec = np.dot(self.axes_10d, self.projection_matrix)
        self.persistence = float(max(b_p, np.linalg.norm(self.axes_10d) / 1.5))
