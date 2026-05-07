#!/usr/bin/env python3
import torch
from engine_core import NeuroCultivationEngine
from render_logic import NeuroRender
from vision_logic import QwenVision

# =========================================================
# НАСТРОЙКИ СМЫСЛОВ (ОДИН РАЗ ЗДЕСЬ)
# =========================================================
PURE_BASE = "meditation in void"
ESSENCE_BASE = "meditation in void with manifestation or essence of "
VOID_W = "nothing"

# Просто список твоих хотелок:
#LIST_OF_ESSENCES = ["FIRE", "WATER", "LEAVES", "LIGHTNING"]
LIST_OF_ESSENCES = ["LIGHTNING", "CHAOS", "FRACTAL", "ABYSS"]
#LIST_OF_ESSENCES = ["ANEMO", "GEO", "ELECTRO", "DENDRO", "HYDRO", "PYRO", "CRYO"]

ANS = " Answer only one word YES or NO. without any comments"

def create_ritual(render, essences):
    tokenizer = render.pipe.tokenizer
    pure_emb = render.encode_prompt(PURE_BASE)
    ess_base_emb = render.encode_prompt(ESSENCE_BASE)
    slot = tokenizer(ESSENCE_BASE).input_ids.index(49407) + 1
    void_t = render.encode_prompt(VOID_W)[0, 1, :]
    
    # Универсальный промпт растворения
    pure_p = "Is there meditation in void but WITHOUT any additional manifestation or special essence?" + ANS
    
    pipe = []
    # Старт
    pipe.append({'t': void_t, 'b': torch.zeros(28), 'p': pure_p, 'a': False, 'l': "DISSOLVE", 'b_emb': pure_emb, 'slot': 0})

    for word in essences:
        emb = render.encode_prompt(word.lower())
        t_eos = tokenizer(word.lower()).input_ids.index(49407)
        vec = emb[0, 1, :] if (t_eos - 1) == 1 else torch.mean(emb[0, 1:t_eos, :], dim=0)
        
        torch.manual_seed(len(word) + 42)
        brain_sig = torch.nn.functional.normalize(torch.randn(28), dim=0)
        
        # Активная стадия
        pipe.append({
            't': vec, 'b': brain_sig, 'a': True, 'l': word,
            'p': f"Is there meditation in void and with manifestation or essence of {word}?" + ANS,
            'b_emb': ess_base_emb, 'slot': slot
        })
        # Растворение
        pipe.append({'t': void_t, 'b': torch.zeros(28), 'p': pure_p, 'a': False, 'l': "DISSOLVE", 'b_emb': pure_emb, 'slot': 0})
        
    return void_t, pipe

if __name__ == "__main__":
    # Инициализация стабильных модулей
    render_init = NeuroRender(mode="lcm")
    vision_shared = QwenVision(model="qwen3.5:0.8b")
    
    # Сборка ритуала
    void_t, pipeline = create_ritual(render_init, LIST_OF_ESSENCES)

    # Движок (использует те же объекты)
    engine = NeuroCultivationEngine(
        render_mode="lcm", vision_obj=vision_shared, seed=123,
        physics_cfg={"decay": 0.82, "impulse": 0.22, "s_mov": 0.55, "s_rest": 0.85, "inertia": 0.0, "color_p": 0.01}
    )

    # Запуск
    engine.run(
        void_t=void_t, pipeline=pipeline,
        win_size=(1280, 720), req_ok=10, v_int=3.0, ok_val="yes"
    )
