import requests, base64, io, re

class QwenVision:
    def __init__(self, model="qwen3.5:2b", prompt="Identify the main object."):
        self.url = "http://localhost:11434/api/chat"
        self.model = model
        self.sys_prompt = prompt

    def ask(self, pil_img, question=None):
        img = pil_img.copy().convert("RGB")
        img.thumbnail((448, 448))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt_to_use = question if question else self.sys_prompt

        payload = {
            "model": self.model,
            "messages":[
                {
                    "role": "user", 
                    "content": prompt_to_use, 
                    "images":[img_b64]
                }
            ],
            "stream": False,
            "think": False,       # ЗАЩИТА ОТ ТОРМОЗОВ РИЗОНИНГА
            "options": {
                "temperature": 0.0,
                "num_predict": 500,
                "top_k": 1
            }
        }

        try:
            r = requests.post(self.url, json=payload, timeout=5)
            if r.status_code == 200:
                ans = r.json().get('message', {}).get('content', '').strip()
                ans = re.sub(r'<think>.*?</think>', '', ans, flags=re.DOTALL).strip()
                return ans.replace('"', '').replace('*', '').lower()
        except Exception as e:
            print(f"[VISION ERROR] {e}")
            
        return "error"
