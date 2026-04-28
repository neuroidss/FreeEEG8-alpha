import requests, base64, io, re, os
from PIL import Image

class QwenVision:
    def __init__(self, model="qwen3.5:0.8b", prompt="Identify the main object. 2 words only."):
        self.url = "http://localhost:11434/api/chat"
        self.model = model
        self.sys_prompt = prompt
        self.last_answer = "void"

    def ask(self, pil_img):
        img = pil_img.copy().convert("RGB")
        img.thumbnail((448, 448))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        payload = {
            "model": self.model,
            "messages":[
                {
                    "role": "user", 
                    "content": self.sys_prompt, 
                    "images": [img_b64]
                }
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 1000,
                "top_k": 1
            }
        }

        try:
            r = requests.post(self.url, json=payload, timeout=5)
            if r.status_code == 200:
                ans = r.json().get('message', {}).get('content', '').strip()
                ans = re.sub(r'<think>.*?</think>', '', ans, flags=re.DOTALL).strip()
                ans = ans.replace('"', '').replace('*', '')
                if ans:
                    self.last_answer = ans
                    return ans
        except Exception as e:
            pass
        return None
