import torch, cv2, numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, AutoPipelineForImage2Image, LCMScheduler, AutoencoderTiny

# --- ФИКС: Объявляем глобальные переменные, чтобы модуль знал размеры ---
GW, GH = 512, 384

class NeuroRender:
    def __init__(self, mode="turbo", compile_unet=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.mode = mode.lower()
        
        if self.mode == "turbo":
            self.pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=self.dtype, variant="fp16").to(self.device)
        else:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", torch_dtype=self.dtype).to(self.device)
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.safety_checker = None
        self.pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=self.dtype).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        
        self.latent_dim = self.pipe.text_encoder.config.hidden_size
        
        if compile_unet and self.device == 'cuda':
            try:
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print(f"[!] Ошибка компиляции UNet: {e}")
        self._warmup()

    def _warmup(self):
        dummy_image = Image.fromarray(np.zeros((GH, GW, 3), dtype=np.uint8))
        with torch.no_grad():
            for _ in range(2):
                self.generate(prompt="warmup", image=dummy_image, strength=1.0)

    def encode_prompt(self, prompt_text):
        with torch.no_grad():
            return self.pipe.text_encoder(self.pipe.tokenizer(
                prompt_text, return_tensors="pt", padding="max_length", 
                max_length=self.pipe.tokenizer.model_max_length, truncation=True
            ).input_ids.to(self.device))[0]

    def generate(self, prompt=None, prompt_embeds=None, image=None, strength=0.5):
        kwargs = {"image": image}
        
        if self.mode == "turbo":
            kwargs.update({"strength": max(0.5, strength), "num_inference_steps": 2, "guidance_scale": 0.0})
        else:
            num_steps = 3
            min_safe_strength = (1.0 / num_steps) + 0.02
            kwargs.update({"strength": max(strength, min_safe_strength), "num_inference_steps": num_steps, "guidance_scale": 1.0})
            
        if prompt_embeds is not None:
            kwargs["prompt_embeds"] = prompt_embeds
        elif prompt is not None:
            kwargs["prompt"] = prompt
            
        return self.pipe(**kwargs).images[0]
