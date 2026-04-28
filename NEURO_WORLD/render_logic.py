import torch, cv2, numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, AutoPipelineForImage2Image, LCMScheduler, AutoencoderTiny

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
        print(f"🎨 Рендер: {self.mode.upper()}, Размерность латентов: {self.latent_dim}")
        
        if compile_unet:
            print("🚀 Компиляция UNet...")
            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=False)
            
        self._warmup()

    def _warmup(self):
        print("🔥 Прогрев...")
        dummy_image = Image.fromarray(np.zeros((384, 512, 3), dtype=np.uint8))
        for _ in range(2):
            # 🔥 ФИКС: Используем именованные аргументы, чтобы избежать путаницы
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
            kwargs.update({"strength": strength, "num_inference_steps": 3, "guidance_scale": 1.0})
            
        if prompt_embeds is not None:
            kwargs["prompt_embeds"] = prompt_embeds
        elif prompt is not None:
            kwargs["prompt"] = prompt
            
        return self.pipe(**kwargs).images[0]

    def match_palette(self, source, target):
        src, tgt = np.array(source).astype(np.float32), np.array(target).astype(np.float32)
        mu_s, std_s = cv2.meanStdDev(src); mu_t, std_t = cv2.meanStdDev(tgt)
        corr = (tgt - mu_t.reshape(1,1,3)) * (std_s.reshape(1,1,3)/(std_t.reshape(1,1,3)+1e-5)) + mu_s.reshape(1,1,3)
        return Image.fromarray(np.clip(corr * 0.3 + tgt * 0.7, 0, 255).astype(np.uint8))
