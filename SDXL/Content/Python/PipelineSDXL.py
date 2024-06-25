import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler, KDPM2DiscreteScheduler, PNDMScheduler, EulerAncestralDiscreteScheduler, DDPMScheduler, DEISMultistepScheduler, DPMSolverMultistepScheduler, KDPM2AncestralDiscreteScheduler, EDMEulerScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, UniPCMultistepScheduler, DPMSolverSinglestepScheduler
from PySide6.QtCore import QSettings
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from compel import Compel, ReturnedEmbeddingsType
import os

# from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from random import randint
import re
# from accelerate import Accelerator


# Main Diffusion Pipeline Class
# This class is responsible for loading the models and generating images
# Settings for Image Generation are set through QtWindow
class SDXLPipeline:
    def __init__(self, filepath):
        self.filePath = filepath
        #current file path
        self.modelFolderPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.basePath = os.path.join(self.modelFolderPath, "stable-diffusion-xl-base-1.0-6.17.2024")
        self.refinerPath = os.path.join(self.modelFolderPath, "stable-diffusion-xl-refiner-1.0-6.17.2024")
        self.lightningPath = os.path.join(self.modelFolderPath, "stable-diffusion-lightning-6.17.2024")

    # Load model function
    # This function loads the models based on the settings provided in QtWindow
    # Called when the user clicks the Load Model button
    def load_models(self, loadSettings):
        self.base = None
        self.refiner = None
        targets = []
        model_id = loadSettings['model']
        refiner_id = loadSettings['refiner']
        if loadSettings['model'] == "Stable Diffusion XL Lightning":
            repo = self.lightningPath
            ckpt = "sdxl_lightning_4step_unet.safetensors"
            unet_config = UNet2DConditionModel.load_config(model_id, subfolder="unet")
            unet = UNet2DConditionModel.from_config(unet_config).to("cuda", torch.float16)
            unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
            self.base = DiffusionPipeline.from_pretrained(model_id, unet=unet, torch_dtype=torch.float16, variant="fp16")

        elif loadSettings['model'] == "Stable Diffusion XL":
            self.base = DiffusionPipeline.from_pretrained(
                self.basePath,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
        else:
            self.base = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
        if loadSettings['refiner'] is not None:
            self.refiner = DiffusionPipeline.from_pretrained(
                self.refinerPath,
                text_encoder_2=self.base.text_encoder_2,
                # vae=self.base.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                )
            if loadSettings['CPUOffload'] == True:
                self.refiner.enable_model_cpu_offload()
            else:
                self.refiner.to("cuda")
                    
            for item in self.refiner.components:
                if "unet" in item or "vae" in item or "text_encoder" in item:
                    module = getattr(self.refiner, item, None)  # Attempt to retrieve variable by name
                    if module is not None:
                        targets.append(module)

        if loadSettings['CPUOffload'] == True:
            self.base.enable_model_cpu_offload()
        else:
            self.base.to("cuda")
        self.base.enable_xformers_memory_efficient_attention()
        self.save_settings()

        for item in self.base.components:
            if "unet" in item or "vae" in item or "text_encoder" in item:
                module = getattr(self.base, item, None)  # Attempt to retrieve variable by name
                if module is not None:
                    targets.append(module)

        self.conv_layers = []
        self.conv_layers_original_paddings = []
        for target in targets:
            for module in target.modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
                    self.conv_layers.append(module)
                    self.conv_layers_original_paddings.append(module.padding_mode)

    # Generate Image function
    # This function generates an image based on the settings provided in QtWindow
    # Called when the user clicks the Generate Image button
    # Returns the generated image
    def generate_image(self, index):
        self.prompt = self.generateSettings["prompt"].strip()
        
        base_compel = Compel(
        tokenizer=[self.base.tokenizer, self.base.tokenizer_2] ,
        text_encoder=[self.base.text_encoder, self.base.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
        )
        
        conditioning_embeds, pooled_embeds = base_compel(self.generateSettings["prompt"])
        negative_conditioning_embeds, negative_pooled_embeds = base_compel(self.generateSettings["negativePrompt"])
        
        if self.refiner is not None:
            refiner_compel = Compel(
            tokenizer=self.refiner.tokenizer_2,
            text_encoder=self.refiner.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            )
            refiner_conditioning_embeds, refiner_pooled_embeds = refiner_compel(self.generateSettings["prompt"])
            refiner_negative_conditioning_embeds, refiner_negative_pooled_embeds = refiner_compel(self.generateSettings["negativePrompt"])

        for cl, opad in zip(self.conv_layers, self.conv_layers_original_paddings):
            if self.generateSettings["tiling"] == True:
                cl.padding_mode = "circular"
            else:
                cl.padding_mode = opad


        generator = torch.Generator().manual_seed(self.generateSettings["seed"] + index)
        if self.generateSettings["refiner"] == True:
            image = self.base(
                prompt_embeds=conditioning_embeds,
                pooled_prompt_embeds=pooled_embeds,
                # prompt = self.prompt,
                # negative_prompt = self.generateSettings["negativePrompt"],
                negative_prompt_embeds=negative_conditioning_embeds,
                negative_pooled_prompt_embeds=negative_pooled_embeds,
                num_inference_steps=self.generateSettings["numInferenceSteps"],
                denoising_end=self.generateSettings["denoisingEnd"],
                guidance_scale = self.generateSettings["CFG"],
                height = self.generateSettings["height"],
                width = self.generateSettings["width"],
                generator = generator,
                output_type="latent",
            ).images
            image = self.refiner(
                prompt_embeds=refiner_conditioning_embeds,
                pooled_prompt_embeds=refiner_pooled_embeds,
                # prompt = self.prompt,
                # negative_prompt = self.generateSettings["negativePrompt"],
                negative_prompt_embeds=refiner_negative_conditioning_embeds,
                negative_pooled_prompt_embeds=refiner_negative_pooled_embeds,
                num_inference_steps= self.generateSettings["numInferenceSteps"],
                denoising_start= self.generateSettings["denoisingEnd"],
                guidance_scale = self.generateSettings["CFG"],
                height = self.generateSettings["height"],
                width =  self.generateSettings["width"],
                generator = generator,
                image=image,
            ).images[0]
            
        else:
            image = self.base(
                prompt_embeds=conditioning_embeds,
                pooled_prompt_embeds=pooled_embeds,
                # prompt = self.prompt,
                # negative_prompt = self.generateSettings["negativePrompt"],
                negative_prompt_embeds=negative_conditioning_embeds,
                negative_pooled_prompt_embeds=negative_pooled_embeds,
                num_inference_steps=self.generateSettings["numInferenceSteps"],
                guidance_scale = self.generateSettings["CFG"],
                height =  self.generateSettings["height"],
                width = self.generateSettings["width"],
                generator = generator,
            ).images[0]

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return image
    

    # Then, create a dictionary to map scheduler names to their classes
    SCHEDULER_CLASSES = {
        "DDIMScheduler": DDIMScheduler,
        "KDPM2DiscreteScheduler": KDPM2DiscreteScheduler,
        "PNDMScheduler": PNDMScheduler,
        "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
        "DDPMScheduler": DDPMScheduler,
        "DEISMultistepScheduler": DEISMultistepScheduler,
        "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
        "KDPM2AncestralDiscreteScheduler": KDPM2AncestralDiscreteScheduler,
        "EDMEulerScheduler": EDMEulerScheduler,
        "HeunDiscreteScheduler": HeunDiscreteScheduler,
        "EulerDiscreteScheduler": EulerDiscreteScheduler,
        "UniPCMultistepScheduler": UniPCMultistepScheduler,
        "DPMSolverSinglestepScheduler": DPMSolverSinglestepScheduler
    }

    # Set Scheduler function
    def set_scheduler(self, scheduler_str):
        if scheduler_str in self.SCHEDULER_CLASSES:
            SchedulerClass = self.SCHEDULER_CLASSES.get(scheduler_str)
        if SchedulerClass:
            self.base.scheduler = SchedulerClass.from_config(self.base.scheduler.config, timestep_spacing="trailing")
            print(f"Scheduler set to {scheduler_str}")
        else:
            print(f"Scheduler {scheduler_str} not found")

    # saving qsettings for the UI
    def save_settings(self):
        self.settings = QSettings("Ashill", "SDtoUnreal")
        pattern = r'(?<=\.)([^.]*?)\'>'
        compatible_schedulers = []
        for scheduler in list(self.base.scheduler.compatibles):
            scheduler_name = str(scheduler)
            match = re.search(pattern, scheduler_name)
            if match:
                result = match.group(1)
                compatible_schedulers.append(result)
        self.settings.setValue("SchedulersList",list(compatible_schedulers))

    # loading qsettings for the UI
    def set_settings(self, settingsDict):
        self.generateSettings = settingsDict
    

