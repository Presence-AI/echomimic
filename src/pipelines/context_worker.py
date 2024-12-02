import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import torch.nn.functional as F
import os
from einops import rearrange
# from diffusers.models import UNet2DConditionModel
from src.models.unet_2d_condition import UNet2DConditionModel


from src.models.unet_3d_echo import EchoUNet3DConditionModel
from diffusers.schedulers import DDIMScheduler
from typing import Optional, Union, List, Callable
from diffusers import DiffusionPipeline
from src.utils.step_func import origin_by_velocity_and_sample, psuedo_velocity_wrt_noisy_and_timestep, get_alpha
from src.models.mutual_self_attention import ReferenceAttentionControl

import time
from diffusers import AutoencoderKL, DDIMScheduler


class WorkerPipeline(DiffusionPipeline):
    def __init__(self, scheduler_arg, config, infer_config, device, step):
        super().__init__()
        weight_dtype = torch.float16 if config.weight_dtype == "fp16" else torch.float32

        reference_unet = self._init_reference_unet(config, weight_dtype, device)
        denoising_unet = self._init_denoising_unet(config, infer_config, weight_dtype, device)
        scheduler = DDIMScheduler(**scheduler_arg)
        scheduler.set_timesteps(step, device=device)

        self.register_modules(
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            scheduler=scheduler
        )
        
    def _init_reference_unet(self, config, weight_dtype, device):
        reference_unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=weight_dtype, device=device)
        reference_unet.load_state_dict(
            torch.load(config.reference_unet_path, map_location="cpu"),
        )
        print("-------- finishing loading unet model--------------")
        return reference_unet
        
    def _init_denoising_unet(self, config, infer_config, weight_dtype, device):
       if os.path.exists(config.motion_module_path):
           denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
               config.pretrained_base_model_path,
               config.motion_module_path,
               subfolder="unet", 
               unet_additional_kwargs=infer_config.unet_additional_kwargs
           ).to(dtype=weight_dtype, device=device)
       else:
           denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
               config.pretrained_base_model_path,
               "",
               subfolder="unet",
               unet_additional_kwargs={
                   "use_motion_module": False,
                   "unet_use_temporal_attention": False,
                   "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
               }
           ).to(dtype=weight_dtype, device=device)
       
       denoising_unet.load_state_dict(
           torch.load(config.denoising_unet_path, map_location="cpu"),
           strict=False
       )
       print("-------- finishing loading denoising model (wk)--------------")
       return denoising_unet



@torch.no_grad() 
def context_worker(input_queue, output_queue,  scheduler_arg, config, infer_config, step):
    """Worker process for context processing"""

     # Initialize on second GPU
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)  

    
    pipe = WorkerPipeline(scheduler_arg, config, infer_config, device, step)
    
    reference_unet = pipe.reference_unet
    denoising_unet = pipe.denoising_unet
    scheduler = pipe.scheduler


    writer = ReferenceAttentionControl(
       reference_unet,
       do_classifier_free_guidance=False, 
       mode="write",
       batch_size=1,
       fusion_blocks="full"
    )
   
    reader = ReferenceAttentionControl(
       denoising_unet,
       do_classifier_free_guidance=False,
       mode="read", 
       batch_size=1,
       fusion_blocks="full"
    )


    # Move alphas_cumprod to device once
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    ref_images_first = None
    while True:
        data = input_queue.get()
        if data is None:  # Poison pill
            break
        context, t_i, video_length, latents, audio_fea_final, t,do_classifier_free_guidance,face_locator_tensor,c_face_locator_tensor, state_dict, ref_image_latents = data


        start_time = time.time()
        
        # Move tensors to worker GPU
        latents = latents.to(device,  non_blocking=True)
        audio_fea_final = audio_fea_final.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        face_locator_tensor = face_locator_tensor.to(device, non_blocking=True)
        c_face_locator_tensor = c_face_locator_tensor.to(device, non_blocking=True)
        if ref_image_latents is not None:
            ref_image_latents = ref_image_latents.to(device, non_blocking=True)

        ##############################
        if t_i == 0:
            ref_image_latents = ref_image_latents.to(device, non_blocking=True)
            reference_unet(
               ref_image_latents,
               torch.zeros_like(t).to(device),
               encoder_hidden_states=None,
               return_dict=False,
            )
            reader.update(writer, do_classifier_free_guidance=do_classifier_free_guidance)
            if ref_image_latents != None:
                ref_images_first = ref_image_latents
        ##############################

        
        
        # Create new context
        new_context = [[0 for _ in range(len(context[c_j]))] for c_j in range(len(context))]
        for c_j in range(len(context)):
            for c_i in range(len(context[c_j])):
                new_context[c_j][c_i] = (context[c_j][c_i] + t_i * 2) % video_length

        # Prepare inputs
        latent_model_input = (
            torch.cat([latents[:, :, c] for c in new_context])
            .to(device)
            .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
        )
        c_audio_latents = torch.cat([audio_fea_final[:, c] for c in new_context]).to(device)
        audio_latents = torch.cat([torch.zeros_like(c_audio_latents), c_audio_latents], 0)
       

        # Scale input
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        t_2 = time.time()

        # UNet forward pass
        pred = denoising_unet(
            latent_model_input,
            t,
            encoder_hidden_states=None,
            audio_cond_fea=audio_latents if do_classifier_free_guidance else c_audio_latents,
            face_musk_fea=face_locator_tensor if do_classifier_free_guidance else c_face_locator_tensor,
            return_dict=False,
        )[0].detach()

        t_3 = time.time()
        print("denoise context wk c1: ", t_3 - t_2)
        t_al = time.time()
        x_pred = origin_by_velocity_and_sample(pred, latent_model_input, scheduler.alphas_cumprod, t)
        pred = psuedo_velocity_wrt_noisy_and_timestep(
            latent_model_input, x_pred, scheduler.alphas_cumprod, t, torch.ones_like(t) * (-1)
        )

        output_pred = pred.to('cuda:0', non_blocking=True)      
        # torch.cuda.empty_cache()
        if t_i % 2 == 0:
            writer.clear()
            reader.clear()
            reference_unet(
               ref_images_first,
               torch.zeros_like(t),
               encoder_hidden_states=None,
               return_dict=False,
            )
            reader.update(writer, do_classifier_free_guidance=do_classifier_free_guidance)
        
        output_queue.put((new_context, output_pred))


@torch.no_grad()
def context_worker_c0(input_queue, output_queue,  scheduler_arg, config, infer_config, step):
    """Worker process for context processing"""

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pipe = WorkerPipeline(scheduler_arg, config, infer_config, device, step)
    reference_unet = pipe.reference_unet
    denoising_unet = pipe.denoising_unet
    scheduler = pipe.scheduler

    writer = ReferenceAttentionControl(
       reference_unet,
       do_classifier_free_guidance=False, 
       mode="write",
       batch_size=1,
       fusion_blocks="full"
    )
   
    reader = ReferenceAttentionControl(
       denoising_unet,
       do_classifier_free_guidance=False,
       mode="read", 
       batch_size=1,
       fusion_blocks="full"
    )
    
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    ref_images_first = None
    while True:
        data = input_queue.get()
        if data is None:  # Poison pill
            break
            
        context, t_i, video_length, latents, audio_fea_final, t,do_classifier_free_guidance,face_locator_tensor,c_face_locator_tensor, state_dict, ref_image_latents = data


        ##############################
        if t_i == 0:
            torch.cuda.reset_peak_memory_stats()
            reference_unet(
               ref_image_latents,
               torch.zeros_like(t),
               encoder_hidden_states=None,
               return_dict=False,
            )
           
            reader.update(writer, do_classifier_free_guidance=do_classifier_free_guidance)
            if ref_image_latents != None:
                ref_images_first = ref_image_latents
        ##############################

        
        
        # Create new context
        new_context = [[0 for _ in range(len(context[c_j]))] for c_j in range(len(context))]
        for c_j in range(len(context)):
            for c_i in range(len(context[c_j])):
                new_context[c_j][c_i] = (context[c_j][c_i] + t_i * 2) % video_length

        # Prepare inputs
        latent_model_input = (
            torch.cat([latents[:, :, c] for c in new_context])
            .to(device)
            .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
        )
        c_audio_latents = torch.cat([audio_fea_final[:, c] for c in new_context]).to(device)
        audio_latents = torch.cat([torch.zeros_like(c_audio_latents), c_audio_latents], 0)

        # Scale input
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        start_time = time.time()
        # UNet forward pass
        pred = denoising_unet(
            latent_model_input,
            t,
            encoder_hidden_states=None,
            audio_cond_fea= c_audio_latents,
            face_musk_fea= c_face_locator_tensor,
            return_dict=False,
        )[0]

        x_pred = origin_by_velocity_and_sample(pred, latent_model_input, scheduler.alphas_cumprod, t)
        pred = psuedo_velocity_wrt_noisy_and_timestep(
            latent_model_input, x_pred, scheduler.alphas_cumprod, t, torch.ones_like(t) * (-1)
        )
    
        # Clear after each iteration
        if t_i % 2 == 0:
            writer.clear()
            reader.clear()
            reference_unet(
               ref_images_first,
               torch.zeros_like(t),
               encoder_hidden_states=None,
               return_dict=False,
            )
            reader.update(writer, do_classifier_free_guidance=do_classifier_free_guidance)

        output_queue.put((new_context, pred.detach()))


@torch.no_grad() 
def context_worker_c2(input_queue, output_queue,  scheduler_arg, config, infer_config, step):
    """Worker process for context processing"""
     # Initialize on second GPU
    device = torch.device('cuda:2')
    torch.cuda.set_device(device)  
    
    pipe = WorkerPipeline(scheduler_arg, config, infer_config, device, step)
    
    reference_unet = pipe.reference_unet
    denoising_unet = pipe.denoising_unet
    scheduler = pipe.scheduler


    writer = ReferenceAttentionControl(
       reference_unet,
       do_classifier_free_guidance=False, 
       mode="write",
       batch_size=1,
       fusion_blocks="full"
    )
   
    reader = ReferenceAttentionControl(
       denoising_unet,
       do_classifier_free_guidance=False,
       mode="read", 
       batch_size=1,
       fusion_blocks="full"
    )


    # Move alphas_cumprod to device once
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    ref_images_first = None
    while True:
        data = input_queue.get()
        if data is None:  # Poison pill
            break
        context, t_i, video_length, latents, audio_fea_final, t,do_classifier_free_guidance,face_locator_tensor,c_face_locator_tensor, state_dict, ref_image_latents = data


        start_time = time.time()
        
        # Move tensors to worker GPU
        latents = latents.to(device,  non_blocking=True)
        audio_fea_final = audio_fea_final.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        face_locator_tensor = face_locator_tensor.to(device, non_blocking=True)
        c_face_locator_tensor = c_face_locator_tensor.to(device, non_blocking=True)
        if ref_image_latents is not None:
            ref_image_latents = ref_image_latents.to(device, non_blocking=True)

        ##############################
        if t_i == 0:
            ref_image_latents = ref_image_latents.to(device, non_blocking=True)
            reference_unet(
               ref_image_latents,
               torch.zeros_like(t).to(device),
               encoder_hidden_states=None,
               return_dict=False,
            )
            reader.update(writer, do_classifier_free_guidance=do_classifier_free_guidance)
            if ref_image_latents != None:
                ref_images_first = ref_image_latents
        ##############################

        
        
        # Create new context
        new_context = [[0 for _ in range(len(context[c_j]))] for c_j in range(len(context))]
        for c_j in range(len(context)):
            for c_i in range(len(context[c_j])):
                new_context[c_j][c_i] = (context[c_j][c_i] + t_i * 2) % video_length

        # Prepare inputs
        latent_model_input = (
            torch.cat([latents[:, :, c] for c in new_context])
            .to(device)
            .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
        )
        c_audio_latents = torch.cat([audio_fea_final[:, c] for c in new_context]).to(device)
        audio_latents = torch.cat([torch.zeros_like(c_audio_latents), c_audio_latents], 0)
       

        # Scale input
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        t_2 = time.time()

        # UNet forward pass
        pred = denoising_unet(
            latent_model_input,
            t,
            encoder_hidden_states=None,
            audio_cond_fea=audio_latents if do_classifier_free_guidance else c_audio_latents,
            face_musk_fea=face_locator_tensor if do_classifier_free_guidance else c_face_locator_tensor,
            return_dict=False,
        )[0].detach()

        t_3 = time.time()
        print("denoise context wk c2: ", t_3 - t_2)
        t_al = time.time()
        x_pred = origin_by_velocity_and_sample(pred, latent_model_input, scheduler.alphas_cumprod, t)
        pred = psuedo_velocity_wrt_noisy_and_timestep(
            latent_model_input, x_pred, scheduler.alphas_cumprod, t, torch.ones_like(t) * (-1)
        )
        output_pred = pred.to('cuda:0', non_blocking=True)      
        # torch.cuda.empty_cache()
        if t_i % 2== 0:
            writer.clear()
            reader.clear()
            reference_unet(
               ref_images_first,
               torch.zeros_like(t),
               encoder_hidden_states=None,
               return_dict=False,
            )
            reader.update(writer, do_classifier_free_guidance=do_classifier_free_guidance)
        
        output_queue.put((new_context, output_pred))


def vae_worker(config, weight_dtype, input_queue, output_queue, device):
    # Initialize VAE in worker process 
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to(device, weight_dtype)
    print("-------- finishing vae worker--------------")

    while True:
        try:
            task = input_queue.get()
            if task is None:  # Stop signal
                print("******* vae break *****")
                break

            frame_idx, frame_latent = task
            frame_latent = frame_latent.to(device,  non_blocking=True)
            decoded = vae.decode(frame_latent).sample
            output_queue.put((frame_idx, decoded.detach()))

        except Exception as e:
            print(f"Error in VAE worker: {e}")
            continue
