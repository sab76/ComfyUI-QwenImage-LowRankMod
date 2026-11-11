import comfy.model_base
import comfy.model_detection
import comfy.model_management as mm
import comfy.model_patcher
import comfy.utils
import folder_paths

from .model import LowRankModQwenImageTransformer


def load_low_rank_qwen_image(path):
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    if "time_text_embed.modulation_down.weight" not in sd:
        raise ValueError("Not a low-rank modulation Qwen-Image checkpoint, use the regular diffusion model loader.")
    sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
    model_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)

    load_device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    quantized = model_config.quant_config is not None
    weight_dtype = None if quantized else comfy.utils.weight_dtype(sd)
    unet_dtype = mm.unet_dtype(model_params=comfy.utils.calculate_parameters(sd), supported_dtypes=model_config.supported_inference_dtypes, weight_dtype=weight_dtype)
    manual_cast_dtype = mm.unet_manual_cast(None if quantized else unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)
    model_config.unet_config["modulation_rank"] = sd["time_text_embed.modulation_down.weight"].shape[0]
    model_config.unet_config["compute_dtype"] = manual_cast_dtype or unet_dtype

    model = comfy.model_base.QwenImage(model_config, unet_model=LowRankModQwenImageTransformer)
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=offload_device)
    if not mm.is_device_cpu(offload_device):
        model.to(offload_device)
    model.load_model_weights(sd, "", assign=patcher.is_dynamic())
    patcher.cached_patcher_init = (load_low_rank_qwen_image, (path,))
    return patcher


class LoadLowRankModQwenImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (folder_paths.get_filename_list("diffusion_models"),)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "advanced/loaders"
    DESCRIPTION = "Load a Qwen-Image checkpoint converted with convert.py (shared low-rank AdaLN modulation)."

    def load_unet(self, unet_name):
        return (load_low_rank_qwen_image(folder_paths.get_full_path_or_raise("diffusion_models", unet_name)),)
