import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Skip vision and MTP weights for Qwen3.5 multimodal checkpoints
                if weight_name.startswith(("visual.", "vision_", "multi_modal_projector.")):
                    continue
                if ".mtp_" in weight_name:
                    continue

                # Map the language submodel in a Qwen3.5 multimodal checkpoint
                # onto nano-vLLM's ``model`` module.  Keeping the ``model.``
                # prefix is important: the target is ``model.layers.*``, not
                # ``layers.*`` at the CausalLM root.
                mapped_name = weight_name
                if mapped_name.startswith("model.language_model."):
                    mapped_name = "model." + mapped_name[len("model.language_model."):]
                elif mapped_name.startswith("text_model."):
                    mapped_name = "model." + mapped_name[len("text_model."):]

                for k in packed_modules_mapping:
                    if k in mapped_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = mapped_name.replace(k, v)
                        try:
                            param = model.get_parameter(param_name)
                        except (AttributeError, KeyError):
                            continue
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(mapped_name)
                    except (AttributeError, KeyError):
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
