# SPDX-License-Identifier: Apache-2.0
import torch
import os

from vllm import LLM, SamplingParams
# unsloth/Llama-3.2-1B-Instruct
from vllm.config import CompilationConfig

from huggingface_hub import hf_hub_download

from huggingface_hub import hf_hub_download
import json

#  import torch.distributed
#         if torch.distributed.get_rank() == 0:
#             print(f"before pass graph: {graph}")
# ===================== when there's error downloading model, run below first =========================
# try:
#     config_path = hf_hub_download(
#         repo_id="neuralmagic/Meta-Llama-3.1-8B-quantized.w8a8", 
#         filename="config.json", 
#         token=os.environ["HF_TOKEN"]
#     )
#     with open(config_path) as f:
#         config = json.load(f)
#     print("model_type in config:", "model_type" in config)
#     if "model_type" in config:
#         print(f"model_type: {config['model_type']}")
# except Exception as e:
#     print(f"Error: {e}")


# ===================== quantization =========================
# import os
# os.environ["LOCAL_RANK"] = "0"
# from transformers import AutoTokenizer, AutoModelForCausalLM

# MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID, device_map="auto", torch_dtype="auto",
# )
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# from llmcompressor.transformers import oneshot
# from llmcompressor.modifiers.quantization import QuantizationModifier

# # Configure the simple PTQ quantization
# recipe = QuantizationModifier(
#   targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# # Apply the quantization algorithm.
# oneshot(model=model, recipe=recipe)

# # Save the model: Meta-Llama-3-8B-Instruct-FP8-Dynamic
# SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
# model.save_pretrained(SAVE_DIR)
# tokenizer.save_pretrained(SAVE_DIR)


#===================== main logic =========================
config = CompilationConfig(
    level=3,
    splitting_ops=[],
    compile_sizes=[4],
    custom_ops=[
        "+rms_norm"]
)

# config.pass_config.enable_sequence_parallelism = True
# config.pass_config.enable_fusion = True
# config.pass_config.enable_async_tp = True
config.pass_config.enable_noop = True
sampling_params = SamplingParams(
        temperature=0,
    )

  # [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]

prompts = [
    "Can you calculate 19 + 20?",
    "How to make a cake?",
    "How old a baby can start to try solid food?",
    "What's pros and cons of using a pacifier for baby?"
]

# allenai/OLMo-1B-hf  unsloth/Meta-Llama-3.1-70B-Instruct
# unsloth/Llama-3.2-1B-Instruct
# meta-llama/Llama-3.1-8B-Instruct
# neuralmagic/Meta-Llama-3-8B-Instruct-FP8

# neuralmagic/Meta-Llama-3.1-8B-quantized.w8a8, _C.dynamic_scaled_int8_quant.default
# _C.cutlass_scaled_mm.default
# MODE model Qwen/Qwen3-30B-A3B


# llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
#           enforce_eager=False,
#           tensor_parallel_size=2,
#           kv_cache_dtype="fp8",
#           calculate_kv_scales=True,
#           compilation_config=config
#           )

# // Qwen/Qwen3-30B-A3B
llm = LLM(model="Qwen/Qwen3-30B-A3B",
          enforce_eager=False,
          tensor_parallel_size=2,
          enable_expert_parallel=True,
          distributed_executor_backend="mp",
          data_parallel_size=2,
          gpu_memory_utilization=0.8,
        #   kv_cache_dtype="fp8",
        #   calculate_kv_scales=True,
           compilation_config=config
           )


outputs = llm.generate(prompts=prompts,
    sampling_params=sampling_params,
        use_tqdm=True)

# Print the outputs.
print("\nGenerated Outputs:\n" + "-" * 60)
for output in outputs:
    prompt = output.prompt
    prompt_token_ids = len(output.prompt_token_ids)
    generated_text = output.outputs[0].text
    generated_text_ids = len(output.outputs[0].token_ids)
    print(f"Prompt:    {prompt!r}, token length: {prompt_token_ids}")
    print(f"Output:    {generated_text!r}, Output token length: {len(generated_text)}")
    print("-" * 60)




# ==================== old logic =========================    
# dtype = torch.float16
# hidden_size = 128
# num_tokens = 12

# x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
# x1 = x.clone()
# x2 = x.clone()
# residual= torch.randn(num_tokens, hidden_size, dtype=torch.float16, device="cuda") / 2
# residual1 = residual.clone()
# residual2 = residual.clone()
# import torch

# from tests.kernels.quant_utils import FP8_DTYPE
# from tests.kernels.utils import opcheck
# from vllm.model_executor.layers.layernorm import RMSNorm
# from torch._higher_order_ops.auto_functionalize import auto_functionalized

# layer = RMSNorm(hidden_size).to(dtype=dtype)
# layer.to(device="cuda")
# layer.weight.data.normal_(mean=1.0, std=0.1)
# # layer.weight.to(device="cuda")
# print(f"{layer.weight.device=}")


# import torch
# import torch.nn as nn
# from typing import Optional, Union
# class RMSNorm(nn.Module):
#     """Root mean square normalization.

#     Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
#     Refer to https://arxiv.org/abs/1910.07467
#     """

#     def __init__(
#         self,
#         hidden_size: int,
#         eps: float = 1e-6,
#         var_hidden_size: Optional[int] = None,
#         has_weight: bool = True,
#         dtype: Optional[torch.dtype] = None,
#     ) -> None:
#         super().__init__()

#         self.hidden_size = hidden_size
#         self.variance_epsilon = eps
#         self.variance_size_override = (None if var_hidden_size == hidden_size
#                                        else var_hidden_size)
#         self.has_weight = has_weight
#         if dtype is not None:
#             self.weight = torch.ones(hidden_size, dtype=dtype)
#         else:
#             self.weight = torch.ones(hidden_size)
#         if self.has_weight:
#             self.weight = nn.Parameter(self.weight)


#     def forward(
#         self,
#         x: torch.Tensor,
#         residual: Optional[torch.Tensor] = None,
#     ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
#         """PyTorch-native implementation equivalent to forward()."""
#         orig_dtype = x.dtype
#         x = x.to(torch.float32)
#         if residual is not None:
#             x = x + residual.to(torch.float32)
#             residual = x.to(orig_dtype)

#         hidden_size = x.shape[-1]
#         if hidden_size != self.hidden_size:
#             raise ValueError("Expected hidden_size to be "
#                              f"{self.hidden_size}, but found: {hidden_size}")

#         if self.variance_size_override is None:
#             x_var = x
#         else:
#             if hidden_size < self.variance_size_override:
#                 raise ValueError(
#                     "Expected hidden_size to be at least "
#                     f"{self.variance_size_override}, but found: {hidden_size}")

#             x_var = x[:, :, :self.variance_size_override]

#         variance = x_var.pow(2).mean(dim=-1, keepdim=True)

#         x = x * torch.rsqrt(variance + self.variance_epsilon)
#         x = x.to(orig_dtype)
#         if self.has_weight:
#             print(f"Cascade RMSNorm: {self.weight.size()=}, {x.size()=}")
#             x = x * self.weight
#         if residual is None:
#             return x
#         else:
#             return x, residual



# # create instance of RMSNorm
# hidden_size = 128
# dtype = torch.float16
# rmsnorm = RMSNorm(hidden_size, dtype=dtype).to(device="cuda")
# rmsnorm.weight.data.normal_(mean=1.0, std=0.1)

# # call forward method
# x = torch.randn(12, hidden_size, dtype=dtype, device="cuda")
# residual = torch.randn(12, hidden_size, dtype=torch.float16, device="cuda") / 2

# output = rmsnorm(x, residual=residual)
# print(f"Output shape: {output.shape if isinstance(output, torch.Tensor) else output[0].shape}")