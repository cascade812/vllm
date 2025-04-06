from vllm import LLM, SamplingParams
# unsloth/Llama-3.2-1B-Instruct
from vllm.config import CompilationConfig, CompilationLevel

import torch

config = CompilationConfig(
    level=3,
    custom_ops = ["+rms_norm"],
    splitting_ops = [],
)
# config.pass_config.enable_collective_fusion = True

llm = LLM(model="unsloth/Llama-3.2-1B-Instruct",
          enforce_eager=False,
          tensor_parallel_size=2,
          disable_custom_all_reduce=False,
          dtype=torch.float16,
          max_num_batched_tokens=2048,
          compilation_config=config)






output = llm.generate(prompts=["What's the result of 9+7? Answer directly.", 
                               "What's the medium height of asian grown men? Answer directly.", 
                               "How old a baby can start to try solid food?", 
                               "What's pros and cons of using a pacifier?"], use_tqdm=True)


print(output)


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

# # NOTE(woosuk): The reference implementation should be executed first
# # because the custom kernel is in-place.
# ref_out = layer.forward_native(x1, residual1)
# out = layer(x2, residual2)

# print(f"{ref_out=}")
# print(f"{out=}")


# # norm_res = torch.ops._C.fused_add_rms_norm(input=x,
# #                                             residual=residual,
# #                                             weight=layer.weight,
# #                                             epsilon=1e-05)

# norm_res = torch.ops.higher_order.auto_functionalized(
#         torch.ops._C.fused_add_rms_norm.default,
#         input=x,
#         residual=residual,
#         weight=layer.weight,
#         epsilon=1e-5)

# print(f"norm_res = {norm_res}")
# import torch
