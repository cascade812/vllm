# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.logger import init_logger

from .vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


class BasePattern:

    def __init__(self, dtype: torch.dtype, device: str):
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()


class GEMMReduceScatterPattern(BasePattern):

    def get_inputs(self):
        mul = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        mm_weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [mul, mm_weight]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(mul: torch.Tensor, mm_weight: torch.Tensor):
            mm = torch.ops.aten.mm.default(mul, mm_weight)
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name)
            return reduce_scatter

        def replacement(mul: torch.Tensor, mm_weight: torch.Tensor):
            gemm_rs = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                mul,
                mm_weight,
                "avg",
                scatter_dim=0,
                group_name=self.tp.device_group.group_name,
            )

            return gemm_rs

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class AllGatherGEMMPattern(BasePattern):

    def get_inputs(self):
        x = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        return [x, weight]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name)

            return torch.ops.aten.mm.default(all_gather, weight)

        def replacement(
                x: torch.Tensor,
                weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
                x,
                [weight],
                gather_dim=0,
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class ScaledMMReduceScatterPattern(BasePattern):

    def get_inputs(self):
        input = torch.empty([16, 16], device=self.device, dtype=self.dtype)
        mm_weight = torch.empty([16, 16], device=self.device, dtype=self.dtype)
        scaled_a = torch.empty([16, 1],
                               device=self.device,
                               dtype=torch.float32)
        scaled_b = torch.empty([1, 16],
                               device=self.device,
                               dtype=torch.float32)
        return [input, mm_weight, scaled_a, scaled_b]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(input: torch.Tensor, mat2: torch.Tensor,
                    scaled_a: torch.Tensor, scaled_b: torch.Tensor):
            scaled_mm = torch.ops.aten._scaled_mm.default(input,
                                                          mat2=mat2,
                                                          scale_a=scaled_a,
                                                          scale_b=scaled_b,
                                                          bias=None,
                                                          scale_result=None,
                                                          out_dtype=self.dtype)
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                scaled_mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name)
            return reduce_scatter

        def replacement(input: torch.Tensor, mat2: torch.Tensor,
                        scaled_a: torch.Tensor, scaled_b: torch.Tensor):
            gemm_rs = torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scaled_a,
                scaled_b,
                "avg",
                scatter_dim=0,
                group_name=self.tp.device_group.group_name,
            )

            return gemm_rs

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


# input: Tensor, mat2: Tensor, scale_a: Tensor, scale_b: Tensor, bias: Optional[Tensor] = None, scale_result: Optional[Tensor] = None, out_dtype: Optional[_dtype] = None, use_fast_accum: _bool = False, *, out: Optional[Tensor] = None)
# "fused_scaled_matmul_reduce_scatter("
#     "Tensor A, Tensor B, Tensor A_scale, Tensor B_scale, "
#     "str reduce_op, int scatter_dim, str group_name, "
#     "Tensor? bias = None, "
#     "Tensor? result_scale = None, "
#     "ScalarType? out_dtype = None, "
#     "bool use_fast_accum = False) -> Tensor",

# %_scaled_mm : [num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%getitem_1, %arg3_1, %arg2_1, %arg4_1, None, None, torch.bfloat16), kwargs = {})
# %empty_1 : [num_users=0] = call_function[target=torch.ops.aten.empty.memory_format](args = ([4, 4096],), kwargs = {dtype: torch.float8_e4m3fn, device: cuda:0, pin_memory: False})
# %reduce_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.vllm.reduce_scatter.default](args = (%_scaled_mm, 0, 2, tp:0), kwargs = {})


class AllGatherScaledMMPattern(BasePattern):

    def get_inputs(self):
        x = torch.empty([16, 16], device=self.device, dtype=self.dtype)
        weight = torch.empty([16, 16], device=self.device, dtype=self.dtype)
        scaled_a = torch.empty([16, 1],
                               device=self.device,
                               dtype=torch.float32)
        scaled_b = torch.empty([1, 16],
                               device=self.device,
                               dtype=torch.float32)

        return [x, weight, scaled_a, scaled_b]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scaled_a: torch.Tensor,
            scaled_b: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name)

            return torch.ops.aten._scaled_mm.default(input=all_gather,
                                                     mat2=weight,
                                                     scale_a=scaled_a,
                                                     scale_b=scaled_b,
                                                     bias=None,
                                                     scale_result=None,
                                                     out_dtype=self.dtype)

        def replacement(
                x: torch.Tensor, weight: torch.Tensor, scaled_a: torch.Tensor,
                scaled_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
                x,
                [weight],
                scaled_a,
                [scaled_b],
                gather_dim=0,
                biases=None,
                result_scales=None,
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


# lib.define(
#     "fused_all_gather_scaled_matmul("
#     "Tensor A, Tensor[] Bs, Tensor A_scale, Tensor[] B_scales, "
#     "int gather_dim, str group_name, "
#     "Tensor?[] biases, "
#     "Tensor?[] result_scales, "
#     "ScalarType?[] out_dtypes, "
#     "bool[] use_fast_accum) -> (Tensor, Tensor[])",

# %all_gather_default_1 : [num_users=1] = call_function[target=torch.ops.vllm.all_gather.default](args = (%getitem_29, 0, 2, tp:0), kwargs = {})
# %empty_2 : [num_users=1] = call_function[target=torch.ops.aten.empty.memory_format](args = ([4, 7168],), kwargs = {dtype: torch.float8_e4m3fn, device: cuda:0, pin_memory: False})
# %_scaled_mm_1 : [num_users=2] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%all_gather_default_1, %arg8_1, %arg7_1, %arg9_1, None, None, torch.bfloat16), kwargs = {})


class AsyncTPPass(VllmInductorPass):

    def __init__(self, config: VllmConfig):
        super().__init__(config)

        # Enable symmetric memory for the TP process group
        enable_symm_mem_for_group(get_tp_group().device_group.group_name)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="async_tp_pass")
        GEMMReduceScatterPattern(self.model_dtype,
                                 self.device).register(self.patterns)

        AllGatherGEMMPattern(self.model_dtype,
                             self.device).register(self.patterns)

        ScaledMMReduceScatterPattern(self.model_dtype,
                                     self.device).register(self.patterns)
        AllGatherScaledMMPattern(self.model_dtype,
                                 self.device).register(self.patterns)

    def is_applicable_for_shape(self, shape: Optional[int]) -> bool:
        # only do replace for specific shapes
        tp_size = get_tensor_model_parallel_world_size()
        return shape is not None and shape % tp_size == 0

    def __call__(self, graph: fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_async_tp_pass")
        if get_tp_group().rank == 0:
            print(f"before graph {graph}")
        count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", count)
        self.dump_graph(graph, "after_async_tp_pass")
        if get_tp_group().rank == 0:
            print(f"after collective graph {graph}")
        self.end_and_log()
