# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


class _SequenceParallelPatternHelper:
    """Base helper for sequence parallelism patterns."""

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str):
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.tp_group = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_reduce(x)

    def _reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.reduce_scatter.default(
            x,
            dim=0,
            world_size=self.tp_size,
            group_name=self.tp_group.unique_name)

    def _all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.all_gather.default(
            x,
            dim=0,
            world_size=self.tp_size,
            group_name=self.tp_group.unique_name)


class _RMSNormOpHelper(_SequenceParallelPatternHelper):
    """Helper for RMSNorm operations in sequence parallelism patterns."""

    def _functional_rmsnorm(self, result_buffer, input_tensor, weight_tensor):
        return torch.ops.higher_order.auto_functionalized(
            torch.ops._C.rms_norm.default,
            result=result_buffer,
            input=input_tensor,
            weight=weight_tensor,
            epsilon=self.epsilon)

    def _functional_fused_add_rmsnorm(self, input_tensor, residual_tensor,
                                      weight_tensor):
        return torch.ops.higher_order.auto_functionalized(
            torch.ops._C.fused_add_rms_norm.default,
            input=input_tensor,
            residual=residual_tensor,
            weight=weight_tensor,
            epsilon=self.epsilon)


class _RMSNormQuantOpHelper(_SequenceParallelPatternHelper):
    """Helper for RMSNorm + Quantization operations in sequence parallelism patterns."""  # noqa: E501

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str,
                 quant_op: torch._ops.OpOverload):
        super().__init__(epsilon, dtype, device)
        self.quant_op = quant_op

    def _functional_rmsnorm_then_quant(self, rmsnorm_result_buffer,
                                       quant_result_buffer, input_tensor,
                                       weight_tensor, scale_tensor):
        rmsnorm_out_tuple = torch.ops.higher_order.auto_functionalized(
            torch.ops._C.rms_norm.default,
            result=rmsnorm_result_buffer,
            input=input_tensor,
            weight=weight_tensor,
            epsilon=self.epsilon)
        quant_out_tuple = torch.ops.higher_order.auto_functionalized(
            self.quant_op,
            result=quant_result_buffer,
            input=rmsnorm_out_tuple[1],
            scale=scale_tensor)
        return quant_out_tuple

    def _functional_fused_add_rmsnorm_then_quant(self, quant_result_buffer,
                                                 input_tensor, residual_tensor,
                                                 weight_tensor, scale_tensor):
        fused_add_rmsnorm_out_tuple = torch.ops.higher_order.auto_functionalized(  # noqa: E501
            torch.ops._C.fused_add_rms_norm.default,
            input=input_tensor,
            residual=residual_tensor,
            weight=weight_tensor,
            epsilon=self.epsilon)
        quant_out_tuple = torch.ops.higher_order.auto_functionalized(
            self.quant_op,
            result=quant_result_buffer,
            input=fused_add_rmsnorm_out_tuple[1],
            scale=scale_tensor)
        return quant_out_tuple, fused_add_rmsnorm_out_tuple[2]


class _FusedRMSNormQuantOpHelper(_SequenceParallelPatternHelper):
    """Helper for Fused RMSNorm + Quantization operations in sequence parallelism patterns."""  # noqa: E501

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str,
                 fused_rmsnorm_quant_op: torch._ops.OpOverload,
                 fused_add_rmsnorm_quant_op: torch._ops.OpOverload):
        super().__init__(epsilon, dtype, device)
        self.fused_rmsnorm_quant_op = fused_rmsnorm_quant_op
        self.fused_add_rmsnorm_quant_op = fused_add_rmsnorm_quant_op

    def _functional_fused_rmsnorm_quant(self, result_buffer, input_tensor,
                                        weight_tensor, scale_tensor):
        return torch.ops.higher_order.auto_functionalized(
            self.fused_rmsnorm_quant_op,
            result=result_buffer,
            input=input_tensor,
            weight=weight_tensor,
            scale=scale_tensor,
            epsilon=self.epsilon)

    def _functional_fused_add_rmsnorm_quant(self, result_buffer, input_tensor,
                                            residual_tensor, weight_tensor,
                                            scale_tensor):
        return torch.ops.higher_order.auto_functionalized(
            self.fused_add_rmsnorm_quant_op,
            result=result_buffer,
            input=input_tensor,
            residual=residual_tensor,
            weight=weight_tensor,
            scale=scale_tensor,
            epsilon=self.epsilon)


class EmbeddingAllReduceRMSNormPattern(_RMSNormOpHelper):

    def get_inputs(self):
        arg2_1 = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        mul_6 = torch.tensor([[3, 7, 1, 4, 9, 2, 5, 0]],
                             device=self.device,
                             dtype=torch.long)
        unsqueeze = torch.rand([1, 8, 1], device=self.device,
                               dtype=self.dtype) > 0.5
        full_default = torch.zeros([1, 8, 4],
                                   device=self.device,
                                   dtype=self.dtype)
        permute = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        arg3_1 = torch.empty([4], device=self.device, dtype=self.dtype)

        return [arg2_1, mul_6, unsqueeze, full_default, permute, arg3_1]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            arg2_1: torch.Tensor,
            mul_6: torch.Tensor,
            unsqueeze: torch.Tensor,
            full_default: torch.Tensor,
            permute: torch.Tensor,
            arg3_1: torch.Tensor,
        ):
            embedding = torch.ops.aten.embedding.default(arg2_1, mul_6)
            where = torch.ops.aten.where.self(unsqueeze, full_default,
                                              embedding)
            all_reduce = self._all_reduce(where)
            rmsnorm = self._functional_rmsnorm(permute, all_reduce, arg3_1)

            return rmsnorm[1], all_reduce

        def replacement(
            arg2_1: torch.Tensor,
            mul_6: torch.Tensor,
            unsqueeze: torch.Tensor,
            full_default: torch.Tensor,
            permute: torch.Tensor,
            arg3_1: torch.Tensor,
        ):
            embedding = torch.ops.aten.embedding.default(arg2_1, mul_6)
            where = torch.ops.aten.where.self(unsqueeze, full_default,
                                              embedding)
            reduce_scatter = self._reduce_scatter(where)

            rmsnorm_result = torch.empty_like(reduce_scatter)
            rmsnorm = self._functional_rmsnorm(rmsnorm_result, reduce_scatter,
                                               arg3_1)

            all_gather = self._all_gather(rmsnorm[1])

            return all_gather, reduce_scatter

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class MiddleAllReduceRMSNormPattern(_RMSNormOpHelper):

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)

        return [
            residual,
            mm_1,
            rms_norm_weights,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                all_reduce, residual, rms_norm_weights)
            return rmsnorm[1], rmsnorm[2]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                reduce_scatter, residual, rms_norm_weights)
            all_gather = self._all_gather(rmsnorm[1])
            return all_gather, rmsnorm[2]

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class LastAllReduceRMSNormPattern(_RMSNormOpHelper):

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)

        return [
            residual,
            mm_1,
            rms_norm_weights,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                all_reduce, residual, rms_norm_weights)
            return rmsnorm[1]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                reduce_scatter, residual, rms_norm_weights)
            normalized = self._all_gather(rmsnorm[1])
            return normalized

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


FP8_DTYPE = current_platform.fp8_dtype()


class EmbeddingAllReduceFusedRMSNormStaticFP8Pattern(_FusedRMSNormQuantOpHelper
                                                     ):

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str):
        super().__init__(epsilon,
                         dtype,
                         device,
                         fused_rmsnorm_quant_op=torch.ops._C.
                         rms_norm_static_fp8_quant.default,
                         fused_add_rmsnorm_quant_op=torch.ops._C.
                         fused_add_rms_norm_static_fp8_quant.default)

    def get_inputs(self):
        arg2_1 = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        mul_6 = torch.tensor([[3, 7, 1, 4, 9, 2, 5, 0]],
                             device=self.device,
                             dtype=torch.long)
        unsqueeze = torch.rand([1, 8, 1], device=self.device,
                               dtype=self.dtype) > 0.5
        full_default = torch.zeros([1, 8, 4],
                                   device=self.device,
                                   dtype=self.dtype)
        result = torch.empty([1, 8, 4], device=self.device, dtype=FP8_DTYPE)
        weight = torch.empty([4], device=self.device, dtype=self.dtype)
        scale = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        return [arg2_1, mul_6, unsqueeze, full_default, result, weight, scale]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            arg2_1: torch.Tensor,
            mul_6: torch.Tensor,
            unsqueeze: torch.Tensor,
            full_default: torch.Tensor,
            result: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            embedding = torch.ops.aten.embedding.default(arg2_1, mul_6)
            where = torch.ops.aten.where.self(unsqueeze, full_default,
                                              embedding)
            all_reduce = self._all_reduce(where)
            rmsnorm = self._functional_fused_rmsnorm_quant(
                result, all_reduce, weight, scale)
            return rmsnorm[1], all_reduce

        def replacement(
            arg2_1: torch.Tensor,
            mul_6: torch.Tensor,
            unsqueeze: torch.Tensor,
            full_default: torch.Tensor,
            result: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            embedding = torch.ops.aten.embedding.default(arg2_1, mul_6)
            where = torch.ops.aten.where.self(unsqueeze, full_default,
                                              embedding)
            reduce_scatter = self._reduce_scatter(where)

            rmsnorm_result = torch.empty_like(reduce_scatter,
                                              dtype=result.dtype)
            rmsnorm = self._functional_fused_rmsnorm_quant(
                rmsnorm_result, reduce_scatter, weight, scale)
            all_gather = self._all_gather(rmsnorm[1])

            return all_gather, reduce_scatter

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class MiddleAllReduceFusedRMSNormStaticFP8Pattern(_FusedRMSNormQuantOpHelper):

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str):
        super().__init__(epsilon,
                         dtype,
                         device,
                         fused_rmsnorm_quant_op=torch.ops._C.
                         rms_norm_static_fp8_quant.default,
                         fused_add_rmsnorm_quant_op=torch.ops._C.
                         fused_add_rms_norm_static_fp8_quant.default)

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)
        result = torch.empty([4, 4], device=self.device, dtype=FP8_DTYPE)
        scale = torch.empty([1, 1], device=self.device, dtype=torch.float32)

        return [
            result,
            residual,
            mm_1,
            rms_norm_weights,
            scale,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm_quant(
                result, all_reduce, residual, rms_norm_weights, scale)
            return rmsnorm[1], rmsnorm[2]

        def replacement(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            rs_result = torch.empty_like(reduce_scatter, dtype=result.dtype)
            rmsnorm = self._functional_fused_add_rmsnorm_quant(
                rs_result, reduce_scatter, residual, rms_norm_weights, scale)
            all_gather = self._all_gather(rmsnorm[1])
            return all_gather, rmsnorm[2]

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class LastAllReduceFusedRMSNormStaticFP8Pattern(_FusedRMSNormQuantOpHelper):

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str):
        super().__init__(epsilon,
                         dtype,
                         device,
                         fused_rmsnorm_quant_op=torch.ops._C.
                         rms_norm_static_fp8_quant.default,
                         fused_add_rmsnorm_quant_op=torch.ops._C.
                         fused_add_rms_norm_static_fp8_quant.default)

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)
        result = torch.empty([4, 4], device=self.device, dtype=FP8_DTYPE)
        scale = torch.empty([1, 1], device=self.device, dtype=torch.float32)

        return [
            result,
            residual,
            mm_1,
            rms_norm_weights,
            scale,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm_quant(
                result, all_reduce, residual, rms_norm_weights, scale)
            return rmsnorm[1]

        def replacement(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            rs_result = torch.empty_like(reduce_scatter, dtype=result.dtype)
            rmsnorm = self._functional_fused_add_rmsnorm_quant(
                rs_result, reduce_scatter, residual, rms_norm_weights, scale)
            normalized = self._all_gather(rmsnorm[1])
            return normalized

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class EmbeddingAllReduceRMSNormStaticFP8Pattern(_RMSNormQuantOpHelper):

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str,
                 op: torch._ops.OpOverload):
        super().__init__(epsilon, dtype, device, quant_op=op)

    def get_inputs(self):
        arg2_1 = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        mul_6 = torch.tensor([[3, 7, 1, 4, 9, 2, 5, 0]],
                             device=self.device,
                             dtype=torch.long)
        unsqueeze = torch.rand([1, 8, 1], device=self.device,
                               dtype=self.dtype) > 0.5
        full_default = torch.zeros([1, 8, 4],
                                   device=self.device,
                                   dtype=self.dtype)
        rmsnorm_result = torch.empty([1, 8, 4],
                                     device=self.device,
                                     dtype=self.dtype)
        quant_result = torch.empty([1, 8, 4],
                                   device=self.device,
                                   dtype=FP8_DTYPE)
        weight = torch.empty([4], device=self.device, dtype=self.dtype)
        scale = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        return [
            arg2_1, mul_6, unsqueeze, full_default, rmsnorm_result,
            quant_result, weight, scale
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            arg2_1: torch.Tensor,
            mul_6: torch.Tensor,
            unsqueeze: torch.Tensor,
            full_default: torch.Tensor,
            rmsnorm_result: torch.Tensor,
            quant_result: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            embedding = torch.ops.aten.embedding.default(arg2_1, mul_6)
            where = torch.ops.aten.where.self(unsqueeze, full_default,
                                              embedding)
            all_reduce = self._all_reduce(where)
            static_fp8 = self._functional_rmsnorm_then_quant(
                rmsnorm_result, quant_result, all_reduce, weight, scale)
            return static_fp8[1], all_reduce

        def replacement(
            arg2_1: torch.Tensor,
            mul_6: torch.Tensor,
            unsqueeze: torch.Tensor,
            full_default: torch.Tensor,
            rmsnorm_result: torch.Tensor,
            quant_result: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            embedding = torch.ops.aten.embedding.default(arg2_1, mul_6)
            where = torch.ops.aten.where.self(unsqueeze, full_default,
                                              embedding)
            reduce_scatter = self._reduce_scatter(where)

            rmsnorm_result = torch.empty_like(reduce_scatter,
                                              dtype=rmsnorm_result.dtype)
            quant_result = torch.empty_like(
                rmsnorm_result,  # Output of RMSNorm
                dtype=quant_result.dtype)
            static_fp8 = self._functional_rmsnorm_then_quant(
                rmsnorm_result, quant_result, reduce_scatter, weight, scale)
            all_gather = self._all_gather(static_fp8[1])

            return all_gather, reduce_scatter

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class MiddleAllReduceRMSNormStaticFP8Pattern(_RMSNormQuantOpHelper):

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str,
                 op: torch._ops.OpOverload):
        super().__init__(epsilon, dtype, device, quant_op=op)

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)
        result = torch.empty([4, 4], device=self.device, dtype=FP8_DTYPE)
        scale = torch.empty([1, 1], device=self.device, dtype=torch.float32)

        return [
            result,
            residual,
            mm_1,
            rms_norm_weights,
            scale,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            static_fp8, rmsnorm_residual_out = self._functional_fused_add_rmsnorm_then_quant(  # noqa: E501
                result, all_reduce, residual, rms_norm_weights, scale)
            return static_fp8[1], rmsnorm_residual_out

        def replacement(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            quant_result_buf = torch.empty_like(reduce_scatter,
                                                dtype=result.dtype)
            static_fp8, rmsnorm_residual_out = self._functional_fused_add_rmsnorm_then_quant(  # noqa: E501
                quant_result_buf, reduce_scatter, residual, rms_norm_weights,
                scale)
            all_gather = self._all_gather(static_fp8[1])
            return all_gather, rmsnorm_residual_out

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class LastAllReduceRMSNormStaticFP8Pattern(_RMSNormQuantOpHelper):

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str,
                 op: torch._ops.OpOverload):
        super().__init__(epsilon, dtype, device, quant_op=op)

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)
        result = torch.empty([4, 4], device=self.device, dtype=FP8_DTYPE)
        scale = torch.empty([1, 1], device=self.device, dtype=torch.float32)

        return [
            result,
            residual,
            mm_1,
            rms_norm_weights,
            scale,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            static_fp8, _ = self._functional_fused_add_rmsnorm_then_quant(
                result, all_reduce, residual, rms_norm_weights, scale)
            return static_fp8[1]

        def replacement(
            result: torch.Tensor,
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            quant_result_buf = torch.empty_like(reduce_scatter,
                                                dtype=result.dtype)
            static_fp8, _ = self._functional_fused_add_rmsnorm_then_quant(
                quant_result_buf, reduce_scatter, residual, rms_norm_weights,
                scale)
            normalized = self._all_gather(static_fp8[1])
            return normalized

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class SequenceParallelismPass(VllmInductorPass):

    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="sequence_parallelism_pass")

        for epsilon in [1e-5, 1e-6]:
            # RMSNorm + Static FP8 quantization patterns
            fp8_quant_op = torch.ops._C.static_scaled_fp8_quant.default
            EmbeddingAllReduceRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device,
                fp8_quant_op).register(self.patterns)
            MiddleAllReduceRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device,
                fp8_quant_op).register(self.patterns)
            LastAllReduceRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device,
                fp8_quant_op).register(self.patterns)

            # Fused RMSNorm + Static FP8 patterns
            EmbeddingAllReduceFusedRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device).register(self.patterns)

            MiddleAllReduceFusedRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device).register(self.patterns)

            LastAllReduceFusedRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device).register(self.patterns)

            # Normal RMSNorm patterns
            EmbeddingAllReduceRMSNormPattern(
                epsilon, self.model_dtype, self.device).register(self.patterns)

            MiddleAllReduceRMSNormPattern(epsilon, self.model_dtype,
                                          self.device).register(self.patterns)

            LastAllReduceRMSNormPattern(epsilon, self.model_dtype,
                                        self.device).register(self.patterns)

            # WARNING: This is a hack to clear the pattern matcher cache
            # and allow multiple values of epsilon.
            torch._inductor.pattern_matcher._seen_patterns.clear()

    def is_applicable_for_shape(self, shape: Optional[int]) -> bool:
        tp_size = get_tensor_model_parallel_world_size()
        return shape is not None and shape % tp_size == 0

    def __call__(self, graph: fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_sequence_parallelism_pass")
        count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", count)
        self.dump_graph(graph, "after_sequence_parallelism_pass")
        self.end_and_log()
