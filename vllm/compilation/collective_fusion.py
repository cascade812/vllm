# SPDX-License-Identifier: Apache-2.0
import contextlib
import operator
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.fx as fx
from torch._inductor.pattern_matcher import (Match, PatternMatcherPass,
                                             fwd_only, register_replacement)

from vllm.compilation.fx_utils import find_auto_fn, find_getitem
from vllm.config import CompilationConfig
from vllm.distributed import (get_group_from_group_name, get_tp_group,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.logger import init_logger
from vllm.utils import direct_register_custom_op

# from .inductor_pass import get_pass_context
from .vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)

def get_world_name() -> str:
    return torch.distributed.group.WORLD.group_name


def residual_slice_shape(residual: torch.Tensor) -> int:
    n_slices = get_tensor_model_parallel_world_size()
    assert residual.size(0) % n_slices == 0
    return residual.size(0) // n_slices


def search_embedding_all_reduce_rmsnorm(
    arg2_1: torch.Tensor,
    mul_6: torch.Tensor,
    unsqueeze: torch.Tensor,
    full_default: torch.Tensor,
    permute: torch.Tensor,
    arg3_1: torch.Tensor,
):
    embedding = torch.ops.aten.embedding.default(arg2_1, mul_6)
    where = torch.ops.aten.where.self(unsqueeze, full_default, embedding)
    all_reduce = tensor_model_parallel_all_reduce(where)
    rmsnorm = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.rms_norm.default,
        result=permute,
        input=all_reduce,
        weight=arg3_1,
        epsilon=1e-5)
    
    return rmsnorm[1], all_reduce
        

def replace_with_embedding_reduce_scatter_rmsnorm(
    arg2_1: torch.Tensor,
    mul_6: torch.Tensor,
    unsqueeze: torch.Tensor,
    full_default: torch.Tensor,
    permute: torch.Tensor,
    arg3_1: torch.Tensor,
):
    embedding = torch.ops.aten.embedding.default(arg2_1, mul_6)
    where = torch.ops.aten.where.self(unsqueeze, full_default, embedding)
    
    tp = get_tp_group()
    tp_size = get_tensor_model_parallel_world_size()
    reduce_scatter = torch.ops.vllm.reduce_scatter.default(
            where, dim=0, world_size=tp_size, group_name=tp.unique_name)
    
    # rmsnorm_result = torch.empty_like(reduce_scatter)
    rmsnorm = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.rms_norm.default,
        result=permute,
        input=reduce_scatter,
        weight=arg3_1,
        epsilon=1e-5)
    
    
    all_gather = torch.ops.vllm.all_gather.default(
            reduce_scatter, dim=0, world_size=tp_size, group_name=tp.unique_name)
    
    print(f"first rms_norm output: {rmsnorm[1].shape}, reduce_scatter = {reduce_scatter.shape}, all_gather={all_gather.shape}" )
    return rmsnorm[1], all_gather

def search_gemm_allreduce_rmsnorm(
    residual: torch.Tensor,
    gemm_1_weights: torch.Tensor,
    gemm_1_activations: torch.Tensor,
    rms_norm_weights: torch.Tensor,
    # gemm_2_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)
    all_reduce = tensor_model_parallel_all_reduce(mm_1)

    rmsnorm = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input=all_reduce,
        residual=residual,
        weight=rms_norm_weights,
        epsilon=1e-5)

    normalized = rmsnorm[1]
    new_residual = rmsnorm[2]

    # gemm_2_w_perm = torch.ops.aten.permute.default(gemm_2_weights, [1, 0])
    # mm_2 = torch.ops.aten.mm.default(normalized, gemm_2_w_perm)

    # return mm_2, new_residual
    return rmsnorm[1], new_residual

def replace_with_gemm_rs_ag_rmsnorm(
    residual: torch.Tensor,
    gemm_1_weights: torch.Tensor,
    gemm_1_activations: torch.Tensor,
    rms_norm_weights: torch.Tensor,
    # gemm_2_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tp = get_tp_group()
    tp_size = get_tensor_model_parallel_world_size()
    slice_shape = residual_slice_shape(residual)
    
    start_idx = tp.rank * slice_shape
    end_idx = (tp.rank + 1) * slice_shape
    residual = residual[start_idx:end_idx, :]
    
    # residual_chunk = torch.ops.aten.split.Tensor(residual, slice_shape)
    # residual = residual_chunk[tp.rank_in_group]
    
    gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)
    reduce_scatter = torch.ops.vllm.reduce_scatter.default(
        mm_1, dim=0, world_size=tp_size, group_name=tp.unique_name)

    # TODO is it possible to extract epsilon from somewhere
    rmsnorm = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input=reduce_scatter,
        residual=residual,
        weight=rms_norm_weights,
        epsilon=1e-5)
    
    print("residual.shape: ", residual.shape)
    print("reduce_scatter.shape: ", reduce_scatter.shape)
    print("rmsnorm[1].shape: ", rmsnorm[1].shape)
    print("rmsnorm[2].shape: ", rmsnorm[2].shape)
    
    # rank 0 and rank 1
    # residual.shape:  torch.Size([(s0//2), 2048])
    # reduce_scatter.shape:  torch.Size([(s0//2), 2048])
    # rmsnorm[1].shape:  torch.Size([(s0//2), 2048])
    # rmsnorm[2].shape:  torch.Size([(s0//2), 2048])
    
    normalized = torch.ops.vllm.all_gather.default(
        rmsnorm[1], dim=0, world_size=tp_size, group_name=tp.unique_name)
    
    new_residual = rmsnorm[2]
    new_residual = new_residual.repeat(tp_size, 1)
    
    
    return normalized, new_residual


def prepare_inputs_for_embedding_rmsnorm():
    # Parameters as specified
    vocab_size = 16
    hidden_size = 4
    seq_len = 8
    batch_size = 1

    # arg2_1: embedding table (vocab_size x hidden_size)
    arg2_1 = torch.empty([vocab_size, hidden_size], device='cuda', dtype=torch.float16)
    arg2_1.normal_(0, 0.02)

    # mul_6: token indices (batch_size x seq_len)
    mul_6 = torch.tensor([[3, 7, 1, 4, 9, 2, 5, 0]], device='cuda', dtype=torch.long)

    # The issue is with the unsqueeze tensor's shape
    # It should be broadcasting-compatible with the embedding output shape
    # Embedding will have shape [batch_size, seq_len, hidden_size] = [1, 8, 4]
    # So unsqueeze should have shape [1, 8, 1] or [1, 8, 4] to broadcast properly

    # unsqueeze: attention mask (batch_size x seq_len x 1)
    # This will broadcast correctly with the embedding output
    unsqueeze = torch.ones([batch_size, seq_len, 1], device='cuda', dtype=torch.bool)

    # full_default: zeros tensor with embedding output shape (batch_size x seq_len x hidden_size)
    full_default = torch.zeros([batch_size, seq_len, hidden_size], device='cuda', dtype=torch.float16)

    # permute: residual connection tensor (batch_size x seq_len x hidden_size)
    permute = torch.empty([batch_size, seq_len, hidden_size], device='cuda', dtype=torch.float16)
    permute.normal_(0, 0.1)

    # arg3_1: RMSNorm weights (hidden_size)
    arg3_1 = torch.ones([hidden_size], device='cuda', dtype=torch.float16)
        
        # arg2_1: torch.Tensor,
        # mul_6: torch.Tensor,
        # unsqueeze: torch.Tensor,
        # full_default: torch.Tensor,
        # permute: torch.Tensor,
        # arg3_1: torch.Tensor,
    
    return [arg2_1, mul_6, unsqueeze, full_default, permute, arg3_1]


class CollectiveFusionPass(VllmInductorPass):

    _instance: 'Optional[CollectiveFusionPass]' = None

    @classmethod
    def instance(cls, config: CompilationConfig) -> "CollectiveFusionPass":
        """
        Get the singleton instance of the CollectiveFusionPass.
        If the instance exists, the config is updated but
        initialization is not repeated.
        """
        if cls._instance is None:
            cls._instance = CollectiveFusionPass(config)
        else:
            cls._instance.config = config
        return cls._instance

    def __init__(self, config: CompilationConfig):
        assert self.__class__._instance is None, \
            "CollectiveFusionPass singleton instance already exists"
        super().__init__(config)

        # self.embedding_ag_rmsnorm_pattern = PatternMatcherPass()
        self.gemm_rs_ag_gemm_pattern = PatternMatcherPass()
        self.final_ar_rmsnorm_pattern = PatternMatcherPass()
        self.matches: List[Match] = []
       
        # inputs_for_embedding_rmsnorm = prepare_inputs_for_embedding_rmsnorm()
        # register_replacement(search_embedding_all_reduce_rmsnorm,
        #                      replace_with_embedding_reduce_scatter_rmsnorm,
        #                      inputs_for_embedding_rmsnorm,
        #                      fwd_only, [self.embedding_ag_rmsnorm_pattern],
        #                      extra_check=lambda m: self.record_match(m))
        

        x = torch.empty([4, 4], device='cuda', dtype=torch.float16)
        w = torch.empty([4, 4], device='cuda', dtype=torch.float16)
        resid = torch.empty([4, 4], device='cuda', dtype=torch.float16)
        resid_w = torch.empty([4, 4], device='cuda', dtype=torch.float16)
        
        inputs = [resid, x, w, resid_w]
        register_replacement(search_gemm_allreduce_rmsnorm,
                             replace_with_gemm_rs_ag_rmsnorm,
                             inputs,
                             fwd_only, [self.gemm_rs_ag_gemm_pattern],
                             extra_check=lambda m: self.record_match(m))

    
    def record_match(self, match: Match) -> bool:
        # record the match for possibile later use
        self.matches.append(match)
        return bool(match)


    def __call__(self, graph: fx.Graph):
        import torch.distributed as dist
        rank = dist.get_rank()
        if rank == 0:
            print(f"before graph {graph}")

        self.dump_graph(graph, "before_collective_fusion")
        # embedding_match_cnt = self.embedding_ag_rmsnorm_pattern.apply(graph)
        match_cnt = self.gemm_rs_ag_gemm_pattern.apply(graph)
        logger.info("all match count = %d, embedding_match_cnt = %d, fused gemm match cnt = %d",
                    len(self.matches), 0, match_cnt)

       
        if rank == 0:
            print(f"after graph {graph}")
        self.dump_graph(graph, "after_collective_fusion")
        self.matches.clear()
