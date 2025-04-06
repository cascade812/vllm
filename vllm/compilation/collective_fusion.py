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

use_flux = False

_first_match_used = [False]


# Reset the state before each pass
def reset_conditional_state():
    _first_match_used[0] = False


def get_world_name() -> str:
    return torch.distributed.group.WORLD.group_name


def residual_slice_shape(residual: torch.Tensor, rank: int) -> int:
    n_slices = get_tensor_model_parallel_world_size()
    assert residual.size(0) % n_slices == 0
    return residual.size(0) // n_slices


# Depends on arch, see auto_tile_shape in include/flux/gemm_hparams.h
# Can be 256 on sm80.
FLUX_TILE_SIZE: int = 128


# Heuristic to check if collective communication kernels should be used for a
# particular problem size.
def use_cc_kernels(m_shape: int, n_slices: Optional[int] = None) -> bool:
    if use_flux:
        if n_slices is None:
            n_slices = get_tensor_model_parallel_world_size()
        return (m_shape % (FLUX_TILE_SIZE * n_slices) == 0
                and m_shape >= FLUX_TILE_SIZE * n_slices)
    else:
        # For symmetric memory kernels.  TODO: Is this ok?
        return True


def find_fn(nodes: Iterable[fx.Node], op) -> Optional[fx.Node]:
    for node in nodes:
        if node.op == "call_function" and node.target == op:
            return node
    return None


def find_op(nodes: Iterable[fx.Node], op: str) -> Optional[fx.Node]:
    for node in nodes:
        if node.op == op:
            return node
    return None


def last_node_in_match(match: Match) -> fx.Node:
    if len(match.nodes) > 0:
        graph = match.nodes[0].graph
        for n in reversed(graph.nodes):
            if n in reversed(match.nodes):
                return n
    raise ValueError("No nodes in graph")


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

    # normalized = rmsnorm[1]
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

    print("_first_match_used-", _first_match_used[0])

    # Choose which replacement to use
    if not _first_match_used[0]:
        _first_match_used[0] = True
        gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
        mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)

        tp = get_tp_group()
        tp_size = get_tensor_model_parallel_world_size()
        reduce_scatter = torch.ops.vllm.reduce_scatter.default(
            mm_1, dim=0, world_size=tp_size, group_name=tp.unique_name)

        # TODO is it possible to extract epsilon from somewhere
        rmsnorm = torch.ops.higher_order.auto_functionalized(
            torch.ops._C.fused_add_rms_norm.default,
            input=reduce_scatter,
            residual=residual,
            weight=rms_norm_weights,
            epsilon=1e-5)

        normalized = torch.ops.vllm.all_gather.default(
            rmsnorm[1], dim=0, world_size=tp_size, group_name=tp.unique_name)
        new_residual = rmsnorm[2]
        # gemm_2_w_perm = torch.ops.aten.permute.default(gemm_2_weights, [1, 0])
        # mm_2 = torch.ops.aten.mm.default(normalized, gemm_2_w_perm)

        # return mm_2, new_residual
        return normalized, new_residual
    else:
        gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
        mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)

        tp = get_tp_group()
        tp_size = get_tensor_model_parallel_world_size()
        reduce_scatter_mm_1 = torch.ops.vllm.reduce_scatter.default(
            mm_1, dim=0, world_size=tp_size, group_name=tp.unique_name)

        reduce_scatter_residual = torch.ops.vllm.reduce_scatter.default(
            residual, dim=0, world_size=tp_size, group_name=tp.unique_name)

        # TODO is it possible to extract epsilon from somewhere
        rmsnorm = torch.ops.higher_order.auto_functionalized(
            torch.ops._C.fused_add_rms_norm.default,
            input=reduce_scatter_mm_1,
            residual=reduce_scatter_residual,
            weight=rms_norm_weights,
            epsilon=1e-5)

        normalized = torch.ops.vllm.all_gather.default(
            rmsnorm[1], dim=0, world_size=tp_size, group_name=tp.unique_name)
        new_residual = rmsnorm[2]
        # gemm_2_w_perm = torch.ops.aten.permute.default(gemm_2_weights, [1, 0])
        # mm_2 = torch.ops.aten.mm.default(normalized, gemm_2_w_perm)

        # return mm_2, new_residual
        return normalized, new_residual


def get_gemm_rs_ag_gemm(max_m: int, gemm_1_type: torch.dtype,
                        gemm_1_weights: torch.Size, gemm_2_type: torch.dtype,
                        gemm_2_weights: torch.Size, tp_group_name: str,
                        is_static_shape: bool) -> Tuple[Callable, Callable]:

    group = get_group_from_group_name(tp_group_name)
    # device_group = group.device_group
    rank = group.rank_in_group

    group_str = tp_group_name.replace(":", "_")
    name = f"gemm_rs_ag_gemm_{group_str}"

    if not hasattr(torch.ops.vllm, name):
        world_group_name = get_world_name()

        def gemm_rs(act, wt):
            return torch.ops.symm_mem.fused_matmul_reduce_scatter.default(
                act, wt.transpose(1, 0), 'avg', 0, world_group_name)

        def ag_gemm(act, wt):
            _, out = torch.ops.symm_mem.fused_all_gather_matmul.default(
                act, [wt.transpose(1, 0)], 0, world_group_name)
            return out[0]

    def gemm_rs_ag_gemm(
        residual: torch.Tensor,
        old_my_residual: torch.Tensor,
        gemm_1_weights: torch.Tensor,
        gemm_1_activations: torch.Tensor,
        rms_norm_weights: torch.Tensor,
        gemm_2_weights: torch.Tensor,
        first_layer: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        do_split = use_cc_kernels(residual.size(0))

        if first_layer and do_split:
            slice_shape = residual_slice_shape(residual, rank)
            residual_chunk = torch.ops.aten.split.Tensor(residual, slice_shape)
            my_residual = residual_chunk[0]
        else:
            my_residual = residual
            slice_shape = residual.size(0)

        if not do_split:
            output = torch.ops.aten.mm.default(gemm_1_activations,
                                               gemm_1_weights.transpose(1, 0))
            reduced_output = tensor_model_parallel_all_reduce(output)

            torch.ops._C.fused_add_rms_norm.default(input=reduced_output,
                                                    residual=my_residual,
                                                    weight=rms_norm_weights,
                                                    epsilon=1e-05)

            mm_2 = torch.ops.aten.mm.default(reduced_output,
                                             gemm_2_weights.transpose(1, 0))

            return mm_2, my_residual, my_residual.clone()
        else:
            output = gemm_rs(gemm_1_activations, gemm_1_weights)

            torch.ops._C.fused_add_rms_norm.default(input=output,
                                                    residual=my_residual,
                                                    weight=rms_norm_weights,
                                                    epsilon=1e-05)

            residual_1 = residual if first_layer else old_my_residual
            slice_scatter = torch.ops.aten.slice_scatter.default(
                residual_1, my_residual, 0, 0, slice_shape)
            split_2 = torch.ops.aten.split.Tensor(slice_scatter, slice_shape)
            new_residual = split_2[0]

            mm_2 = ag_gemm(output, gemm_2_weights)

            return mm_2, new_residual, slice_scatter

    def gemm_rs_ag_gemm_static(
        residual: torch.Tensor,
        old_my_residual: torch.Tensor,
        gemm_1_weights: torch.Tensor,
        gemm_1_activations: torch.Tensor,
        rms_norm_weights: torch.Tensor,
        gemm_2_weights: torch.Tensor,
        first_layer: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if first_layer:
            slice_shape = residual_slice_shape(residual, rank)
            residual_chunk = torch.ops.aten.split.Tensor(residual, slice_shape)
            my_residual = residual_chunk[0]
        else:
            slice_shape = residual.size(0)
            my_residual = residual

        output = gemm_rs(gemm_1_activations, gemm_1_weights)

        torch.ops._C.fused_add_rms_norm.default(input=output,
                                                residual=my_residual,
                                                weight=rms_norm_weights,
                                                epsilon=1e-05)

        residual_1 = residual if first_layer else old_my_residual
        slice_scatter = torch.ops.aten.slice_scatter.default(
            residual_1, my_residual, 0, 0, slice_shape)
        split_2 = torch.ops.aten.split.Tensor(slice_scatter, slice_shape)
        new_residual = split_2[0]

        mm_2 = ag_gemm(output, gemm_2_weights)

        return mm_2, new_residual, slice_scatter

    def gemm_rs_ag_gemm_fake(
        residual: torch.Tensor,
        my_residual: torch.Tensor,
        gemm_1_weights: torch.Tensor,
        gemm_1_activations: torch.Tensor,
        rms_norm_weights: torch.Tensor,
        gemm_2_weights: torch.Tensor,
        first_layer: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if first_layer and use_cc_kernels(residual.size(0)):
            slice_shape = residual_slice_shape(residual, rank)
            my_residual = torch.empty((slice_shape, residual.size(1)),
                                      device=residual.device,
                                      dtype=residual.dtype)
        else:
            my_residual = residual

        # TODO: verify the type is always correct
        mm_res = torch.empty(
            (gemm_1_activations.size(0), gemm_2_weights.size(0)),
            device=gemm_1_activations.device,
            dtype=gemm_1_activations.dtype)

        return (mm_res, my_residual, residual)

    if not hasattr(torch.ops.vllm, name):
        logger.info("registering torch.ops.vllm.%s", name)
        grag = gemm_rs_ag_gemm_static if is_static_shape else gemm_rs_ag_gemm
        direct_register_custom_op(name,
                                  grag,
                                  mutates_args=[],
                                  fake_impl=gemm_rs_ag_gemm_fake)
        assert getattr(torch.ops.vllm, name)

    return getattr(torch.ops.vllm, name).default, gemm_rs_ag_gemm_fake


def search_final_gemm_all_reduce(
    residual: torch.Tensor,
    gemm_1_weights: torch.Tensor,
    gemm_1_activations: torch.Tensor,
    rms_norm_weights: torch.Tensor,
) -> torch.Tensor:
    gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)

    all_reduce = tensor_model_parallel_all_reduce(mm_1)

    rmsnorm = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input=all_reduce,
        residual=residual,
        weight=rms_norm_weights,
        epsilon=1e-05)

    return rmsnorm[1], rmsnorm[2]


# Register this as a custom op since all gather cannot be torch.compiled yet.
def replace_final_gemm_ag(residual: torch.Tensor, gemm_1_weights: torch.Tensor,
                          gemm_1_activations: torch.Tensor,
                          rms_norm_weights: torch.Tensor) -> torch.Tensor:
    gemm_1_w_perm = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, gemm_1_w_perm)

    all_reduce = tensor_model_parallel_all_reduce(mm_1)

    # residual is reduce scattered in last pattern match, need to all gather here before proceeding
    # with the RMS norm
    tp = get_tp_group()
    tp_size = get_tensor_model_parallel_world_size()
    ag_residual = torch.ops.vllm.all_gather.default(residual,
                                                    dim=0,
                                                    world_size=tp_size,
                                                    group_name=tp.unique_name)

    rmsnorm = torch.ops.higher_order.auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input=all_reduce,
        residual=ag_residual,
        weight=rms_norm_weights,
        epsilon=1e-05)

    new_residual = torch.ops.vllm.reduce_scatter.default(
        rmsnorm[2], dim=0, world_size=tp_size, group_name=tp.unique_name)

    return rmsnorm[1], new_residual


def gemm_ag_final_static(my_residual: torch.Tensor,
                         gemm_1_weights: torch.Tensor,
                         gemm_1_activations: torch.Tensor,
                         rms_norm_weights: torch.Tensor) -> torch.Tensor:
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations,
                                     gemm_1_weights.transpose(1, 0))

    reduced = tensor_model_parallel_all_reduce(mm_1)

    wait_tensor = tensor_model_parallel_all_gather(my_residual)

    torch.ops._C.fused_add_rms_norm.default(input=reduced,
                                            residual=wait_tensor,
                                            weight=rms_norm_weights,
                                            epsilon=1e-05)

    return reduced


def gemm_ag_final_fake(my_residual: torch.Tensor, gemm_1_weights: torch.Tensor,
                       gemm_1_activations: torch.Tensor,
                       rms_norm_weights: torch.Tensor) -> torch.Tensor:
    return torch.empty([gemm_1_activations.size(0),
                        my_residual.size(1)],
                       dtype=my_residual.dtype,
                       device=my_residual.device)


# direct_register_custom_op("gemm_ag_final",
#                           gemm_ag_final,
#                           mutates_args=[],
#                           fake_impl=gemm_ag_final_fake)

#direct_register_custom_op("gemm_ag_final_static",
#                          gemm_ag_final_static,
#                          mutates_args=[],
#                          fake_impl=gemm_ag_final_fake)


def trace_fn(fn: Any, args: Sequence[Any]) -> fx.GraphModule:
    from torch._inductor.virtualized import NullHandler, V
    context = (V.fake_mode if
               (not isinstance(V.fake_mode, NullHandler) or
                (V.fake_mode is None)) else contextlib.nullcontext())

    with context:
        return fwd_only(fn, args)


match_count = [0]


def conditional_extra_check(match):
    # Increment the counter and return True to allow the replacement
    match_count[0] += 1
    # Record the match if needed
    self.record_match(match)
    return True


def conditional_replacement(match):
    # Choose replacement based on the current match count
    if match_count[0] == 1:
        # For the first match, use B1
        return replace_with_b1(match.nodes_matched["residual"],
                               match.nodes_matched["gemm_1_weights"],
                               match.nodes_matched["gemm_1_activations"],
                               match.nodes_matched["rms_norm_weights"])
    else:
        # For subsequent matches, use B2
        return replace_with_b2(match.nodes_matched["residual"],
                               match.nodes_matched["gemm_1_weights"],
                               match.nodes_matched["gemm_1_activations"],
                               match.nodes_matched["rms_norm_weights"])


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

        self.gemm_rs_ag_gemm_pattern = PatternMatcherPass()
        self.final_ar_rmsnorm_pattern = PatternMatcherPass()
        self.matches: List[Match] = []

        x = torch.empty([4, 4], device='cuda', dtype=torch.float16)
        w = torch.empty([4, 4], device='cuda', dtype=torch.float16)
        resid = torch.empty([4, 4], device='cuda', dtype=torch.float16)
        resid_w = torch.empty([4, 4], device='cuda', dtype=torch.float16)
        x2 = torch.empty([4, 4], device='cuda', dtype=torch.float16)
        inputs = [resid, x, w, resid_w]
        # inputs = [resid, x, w, resid_w, x2]
        # final_inputs = [x, w, resid, resid_w]
        # inputs = [resid]
        register_replacement(search_gemm_allreduce_rmsnorm,
                             replace_with_gemm_rs_ag_rmsnorm,
                             inputs,
                             fwd_only, [self.gemm_rs_ag_gemm_pattern],
                             extra_check=lambda m: self.record_match(m))

        # Nice TODO: handle static shape
        # register_replacement(search_final_gemm_all_reduce, replace_final_gemm_ag,
        #                      final_inputs,
        #                      fwd_only,
        #                      [self.final_ar_rmsnorm_pattern])

    def is_static_shape(self):
        # pass_context = get_pass_context()
        # return pass_context.runtime_shape is not None
        return True

    # TODO: Add type check
    def should_rewrite(self, match: Match) -> bool:
        return True
        # pass_context = get_pass_context()
        # if pass_context.runtime_shape is None:
        #     return self.config.enable_dynamic_collective_fusion
        # return use_cc_kernels(pass_context.runtime_shape)

    def record_match(self, match: Match) -> bool:
        # Hijack the extra_check to record the match and
        # save it for post-processing.
        # if self.should_rewrite(match):
        self.matches.append(match)

        # Return False to prevent automatic replacement.
        return bool(match)

    def collect_args(self, kwargs) -> Tuple[List[Any], List[Any]]:
        arg_names = {
            "residual": 0,
            "old_my_residual": 1,
            "gemm_1_weights": 2,
            "gemm_1_activations": 3,
            "rms_norm_weights": 4,
            "gemm_2_weights": 5,
            "first_layer": 6,
        }
        args = [None] * len(arg_names)
        node_args = [None] * len(arg_names)
        for k, v in kwargs.items():
            idx = arg_names[k]
            if isinstance(v, torch.fx.Node):
                if v.meta.get("val") is not None:
                    args[idx] = v.meta["val"]
            else:
                args[idx] = v
            node_args[idx] = v
        return node_args, args

    def my_process_matches(self, graph: fx.Graph) -> None:
        nodes = list(graph.nodes)

        def find_min_index(match: Match) -> int:
            return min(match.nodes, key=lambda x: nodes.index(x))

        # "sort" matches in topo order.
        matches = sorted(self.matches, key=lambda x: find_min_index(x))

        res_replacements: List[fx.Node] = []
        my_res_replacements: List[fx.Node] = []

        max_m = self.config.max_num_batched_tokens
        logger.info("max m = %d", max_m)

        n = 0
        for match in matches:
            last_node = last_node_in_match(match)
            first_layer = match == matches[0]

    def my_process_matches(self, graph: fx.Graph) -> None:
        nodes = list(graph.nodes)

        def find_min_index(match: Match) -> int:
            return min(match.nodes, key=lambda x: nodes.index(x))

        # "sort" matches in topo order.
        matches = sorted(self.matches, key=lambda x: find_min_index(x))

        first_rms_node = find_auto_fn(self.match.nodes,
                                      torch.ops._C.fused_add_rms_norm.default)

    def process_matches(self, graph: fx.Graph) -> None:

        def find_min_index(match: Match) -> int:
            return min(match.nodes, key=lambda x: nodes.index(x))

        nodes = list(graph.nodes)

        # "sort" matches in topo order.
        matches = sorted(self.matches, key=lambda x: find_min_index(x))

        res_replacements: List[fx.Node] = []
        my_res_replacements: List[fx.Node] = []

        max_m = self.config.max_num_batched_tokens
        logger.info("max m = %d", max_m)

        n = 0
        for match in matches:
            last_node = last_node_in_match(match)
            first_layer = match == matches[0]
            n = n + 1

            with graph.inserting_after(last_node):
                kwargs = match.kwargs
                kwargs["first_layer"] = first_layer
                kwargs["residual"] = res_replacements[-1] if len(
                    res_replacements) > 0 else match.kwargs["residual"]
                kwargs["old_my_residual"] = my_res_replacements[-1] if len(
                    my_res_replacements) > 0 else match.kwargs["residual"]

                gemm_1 = kwargs["gemm_1_weights"].meta.get("val")
                gemm_2 = kwargs["gemm_2_weights"].meta.get("val")
                if gemm_1 is None or gemm_2 is None:
                    raise ValueError("Missing 'val' in gemm weights meta data")

                # Extract group_name from matched code.  Use to
                # generate proper replacement code.
                ar_node = find_fn(match.nodes,
                                  torch.ops.vllm.all_reduce.default)
                assert ar_node is not None
                tp_group_name = ar_node.args[1]

                fused_gemm_func, fused_gemm_fake_func = get_gemm_rs_ag_gemm(
                    max_m, gemm_1.dtype, gemm_1.shape, gemm_2.dtype,
                    gemm_2.shape, tp_group_name, self.is_static_shape())

                fused_node = graph.call_function(fused_gemm_func,
                                                 kwargs=kwargs)

                graph.inserting_after(fused_node)
                result_node_new = graph.call_function(operator.getitem,
                                                      (fused_node, 0))
                residual_node_new = graph.call_function(
                    operator.getitem, (fused_node, 1))
                my_residual_node_new = graph.call_function(
                    operator.getitem, (fused_node, 2))

                res_replacements.append(residual_node_new)
                my_res_replacements.append(my_residual_node_new)

            rms_node = find_auto_fn(reversed(match.nodes),
                                    torch.ops._C.fused_add_rms_norm.default)
            gemm_node = find_fn(reversed(match.nodes),
                                torch.ops.aten.mm.default)
            assert rms_node is not None
            assert gemm_node is not None

            assert len(rms_node.users) == 2
            assert len(gemm_node.users) == 1 or len(gemm_node.users) == 2

            # Update meta data by using the fake function (optional)
            if False:
                node_args, args = self.collect_args(kwargs)
                fake_mod = trace_fn(fused_gemm_fake_func, args)
                outputs = [n for n in fake_mod.graph.nodes if n.op == 'output']
                assert len(outputs) == 1
                metas = []
                for out in outputs[0].args[0]:
                    metas.append(out.meta["val"])
                tm = tuple(metas)

                fused_node.meta["val"] = tm
                result_node_new.meta["val"] = tm[0]
                residual_node_new.meta["val"] = tm[1]
                my_residual_node_new.meta["val"] = tm[2]

            residual_getter_node = find_getitem(rms_node, 2)
            assert residual_getter_node is not None
            residual_getter_node.replace_all_uses_with(residual_node_new)
            gemm_node.replace_all_uses_with(result_node_new)

        # Finally, remove matched nodes
        graph.eliminate_dead_code()
        assert all(node not in graph.nodes for match in matches
                   for node in match.nodes)

    def __call__(self, graph: fx.Graph):
        # pass_context = get_pass_context()

        # if (pass_context.runtime_shape is None
        #         and not self.config.enable_dynamic_collective_fusion):
        #     logger.info("CollectiveFusionPass disabled for general shape.")
        #     return
        # else:
        #     logger.info("CollectiveFusionPass shape=%s",
        #                 pass_context.runtime_shape)

        #
        # TODO: would be nice to disable/assert if chunk prefill size is too
        # small but that info is not easily available here.
        #
        import torch.distributed as dist
        rank = dist.get_rank()
        if rank == 0:
            print(f"cascade before graph {graph}")

        self.dump_graph(graph, "before_collective_fusion")
        match_cnt = self.gemm_rs_ag_gemm_pattern.apply(graph)
        logger.info("fused gemm match count = %d, count2 = %d",
                    len(self.matches), match_cnt)

        # Don't apply final pattern unless we've matched and replaced the
        # gemm+collective ops.
        # if len(self.matches) > 0:
        #     count = self.final_ar_rmsnorm_pattern.apply(graph)
        #     logger.info("final pattern match count = %d", count)
        #     # self.process_matches(graph)
        if rank == 0:
            print(f"cascade after graph {graph}")
        self.dump_graph(graph, "after_collective_fusion")
        self.matches.clear()
