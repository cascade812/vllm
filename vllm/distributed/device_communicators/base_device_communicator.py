# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


class DeviceCommunicatorBase:
    """
    Base class for device-specific communicator.
    It can use the `cpu_group` to initialize the communicator.
    If the device has PyTorch integration (PyTorch can recognize its
    communication backend), the `device_group` will also be given.
    """

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        self.device = device or torch.device("cpu")
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name
        self.rank = dist.get_rank(cpu_group)
        self.world_size = dist.get_world_size(cpu_group)
        self.ranks = dist.get_process_group_ranks(cpu_group)
        self.global_rank = dist.get_rank()
        self.global_world_size = dist.get_world_size()
        self.rank_in_group = dist.get_group_rank(self.cpu_group,
                                                 self.global_rank)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(input_, group=self.device_group)
        return input_
    
    def reduce_scatter(self, input_: torch.Tensor) -> torch.Tensor:   
        # Since reduce_scatter_tensor strictly requires input size = output_size * world_size, we need to pad the input tensor before calling the function  
        input_, pad_amount = self.pad_to_divisible(input_)
        input_size_after_padding = input_.size()
        output_size = (input_size_after_padding[0] // self.world_size, ) + input_size_after_padding[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(output_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        dist.reduce_scatter_tensor(output_tensor, input_, group=self.device_group)

        # After reduce_scatter_tensor, truncate the excess padding across ranks
        valid_tokens_per_rank = input_.shape[0] // self.world_size  # This assumes equal distribution
        extra_tokens = input_.shape[0] % self.world_size  # Remainder that was padded

        if self.rank < extra_tokens:
            valid_tokens_per_rank += 1  # Some ranks get 1 more token

        print(f"{self.rank=}, {valid_tokens_per_rank=}, {output_tensor.size()=}")
        output_tensor = output_tensor[:valid_tokens_per_rank]  # Truncate correctly
        print(f"{self.rank=}, after truncate {output_tensor.size()=}")
        return output_tensor


    def pad_to_divisible(self, input_: torch.Tensor, pad_value=0):
        seq_len = input_.shape[0]  # Assuming tensor shape is (seq_len, hidden_size)
        padded_len = ((seq_len + self.world_size - 1) // self.world_size) * self.world_size  # Next multiple of world_size
        pad_amount = padded_len - seq_len
        
        if pad_amount > 0:
            pad_tensor = torch.full((pad_amount, input_.shape[1]), pad_value, dtype=input_.dtype, device=input_.device)
            input_ = torch.cat([input_, pad_tensor], dim=0)  # Concatenate along sequence dimension
        
        return input_, pad_amount

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size, ) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(output_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # All-gather.
        dist.all_gather_into_tensor(output_tensor,
                                    input_,
                                    group=self.device_group)
        # Reshape
        output_tensor = output_tensor.reshape((self.world_size, ) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (self.world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor

    def gather(self,
               input_: torch.Tensor,
               dst: int = 0,
               dim: int = -1) -> Optional[torch.Tensor]:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # Gather.
        torch.distributed.gather(input_,
                                 gather_list,
                                 dst=self.ranks[dst],
                                 group=self.device_group)
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(self,
             size: torch.Size,
             dtype: torch.dtype,
             src: Optional[int] = None) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        pass
