from vllm import LLM
llm = LLM("unsloth/Llama-3.2-1B-Instruct", tensor_parallel_size=4, enforce_eager=True)
output = llm.generate(prompts=["List an example of a wellknown city"] * 8, use_tqdm=True)

print(output)

# import torch
# tp_size = 4
# num_cols = 8
# rank = 1

# num_rows = tp_size + 1
# all_tensors = [
#     torch.rand((num_rows, num_cols), dtype=torch.float32, device="cuda") *
#     (r + 1) for r in range(tp_size)
# ]

# index = rank % tp_size
# rows_per_partition = (num_rows + tp_size - 1) // tp_size
# all_reduce = torch.sum(torch.stack(all_tensors, dim=0), dim=0)
# end = min((index+1) * rows_per_partition, num_rows)
# expected = all_reduce[index * rows_per_partition : end, :]
# t = all_tensors[index]

# print(f"{all_tensors=}")
# print(f"{all_reduce=}")
# print(f"{expected=}")
# print(f"{type(t)}, {t=}")


# import torch
# num_dimensions = 3
# tp_size = 2
# rank = 0
# tensor_size = list(range(2, num_dimensions + 2))
# total_size = 1
# for s in tensor_size:
#     total_size *= s
# print(f"{tensor_size=}, {total_size=}")

# for all_gather_dimension in range(num_dimensions):
#     all_tensors = [
#         torch.arange(total_size, dtype=torch.float32,
#                         device="cuda").reshape(tensor_size) * (r + 1)
#         for r in range(tp_size)
#     ]
#     print(f"{all_gather_dimension}, {all_tensors=}")
#     expected = torch.cat(all_tensors, dim=all_gather_dimension)
#     t = all_tensors[rank % tp_size]
    
#     print(f"expected.size={expected.shape},{expected=}")
#     print(f"t.size={t.shape}, {t=}")