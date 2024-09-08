from collections import defaultdict
from absl import logging
from typing import Dict

import torch
import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default
from torch.distributed import distributed_c10d


class LowRankApproximationState:
    """ A class to store all the state information for
        the communication hook
    """

    __slots__ = [
            "n_gpus",
            "matrix_approximation_rank",
            "batch_tensors_with_same_shape",
            "l_memory_dict",
            "r_memory_dict",
            "global_step"
            ]
    def __init__(
            self,
            n_gpus,
            matrix_approximation_rank=8,
            batch_tensors_with_same_shape: bool = True
            ):
        self.n_gpus = n_gpus
        self.matrix_approximation_rank = matrix_approximation_rank
        self.batch_tensors_with_same_shape = batch_tensors_with_same_shape
        self.l_memory_dict: Dict[int, torch.Tensor] = {}
        self.r_memory_dict: Dict[int, torch.Tensor] = {}
        self.global_step = 0

    def __getstate__(self):
        return {
                slot: getattr(self, slot)
                for slot in self.__slots__
                }


    def __setstate__(self, state):
        for slot, value in state.items():
            setattr(self, slot, value)


def lwrk_hook(state: LowRankApproximationState, bucket):
    n_gpus = state.n_gpus

    input_tensor = bucket.buffer()

    device = input_tensor.device
    dtype = input_tensor.dtype

    bucket_index = bucket.index()

    tensors = bucket.gradients()
    tensors_to_compress, uncompressed_tensors = [], []
    total_Ls_size = 0
    total_Rs_size = 0
    total_Xs_size = 0
    total_Ys_size = 0
    for tensor in tensors:
        matrix = tensor.view(tensor.shape[0], -1)
        m, n = matrix.shape
        if min(m, n) > state.matrix_approximation_rank:
            tensors_to_compress.append(matrix)
            total_Ls_size += m * state.matrix_approximation_rank * n_gpus
            total_Rs_size += n * state.matrix_approximation_rank * n_gpus
            total_Ys_size += m * state.matrix_approximation_rank
            total_Xs_size += n * state.matrix_approximation_rank
        else:
            uncompressed_tensors.append(tensor)

    uncompressed_tensors_memory = (
            torch.cat([tensor.view(-1) for tensor in uncompressed_tensors])
            if uncompressed_tensors
            else torch.tensor([], device=device, dtype=dtype)
            )
    
    if bucket_index not in state.l_memory_dict:
        state.l_memory_dict[bucket_index] = torch.empty(
                total_Ls_size, device=device, dtype=dtype
                )
        state.r_memory_dict[bucket_index] = torch.empty(
                total_Rs_size, device=device, dtype=dtype
                )
    shape_to_tensors = defaultdict(list)
    for tensor in tensors_to_compress:
        shape_to_tensors[tensor.shape].append(tensor)

    def maybe_batched_tensors_to_compress():
        for tensors in shape_to_tensors.values():
            if state.batch_tensors_with_same_shape:
                batch_size = len(tensors)
                if batch_size == 1:
                    yield tensors[0].unsqueeze(0)
                else:
                    yield torch.stack(tensors)
            else:
                for tensor in tensors:
                    yield tensor.unsqueeze(0)

    tensors_to_compress = []
    ls = []
    rs = []
    Ys = []
    Xs = []
    l_idx = 0
    r_idx = 0
    y_idx = 0
    x_idx = 0
    for tensor in maybe_batched_tensors_to_compress():
        batch_size, m, n = tensor.shape
        tensors_to_compress.append(tensor)
        ls.append(
                state.l_memory_dict[bucket_index][
                    l_idx : l_idx + batch_size * m * state.matrix_approximation_rank * n_gpus
                    ].view(n_gpus, batch_size, m, state.matrix_approximation_rank)
                )
        rs.append(
                state.r_memory_dict[bucket_index][
                    r_idx : r_idx + batch_size * n * state.matrix_approximation_rank * n_gpus
                    ].view(n_gpus, batch_size, n, state.matrix_approximation_rank)
                )
        Ys.append(torch.randn(batch_size, state.matrix_approximation_rank, m, device=device))
        Xs.append(torch.randn(batch_size, n, state.matrix_approximation_rank, device=device))

        l_idx += batch_size * m * state.matrix_approximation_rank * n_gpus
        r_idx += batch_size * n * state.matrix_approximation_rank * n_gpus

    for tensor, X, Y in zip(tensors_to_compress, Xs, Ys):
        batch_size, m, n = tensor.shape
        if n <= m:
            Y = torch.bmm(Y, tensor)
            middle, X = torch.bmm(torch.cat((Y, tensor), dim=1), X).split([state.matrix_approximation_rank, m], 1)
            a, tau = torch.geqrf(middle)
            Y = torch.ormqr(torch.tril(a, diagonal=-1), tau, Y, left=True, transpose=True)
            X = torch.linalg.solve_triangular(a, X, upper=True, left=False)

        else:
            X = torch.bmm(tensor, X)
            middle, Y = torch.bmm(Y, torch.cat((X,tensor), dim=2)).split([state.matrix_approximation_rank, n], 2)
            a, tau = torch.geqrf(middle)
            Y = torch.ormqr(torch.tril(a, diagonal=-1), tau, Y, left=True, transpose=True)
            X = torch.linalg.solve_triangular(a, X, upper=True, left=False)


    allreduce_contiguous_uncompressed_tensors_fut = dist.all_reduce(
            uncompressed_tensors_memory, async_op=True
            ).get_future()

    def unpack_uncompressed_tensors_and_allgather_ls_and_rs(fut):
        uncompressed_tensors_memory = fut.value()[0].div_(n_gpus)
        idx = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(
                    uncompressed_tensors_memory[idx : idx + tensor.numel()].view_as(tensor)
                    )
            idx += tensor.numel()

        return (
                torch.futures.collect_all([
                    dist.all_gather(state.l_memory_dict[bucket_index], Ys, async_op=True
                    ).get_future(),
                    dist.all_gather(state.r_memory_dict[bucket_index], Xs, async_op=True
                    ).get_future()])
                .wait()
                )

    def decompress_ls_and_rs(fut):
        state.l_memory_dict[bucket_index] = fut.wait()[0].value()
        state.r_memory_dict[bucket_index] = fut.wait()[1].value()
        for l, r, tensor in zip(ls, rs, tensors_to_compress):
            l = torch.cat(torch.unbind(l, dim=0), dim=-1)
            r = torch.cat(torch.unbind(r, dim=0), dim=-1)
            torch.bmm(l, r, out=tensor)

        if state.batch_tensors_with_same_shape:
            for tensor in tensor_to_compress:
                if tensor.shape[0] == 1:
                    continue
                original_tensors = shape_to_tensors[tensor.shape[1:]]
                for i, original_tensor in enumerate(original_tensors):
                    original_tensor.copy_(tensor[i])

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        state.l_memory_dict.clear()
        state.r_memory_dict.clear()

        return input_tensor

    return (
            allreduce_contiguous_uncompressed_tensors_fut.then(
                unpack_uncompressed_tensors_and_allgather_ls_and_rs
                )
            .then(decompress_ls_and_rs)
            )
