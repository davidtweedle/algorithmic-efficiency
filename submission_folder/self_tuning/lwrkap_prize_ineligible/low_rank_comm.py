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
            "eps",
            "l_memory_dict",
            "r_memory_dict",
            "X_memory_dict",
            "Y_memory_dict",
            ]
    def __init__(
            self,
            n_gpus,
            eps=1e-5,
            matrix_approximation_rank=8,
            batch_tensors_with_same_shape: bool = True
            ):
        self.n_gpus = n_gpus
        self.matrix_approximation_rank = matrix_approximation_rank
        self.batch_tensors_with_same_shape = batch_tensors_with_same_shape
        self.eps = eps
        self.l_memory_dict: Dict[int, torch.Tensor] = {}
        self.r_memory_dict: Dict[int, torch.Tensor] = {}
        self.Y_memory_dict: Dict[int, torch.Tensor] = {}
        self.X_memory_dict: Dict[int, torch.Tensor] = {}

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

    state.l_memory_dict[bucket_index] = torch.empty(
            total_Ls_size, device=device, dtype=dtype
            )
    state.r_memory_dict[bucket_index] = torch.empty(
            total_Rs_size, device=device, dtype=dtype
            )
    state.X_memory_dict[bucket_index] = torch.empty(
            total_Xs_size, device=device, dtype=dtype
            )
    state.Y_memory_dict[bucket_index] = torch.empty(
            total_Ys_size, device=device, dtype=dtype
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
                    l_idx: l_idx + batch_size * m * state.matrix_approximation_rank * n_gpus
                    ].view(n_gpus, batch_size, m, state.matrix_approximation_rank)
                )
        rs.append(
                state.r_memory_dict[bucket_index][
                    r_idx: r_idx + batch_size * n * state.matrix_approximation_rank * n_gpus
                    ].view(n_gpus, batch_size, state.matrix_approximation_rank, n)
                )
        Ys.append(
                state.Y_memory_dict[bucket_index][
                    y_idx: y_idx + batch_size * m * state.matrix_approximation_rank
                    ].view(batch_size, m, state.matrix_approximation_rank)
                )
        Xs.append(
                state.X_memory_dict[bucket_index][
                    x_idx: x_idx + batch_size * n * state.matrix_approximation_rank
                    ].view(batch_size, state.matrix_approximation_rank, n)
                )
        l_idx += batch_size * m * state.matrix_approximation_rank * n_gpus
        r_idx += batch_size * n * state.matrix_approximation_rank * n_gpus
        y_idx += batch_size * m * state.matrix_approximation_rank
        x_idx += batch_size * n * state.matrix_approximation_rank

    for i, tensor in enumerate(tensors_to_compress):
        batch_size, m, n = tensor.shape
        u = torch.randn(batch_size, n, state.matrix_approximation_rank, device=device)
        Y = torch.matmul(tensor, u)
        v = torch.randn(batch_size, state.matrix_approximation_rank, m, device=device)
        X = torch.matmul(v, tensor)
        middle = torch.matmul(X, u) if n < m else torch.matmul(v, Y)
        u, s, v = torch.linalg.svd(middle)
        s = torch.where(s > state.eps, s.pow(-1), torch.zeros_like(s))
        v = torch.matmul(v, torch.diag_embed(s, dim1=-2, dim2=-1))
        Y = torch.matmul(Y, v)
        X = torch.matmul(u.transpose(-1,-2), X)
        Xs[i].copy_(X)
        Ys[i].copy_(Y)


    allreduce_contiguous_uncompressed_tensors_fut = dist.all_reduce(
            uncompressed_tensors_memory, async_op=True
            ).get_future()

    def unpack_uncompressed_tensors_and_allgather_ls_and_rs(fut):
        uncompressed_tensors_memory = fut.value()[0].div_(n_gpus)
        idx = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(
                    uncompressed_tensors_memory[idx: idx + tensor.numel()].view_as(tensor)
                    )
            idx += tensor.numel()

        return (
                torch.futures.collect_all(
                    [dist.all_gather_into_tensor(state.l_memory_dict[bucket_index], state.Y_memory_dict[bucket_index], async_op=True
                    ).get_future(),
                    dist.all_gather_into_tensor(state.r_memory_dict[bucket_index], state.X_memory_dict[bucket_index], async_op=True
                    ).get_future()])
                .wait()
                )

    def decompress_ls_and_rs(fut):
        state.l_memory_dict[bucket_index] = fut.wait()[0].value()
        state.r_memory_dict[bucket_index] = fut.wait()[1].value()
        for l, r, tensor in zip(ls, rs, tensors_to_compress):
            tensor.copy_(
                    torch.sum(
                        torch.matmul(l, r), dim=0
                        )
                    )
            tensor.div_(n_gpus)

        if state.batch_tensors_with_same_shape:
            for tensor in tensors_to_compress:
                original_tensors = shape_to_tensors[tensor.shape[1:]]
                for i, original_tensor in enumerate(original_tensors):
                    original_tensor.copy_(tensor[i])

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        state.l_memory_dict.clear()
        state.r_memory_dict.clear()
        state.X_memory_dict.clear()
        state.Y_memory_dict.clear()

        return input_tensor

    return (
            allreduce_contiguous_uncompressed_tensors_fut.then(
                unpack_uncompressed_tensors_and_allgather_ls_and_rs
                )
            .then(decompress_ls_and_rs)
            )
