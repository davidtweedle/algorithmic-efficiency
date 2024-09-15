from collections import defaultdict
from typing import Dict

import torch
import torch.distributed as dist


class LowRankApproximationState:
    """ A class to store all the state information for
        the communication hook
    """

    __slots__ = [
            "n_gpus",
            "matrix_approximation_rank",
            "batch_tensors_with_same_shape",
            "eps",
            ]

    def __init__(
            self,
            n_gpus,
            eps=1e-16,
            matrix_approximation_rank=8,
            batch_tensors_with_same_shape: bool = True
            ):
        self.n_gpus = n_gpus
        self.matrix_approximation_rank = matrix_approximation_rank
        self.batch_tensors_with_same_shape = batch_tensors_with_same_shape
        self.eps = eps

    def __getstate__(self):
        return {
                slot: getattr(self, slot)
                for slot in self.__slots__
                }

    def __setstate__(self, state):
        for slot, value in state.items():
            setattr(self, slot, value)


def low_rank_sketch(grad, state: LowRankApproximationState):
    batch_size, m, n = grad.shape
    device = grad.device
    dtype = grad.dtype
    switch = m < n
    k2 = int(state.matrix_approximation_rank * (1 + switch * 0.5))
    k1 = int(state.matrix_approximation_rank * (1.5 - switch * 0.5))
    u = torch.randn(batch_size, n, k1, dtype=dtype, device=device)
    v = torch.randn(batch_size, k2, m, dtype=dtype, device=device)
    Y = torch.matmul(grad, u)
    X = torch.matmul(v, grad)
    mid = torch.matmul(v, Y) if switch else torch.matmul(X, u)
    u, S, v = torch.linalg.svd(mid, full_matrices=False)
    S = torch.where(S > state.eps, S, torch.ones_like(S) * state.eps).pow(-1)
    v = torch.matmul(v.transpose(-1, -2), S.diag_embed())
    X = torch.matmul(u.transpose(-1, -2), X)
    Y = torch.matmul(Y, v)
    return Y, X


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

    l_memory = torch.empty(
            total_Ls_size, device=device, dtype=dtype
            )
    r_memory = torch.empty(
            total_Rs_size, device=device, dtype=dtype
            )
    X_memory = torch.empty(
            total_Xs_size, device=device, dtype=dtype
            )
    Y_memory = torch.empty(
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
                l_memory[
                    l_idx: l_idx + batch_size * m * state.matrix_approximation_rank * n_gpus
                    ].view(n_gpus, batch_size, m, state.matrix_approximation_rank)
                )
        rs.append(
                r_memory[
                    r_idx: r_idx + batch_size * n * state.matrix_approximation_rank * n_gpus
                    ].view(n_gpus, batch_size, state.matrix_approximation_rank, n)
                )
        Ys.append(
                Y_memory[
                    y_idx: y_idx + batch_size * m * state.matrix_approximation_rank
                    ].view(batch_size, m, state.matrix_approximation_rank)
                )
        Xs.append(
                X_memory[
                    x_idx: x_idx + batch_size * n * state.matrix_approximation_rank
                    ].view(batch_size, state.matrix_approximation_rank, n)
                )
        l_idx += batch_size * m * state.matrix_approximation_rank * n_gpus
        r_idx += batch_size * n * state.matrix_approximation_rank * n_gpus
        y_idx += batch_size * m * state.matrix_approximation_rank
        x_idx += batch_size * n * state.matrix_approximation_rank

    for i, tensor in enumerate(tensors_to_compress):
        Y, X = low_rank_sketch(tensor, state)
        Xs[i] = X.div_(n_gpus.pow(0.5))
        Ys[i] = Y.div_(n_gpus.pow(0.5))

    allreduce_contiguous_uncompressed_tensors_fut = dist.all_reduce(
            uncompressed_tensors_memory, async_op=True
            ).get_future()

    def unpack_uncompressed_tensors_and_allgather_ls_and_rs(fut):
        uncompressed_tensors_memory = fut.value()[0].div_(n_gpus)
        idx = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(
                    uncompressed_tensors_memory[
                        idx: idx + tensor.numel()
                        ].view_as(tensor)
                    )
            idx += tensor.numel()

        return (
                torch.futures.wait_all(
                    [
                        dist.all_gather_into_tensor(
                            l_memory,
                            Y_memory,
                            async_op=True
                            ).get_future(),
                        dist.all_gather_into_tensor(
                            r_memory,
                            X_memory,
                            async_op=True
                            ).get_future()
                        ]
                    )
                )

    def decompress_ls_and_rs(fut):
        fut_list = fut.value()
        l_memory = fut_list[0]
        r_memory = fut_list[1]
        for l, r, tensor in zip(ls, rs, tensors_to_compress):
            tensor.copy_(
                    torch.sum(
                        torch.matmul(l, r), dim=0
                        )
                    )

        if state.batch_tensors_with_same_shape:
            for tensor in tensors_to_compress:
                if tensor.shape[0] == 1:
                    continue
                original_tensors = shape_to_tensors[tensor.shape[1:]]
                for i, original_tensor in enumerate(original_tensors):
                    original_tensor.copy_(tensor[i])

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        return input_tensor

    return (
            allreduce_contiguous_uncompressed_tensors_fut.then(
                unpack_uncompressed_tensors_and_allgather_ls_and_rs
                )
            .then(decompress_ls_and_rs)
            )
