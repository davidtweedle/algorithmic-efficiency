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
            "l_memory_dict",
            "r_memory_dict",
            "X_memory_dict",
            "Y_memory_dict",
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
    switch = m < n
    k2 = int(state.matrix_approximation_rank * (1 + switch * 0.5))
    k1 = int(state.matrix_approximation_rank * (1.5 - switch * 0.5))
    u = torch.randn(batch_size, n, k1, device=device)
    v = torch.randn(batch_size, k2, m, device=device)
    Y = torch.matmul(grad, u)
    X = torch.matmul(v, grad)
    mid = torch.matmul(v, Y) if switch else torch.matmul(X, u)
    U, S, Vh = torch.linalg.svd(mid, full_matrices=False)
    S = torch.where(S > state.eps, S.pow(-1), torch.ones_like(S) * state.eps)
    Vh = torch.matmul(Vh.transpose(-1, -2), S.diag_embed())
    X = torch.matmul(U.transpose(-1, -2), X)
    Y = torch.matmul(Y, Vh)
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
    l_memory_dict: Dict[int, torch.Tensor] = {}
    r_memory_dict: Dict[int, torch.Tensor] = {}
    Y_memory_dict: Dict[int, torch.Tensor] = {}
    X_memory_dict: Dict[int, torch.Tensor] = {}

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

    l_memory_dict[bucket_index] = torch.empty(
            total_Ls_size, device=device, dtype=dtype
            )
    r_memory_dict[bucket_index] = torch.empty(
            total_Rs_size, device=device, dtype=dtype
            )
    X_memory_dict[bucket_index] = torch.empty(
            total_Xs_size, device=device, dtype=dtype
            )
    Y_memory_dict[bucket_index] = torch.empty(
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
                l_memory_dict[bucket_index][
                    l_idx: l_idx + batch_size * m * state.matrix_approximation_rank * n_gpus
                    ].view(n_gpus, batch_size, m, state.matrix_approximation_rank)
                )
        rs.append(
                r_memory_dict[bucket_index][
                    r_idx: r_idx + batch_size * n * state.matrix_approximation_rank * n_gpus
                    ].view(n_gpus, batch_size, state.matrix_approximation_rank, n)
                )
        Ys.append(
                Y_memory_dict[bucket_index][
                    y_idx: y_idx + batch_size * m * state.matrix_approximation_rank
                    ].view(batch_size, m, state.matrix_approximation_rank)
                )
        Xs.append(
                X_memory_dict[bucket_index][
                    x_idx: x_idx + batch_size * n * state.matrix_approximation_rank
                    ].view(batch_size, state.matrix_approximation_rank, n)
                )
        l_idx += batch_size * m * state.matrix_approximation_rank * n_gpus
        r_idx += batch_size * n * state.matrix_approximation_rank * n_gpus
        y_idx += batch_size * m * state.matrix_approximation_rank
        x_idx += batch_size * n * state.matrix_approximation_rank

    for i, tensor in enumerate(tensors_to_compress):
        Y, X = low_rank_sketch(tensor, state)
        Xs[i] = X
        Ys[i] = Y

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
                    [
                        dist.all_gather_into_tensor(
                            l_memory_dict[bucket_index],
                            Y_memory_dict[bucket_index],
                            async_op=True
                            ).get_future(),
                        dist.all_gather_into_tensor(
                            r_memory_dict[bucket_index],
                            X_memory_dict[bucket_index],
                            async_op=True
                            ).get_future()
                        ]
                    ).wait()
                )

    def decompress_ls_and_rs(fut):
        l_memory_dict[bucket_index] = fut.wait()[0].value()
        r_memory_dict[bucket_index] = fut.wait()[1].value()
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

        return input_tensor

    return (
            allreduce_contiguous_uncompressed_tensors_fut.then(
                unpack_uncompressed_tensors_and_allgather_ls_and_rs
                )
            .then(decompress_ls_and_rs)
            )
