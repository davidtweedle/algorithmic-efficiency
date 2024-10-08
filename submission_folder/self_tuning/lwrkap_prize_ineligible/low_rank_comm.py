from absl import logging
from collections import defaultdict
from typing import Dict

import torch
import torch.distributed as dist

from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

class LowRankApproximationState:
    """ A class to store all the state information for
        the communication hook
    """

    __slots__ = [
            "n_gpus",
            "matrix_approximation_rank",
            "upper_bound_rank",
            "batch_tensors_with_same_shape",
            "eps",
            "global_step",
            "cur_grad_norm",
            "num_iter_svd"
            ]

    def __init__(
            self,
            n_gpus,
            eps=1e-16,
            matrix_approximation_rank=8,
            upper_bound_rank=32,
            batch_tensors_with_same_shape: bool = True,
            global_step=0,
            num_iter_svd=0,
            ):
        self.n_gpus = n_gpus
        self.matrix_approximation_rank = matrix_approximation_rank
        self.upper_bound_rank = upper_bound_rank
        self.batch_tensors_with_same_shape = batch_tensors_with_same_shape
        self.eps = eps
        self.global_step = global_step
        self.cur_grad_norm = 0
        self.num_iter_svd = num_iter_svd

    def __getstate__(self):
        return {
                slot: getattr(self, slot)
                for slot in self.__slots__
                }

    def __setstate__(self, state):
        for slot, value in state.items():
            setattr(self, slot, value)

    def maybe_increase_iter(self, bucket):
        if bucket.is_last():
            self.global_step += 1

def svd_approximator(grad, upper_bound_rank, svd_rank, device, n_gpus):
    oldshape = grad.shape
    reshaped_grad = grad.reshape(oldshape[0], -1)
    m, n, _ = *reshaped_grad.shape, 1
    upper_rank = min(m, n, upper_bound_rank)
    rank = min(upper_rank, svd_rank)
    if upper_rank > 1:
        try:
            U, S, V = torch.svd_lowrank(
                    reshaped_grad,
                    q=upper_rank
                    )
            U = U[:, :rank]
            S = S[:rank]
            V = V[:, :rank]
            reshaped_grad = (U * S) @ V.T
        except torch._C._LinAlgError as err:
            logging.info(f'SVD approximator threw error {err}')
    grad = reshaped_grad.reshape(*oldshape)
    grad.div_(n_gpus)
    return grad

def normalize_sv_approximator(grad, rank, device, n_gpus):
    oldshape = grad.shape
    reshaped_grad = grad.reshape(oldshape[0], -1)
    m, n, _ = *reshaped_grad.shape, 1
    rank = min(m, n, rank)
    if min(m, n) > 1:
        try:
            U, _, V = torch.svd_lowrank(
                    reshaped_grad,
                    q=rank,
                    niter=1
                    )
            reshaped_grad = U @ V.transpose(-1, -2)
        except torch._C._LinAlgError as err:
            logging.info(f'SVD approximator threw error {err}')
    grad = reshaped_grad.reshape(*oldshape)
    grad.div_(n_gpus)
    return grad


def sketch_approximator(grad, low_rank, device, n_gpus):
    ## figure out how to set random seeds on each device independently
    oldshape = grad.shape
    reshaped_grad = grad.reshape(oldshape[0], -1)
    m, n, _ = *reshaped_grad.shape, 1
    switch = m < n
    if switch:
        reshaped_grad = reshaped_grad.transpose(-1, -2)
        m, n = n, m
    if n > low_rank:
        Y = torch.randn(low_rank, m, device=device)
        Y = torch.matmul(Y, reshaped_grad)
        X = torch.randn(n, int(low_rank * 1.5), device=device)
        Y = torch.linalg.lstsq(torch.matmul(Y, X), Y).solution
        X = torch.matmul(reshaped_grad, X)
        grad = torch.matmul(X, Y)
        grad.div_(n_gpus)
        if switch:
            grad = grad.transpose(-1, -2)
        grad = grad.reshape(*oldshape)
    return grad


def low_rank_sketch(grad, state: LowRankApproximationState):
    m, n = grad.shape[-2:]
    rank = min(state.matrix_approximation_rank, m, n)
    U, _, V = torch.svd_lowrank(grad, niter=state.num_iter_svd, q=rank)
    return U, V.transpose(-1, -2)


def lwrk_hook(state: LowRankApproximationState, bucket):
    n_gpus = state.n_gpus

    input_tensor = bucket.buffer()

    device = input_tensor.device
    dtype = input_tensor.dtype

    bucket_index = bucket.index()

    tensors = bucket.gradients()

    tensors_to_compress, uncompressed_tensors = [], []
    total_Xs_size = 0
    total_Ys_size = 0

    for tensor in tensors:
        matrix = tensor.view(tensor.shape[0], -1)
        m, n = matrix.shape
        rank = min(m, n, state.matrix_approximation_rank)
        if min(m, n) > 1:
            tensors_to_compress.append(matrix)
            total_Ys_size += m * rank
            total_Xs_size += n * rank
        else:
            uncompressed_tensors.append(tensor)

    uncompressed_tensors_memory = (
            torch.cat([tensor.view(-1) for tensor in uncompressed_tensors])
            if uncompressed_tensors
            else torch.tensor([], device=device, dtype=dtype)
            )

    l_memory = [
            torch.empty(
                total_Ys_size, device=device, dtype=dtype) 
            for _ in range(n_gpus)
            ]
    r_memory = [
            torch.empty(
                total_Xs_size, device=device, dtype=dtype)
            for _ in range(n_gpus)
            ]
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
    y_idx = 0
    x_idx = 0
    for tensor in maybe_batched_tensors_to_compress():
        batch_size, m, n = tensor.shape
        rank = min(m, n, state.matrix_approximation_rank)
        tensors_to_compress.append(tensor)
        ls.append(list([l_memory[j][
            y_idx: y_idx + batch_size * m * rank
            ].view(batch_size, m, rank 
                   ) for j in range(n_gpus)])
                )
        rs.append(list([r_memory[j][
            x_idx: x_idx + batch_size * n * rank
            ].view(batch_size, rank, n
                   ) for j in range(n_gpus)]))
        Ys.append(Y_memory[
            y_idx: y_idx + batch_size * m * rank
            ].view(batch_size, m, rank)
                  )
        Xs.append(X_memory[
            x_idx: x_idx + batch_size * n * rank
            ].view(batch_size, rank, n)
                )
        y_idx += batch_size * m * rank
        x_idx += batch_size * n * rank

    for i, tensor in enumerate(tensors_to_compress):
        U, Vh = low_rank_sketch(tensor, state)
        Ys[i].copy_(U)
        Xs[i].copy_(Vh)

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
                        dist.all_gather(
                            l_memory,
                            Y_memory,
                            async_op=True
                            ).get_future(),
                        dist.all_gather(
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
            gathered_tensor = torch.matmul(
                    torch.cat(l, dim=-1), torch.cat(r, dim=-2)
                    )
            gathered_tensor.div_(n_gpus)
            tensor.copy_(gathered_tensor)

        if state.batch_tensors_with_same_shape:
            for tensor in tensors_to_compress:
                if tensor.shape[0] == 1:
                    continue
                original_tensors = shape_to_tensors[tensor.shape[1:]]
                for i, original_tensor in enumerate(original_tensors):
                    original_tensor.copy_(tensor[i])

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        state.maybe_increase_iter(bucket)
        return input_tensor

    return (
            allreduce_contiguous_uncompressed_tensors_fut.then(
                unpack_uncompressed_tensors_and_allgather_ls_and_rs
                )
            .then(decompress_ls_and_rs)
            )


def simple_lwrk_hook(state: LowRankApproximationState, bucket):
    input_tensor = bucket.buffer()
    state.cur_grad_norm += torch.norm(input_tensor)
    if bucket.is_last():
        logging.info(f"Current grad norm is: {state.cur_grad_norm}")
        state.cur_grad_norm = 0
    dtype = input_tensor.dtype
    device = input_tensor.device
    n_gpus = state.n_gpus
    rank = state.matrix_approximation_rank
    for grad in bucket.gradients():
        grad.copy_(
                normalize_sv_approximator(
                    grad.clone().detach(),
                    rank,
                    device,
                    n_gpus
                    )
                )
    state.maybe_increase_iter(bucket)    
    return dist.all_reduce(
            input_tensor, 
            async_op=True
            ).get_future(
                    ).then(lambda fut: fut.value()[0]
                           )
