import torch.distributed as dist
import torch

from absl import logging

class LowRankApproximationState:
  """ A class to store all the state information for
      the communication hook
  """

  def __init__(
          self,
          approximator
          ):
    self.approximator = approximator

  def __setstate__(self, state):
    self.approximator = state['approximator']

  def __getstate__(self):
    res = {'approximator': self.approximator
          }
    return res

def lwrk_hook(lrka_state: LowRankApproximationState, bucket):
  approximator = lrka_state.approximator
  for grad in bucket.gradients():
    grad = approximator(grad)
  return dist.all_reduce(
          bucket.buffer(), 
          async_op=True
          ).get_future(
                  ).then(
                  lambda fut: fut.value()[0]
                  )

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
      lrka_state.num_errs += 1
      logging.info(f'SVD approximator threw error {err}')
  grad = reshaped_grad.reshape(*oldshape)
  grad.div_(n_gpus)
  return grad

def sketch_approximator(grad, rank, device, n_gpus)
  oldshape = grad.shape
  reshaped_grad = grad.reshape(oldshape[0], -1)
  m, n, _ = *reshaped_grad.shape, 1
  rank = int(min(m, n, low_rank))
  if n > 1:      
    Y = torch.randn(rank, m, device=device)
    Y = torch.matmul(Y, reshaped_grad)
    X = torch.randn(n, rank, device=device)
    Y = torch.linalg.lstsq(torch.matmul(Y,X),Y).solution
    X = torch.matmul(reshaped_grad, X)
    grad = torch.matmul(X, Y).reshape(*oldshape)
    grad.div_(n_gpus)
  return grad
