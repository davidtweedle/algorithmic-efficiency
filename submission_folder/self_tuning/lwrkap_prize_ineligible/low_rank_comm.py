import torch.distributed as dist
import torch

from absl import logging

class LowRankApproximationState:
  """ A class to store all the state information for
      the communication hook
  """

  def __init__(
          self,
          svd_rank=3,
          upper_bound_rank=30,
          tol=1e-3,
          random_state=0,
          gpu_id=0,
          n_gpus=1,
          global_step=0,
          num_errs=0
          ):
    self.svd_rank = svd_rank
    self.upper_bound_rank = upper_bound_rank
    self.tol = tol
    self.random_state = random_state
    self.gpu_id = gpu_id
    self.n_gpus = n_gpus
    self.global_step = global_step
    self.num_errs = num_errs

  def __setstate__(self, state):
    self.svd_rank = state['svd_rank']
    self.upper_bound_rank = state['upper_bound_rank']
    self.tol = state['tol']
    self.random_state = state['random_state']
    self.gpu_id = state['gpu_id']
    self.n_gpus = state['n_gpus']
    self.global_step = state['global_step']
    self.num_errs = state['num_errs']

  def __getstate__(self):
    res = {'svd_rank': self.svd_rank,
           'upper_bound_rank': self.upper_bound_rank,
           'tol': self.tol,
           'random_state': self.random_state,
           'gpu_id': self.gpu_id,
           'n_gpus': self.n_gpus,
           'global_step': self.global_step,
           'num_errs': self.num_errs
           }
    return res

  def getstep(self):
    return self.global_step

  def setstep(self, step):
    self.global_step = step

def svd_hook(lrka_state: LowRankApproximationState, bucket):
  if lrka_state.getstep() > 1:
    for grad in bucket.gradients():
      oldshape = grad.shape
      reshaped_grad = grad.reshape(oldshape[0], -1)
      m, n, _ = *reshaped_grad.shape, 1
      upper_rank = min(m, n, lrka_state.upper_bound_rank)
      rank = min(upper_rank, lrka_state.svd_rank)
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
          logging.info(f'Communication hook threw error number {lrka_state.num_errs}')
      grad = reshaped_grad.reshape(*oldshape)
      grad.div_(lrka_state.n_gpus)
  return dist.all_reduce(bucket.buffer(), async_op=True).get_future().then(lambda fut: fut.value()[0])
