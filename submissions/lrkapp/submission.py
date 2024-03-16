"""Submission by David Tweedle to ML Commons algorithmic-efficiency contest
reference_algorithms/paper_baselines/momentum/pytorch/submission.py used as a base

Added low rank approximation to SGD
"""

from typing import Callable, Dict, Iterator, List, Tuple
import collections

from absl import logging
import optax
import torch
import torch.distributed.nn as dist_nn
from torch.optim.lr_scheduler import LambdaLR

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()

import tensorly as tl
tl.set_backend("pytorch")

def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a Nesterov optimizer and a learning rate schedule.
  Returns:
  optimizer state
  optimizer_update_fn
  """
  del model_state
  del rng
  if hyperparameters is None:
    hparams_dict = {'learning_rate': 0.5,
                    'momentum': 0,
                    'l2': 0,
                    'warmup_factor': 0.05
                    }
    hyperparameters = collections.namedtuple('Hyperparameters', hparams_dict)(**hparams_dict)

  optimizer_state = {
      'optimizer':
          torch.optim.SGD(
              model_params.parameters(),
              lr=hyperparameters.learning_rate,
              momentum=hyperparameters.momentum,
              weight_decay=hyperparameters.l2),
  }

  lr_schedule_fn = create_lr_schedule_fn(workload.step_hint, hyperparameters)
  def _lr_lambda(step: int) -> float:
    return lr_schedule_fn(step).item() / hyperparameters.learning_rate

  optimizer_state['scheduler'] = LambdaLR(
      optimizer_state['optimizer'], lr_lambda=_lr_lambda)

  return optimizer_state
 
def create_lr_schedule_fn(
    step_hint: int,
    hyperparameters: spec.Hyperparameters) -> Callable[[int], float]:
  warmup_steps = int(hyperparameters.warmup_factor * step_hint)
  warmup_fn = optax.linear_schedule(
      init_value=0.,
      end_value=hyperparameters.learning_rate,
      transition_steps=warmup_steps)
  decay_steps = step_hint - warmup_steps
  polynomial_schedule_fn = optax.polynomial_schedule(
      init_value=hyperparameters.learning_rate,
      end_value=hyperparameters.learning_rate * hyperparameters.end_factor,
      power=1,
      transition_steps=int(decay_steps * hyperparameters.decay_steps_factor))
  lr_schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, polynomial_schedule_fn], boundaries=[warmup_steps])
  return lr_schedule_fn


def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_pa,ams_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """
    Returns:
     (new_optimizer_state, update_fn)
     new_params
     new_model_state
    """
  del current_params_types
  del loss_type
  del eval_results


  ddp = model_state[0]
  assert USE_PYTORCH_DDP

  current_model = current_param_container
  current_model.train()
  optimizer_state['optimizer'].zero_grad()

  with ddp.no_sync():
    logits_batch, new_model_state = workload.model_fn(
        params=current_model,
        augmented_and_preprocessed_input_batch=batch,
        model_state=model_state,
        mode=spec.ForwardPassMode.TRAIN,
        rng=rng,
        update_batch_norm=True)

    label_smoothing = (
        hyperparameters.label_smoothing if hasattr(hyperparameters,
                                                 'label_smoothing') else 0.0)
    if hasattr(hyperparameters, 'grad_clip'):
      grad_clip = hyperparameters.grad_clip
    else:
      grad_clip = None

    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits_batch,
        mask_batch=batch.get('weights'),
        label_smoothing=label_smoothing)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    loss = summed_loss / n_valid_examples
    loss.backward()
    for p in current_model.parameters():
      if p.grad is not None and len(p.grad.size()) >= 2:
        p.grad = approximator(p.grad)
    for p in current_model.parameters():
      dist_nn.all_reduce(p.grad)
    dist_nn.all_reduce(summed_loss)
    dist_nn.all_reduce(n_valid_examples)
    loss = summed_loss / n_valid_examples

  if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(
        current_model.parameters(), max_norm=grad_clip)
  optimizer_state['optimizer'].step()
  optimizer_state['scheduler'].step()

  # Log training metrics - loss, grad_norm, batch_size.
  if global_step <= 100 or global_step % 500 == 0:
    with torch.no_grad():
      parameters = [p for p in current_model.parameters() if p.grad is not None]
      grad_norm = torch.norm(
          torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
    if workload.metrics_logger is not None:
      workload.metrics_logger.append_scalar_metrics(
          {
              'loss': loss.item(),
              'grad_norm': grad_norm.item(),
          }, global_step)
    logging.info('%d) loss = %0.3f, grad_norm = %0.3f',
                 global_step,
                 loss.item(),
                 grad_norm.item())

  return (optimizer_state, current_param_container, new_model_state)


def get_batch_size(workload_name):
  """
    Gets batch size for workload.
    Args: 
      workload_name (str): Valid workload_name values are: "wmt", "ogbg", 
        "criteo1tb", "fastmri", "imagenet_resnet", "imagenet_vit", 
        "librispeech_deepspeech", "librispeech_conformer".
    Returns:
      int: batch_size 
    Raises:
      ValueError: If workload_name is not handled.
    """
  if workload_name == 'criteo1tb':
    return N_GPUS * 262_144
  elif workload_name == 'fastmri':
    return N_GPUS * 32
  elif workload_name == 'imagenet_resnet':
    return N_GPUS * 1024
  elif workload_name == 'imagenet_vit':
    return N_GPUS * 1024
  elif workload_name == 'librispeech_conformer':
    return N_GPUS * 256
  elif workload_name == 'librispeech_deepspeech':
    return N_GPUS * 256
  elif workload_name == 'ogbg':
    return N_GPUS * 512
  elif workload_name == 'wmt':
    return N_GPUS * 128
  elif workload_name == 'mnist':
    return N_GPUS * 16
  elif workload_name =='cifar':
    return N_GPUS * 128
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')


def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   model_state: spec.ModelAuxiliaryState,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.
    Each element of the queue is a batch of training examples and labels.
    Tip:
    If you would just like the next batch from the input queue return next(input_queue).

    Returns:
     batch: next batch of input data
    """
  return next(input_queue)

def approximator(grad: torch.tensor):
  """
  Replace a gradient with a low rank approximation
  For this purposes of this test, we will use
  CP decomposition from tensorly
  rank of 1
  """
  return tl.cp_tensor.cp_to_tensor(
            tl.decomposition.CP(rank=1,
                                tol=0.1,
                                init="random",
                                n_iter_max=5
                                ).fit_transform(grad)
            )
