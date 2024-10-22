"""
Submission file for a Pytorch optimizer with low rank approximation of gradients
SGD with HeavyBall momentum code originally from: 
github.com/mlcommons/algorithmic-efficiency
"""

from typing import Callable, Dict, Iterator, List, Tuple

from absl import logging
import optax
import torch
import torch.distributed.nn as dist_nn
from torch.optim.lr_scheduler import LambdaLR

from algorithmic_efficiency import spec
from algorithmic_efficiency.spec import TrainingCompleteError
from algorithmic_efficiency.pytorch_utils import pytorch_setup

from .lowrankapproximation.pytorch.low_rank_comm import \
        LowRankApproximationState, lwrk_hook, simple_lwrk_hook

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()
lrka_state = None
# need to keep global lrka_state
# The reason is that the implementation of cifar
# resets the parameters on each tuning run
# instead of deleting the model on each run
# which seems to be the case for all other implementations
# of the models


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """
  Creates a Nesterov optimizer and a learning rate schedule.
  Attaches a low rank communication hook to the DDP model. 
  """
  del model_state
  del rng

  global lrka_state
  lrka_state_args = {
          'global_step': 0,
          'matrix_approximation_rank': hyperparameters.matrix_approximation_rank,
          'n_gpus': N_GPUS,
          'batch_tensors_with_same_shape': False,
          'num_iter_svd': hyperparameters.num_iter_svd,
          'num_errs': 0
          }
  if lrka_state is None:
    # if this is the first run, initialize the lrka state and attach
    # it to the DDP model
    lrka_state = LowRankApproximationState(**lrka_state_args)
    model_params.register_comm_hook(lrka_state, simple_lwrk_hook)
  else:
    # this is not the first run
    # we must try to attach the comm hook to the model
    # if it fails it is because a hook is already attached
    # which happens when tuning the cifar model
    lrka_state.__setstate__(lrka_state_args)
    try:
      model_params.register_comm_hook(lrka_state, simple_lwrk_hook)
    except RuntimeError as err:
      if err.args[0] == 'register_comm_hook or register_builtin_comm_hook can only be called once.':
        pass
      else:
        raise

  # Create optimizer.
  optimizer_state = {
      'optimizer':
          torch.optim.SGD(
              model_params.parameters(),
              lr=hyperparameters.learning_rate,
              momentum=1.0 - hyperparameters.one_minus_beta1,
              weight_decay=hyperparameters.weight_decay,
              nesterov=False),
  }

  # Create learning rate schedule.
  lr_schedule_fn = create_lr_schedule_fn(workload.step_hint, hyperparameters)

  # PyTorch's LambdaLR expects the lr_lambda fn to return a factor which will
  # be multiplied with the base lr, so we have to divide by it here.
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
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del eval_results

  current_model = current_param_container
  current_model.train()
  optimizer_state['optimizer'].zero_grad()

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
  n_valid_examples.requires_grad = False
  torch.distributed.all_reduce(n_valid_examples)
  loss = summed_loss / n_valid_examples
  # still must divide by n_gpus after all_reduce
  # because writing grad = uv^T
  # will normalize

  # all reducing of gradients is handled in communication hook
  loss.backward()

  if lrka_state.num_errs > 10 * N_GPUS:
    raise TrainingCompleteError


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
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
    return 1024
  elif workload_name == 'imagenet_resnet_silu':
    return 512
  elif workload_name == 'imagenet_resnet_gelu':
    return 512
  elif workload_name == 'imagenet_vit':
    return 1024
  elif workload_name == 'librispeech_conformer':
    return 256
  elif workload_name == 'librispeech_deepspeech':
    return 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 128
  elif workload_name == 'mnist':
    return 16
  elif workload_name == 'cifar':
    return 128
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
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  batch = next(input_queue)
  return batch
