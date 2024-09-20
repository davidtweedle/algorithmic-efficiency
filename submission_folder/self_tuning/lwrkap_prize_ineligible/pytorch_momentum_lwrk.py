"""Submission file for a SGD with HeavyBall momentum optimizer in PyTorch."""

import torch
from torch.optim.lr_scheduler import LambdaLR

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from reference_algorithms.target_setting_algorithms.data_selection import \
    data_selection  # pylint: disable=unused-import
from reference_algorithms.target_setting_algorithms.get_batch_size import \
    get_batch_size  # pylint: disable=unused-import
from reference_algorithms.target_setting_algorithms.jax_momentum import \
    create_lr_schedule_fn
from reference_algorithms.target_setting_algorithms.pytorch_submission_base import \
    update_params  # pylint: disable=unused-import

from .low_rank_comm import LowRankApproximationState, simple_lwrk_hook

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()
lrka_state = None


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a Nesterov optimizer and a learning rate schedule."""
  del model_state
  del rng

  global lrka_state
  lrka_state_args = {
          'global_step': 0,
          'matrix_approximation_rank': hyperparameters.matrix_approximation_rank,
          'n_gpus': N_GPUS,
          'batch_tensors_with_same_shape': True
          }

  if lrka_state is None:
    lrka_state = LowRankApproximationState(**lrka_state_args)
    model_params.register_comm_hook(lrka_state, simple_lwrk_hook)
    # register the communication hook which will
    # approximate the gradient on each gpu
    # then all reduce the results
    # cannot change hook between svd and low rank on same run
  else:
    lrka_state.__setstate__(lrka_state_args)
    # if this has been run using num_tuning_trials > 1
    # then we will need to re use the previous communication hook
  # Create optimizer.
  optimizer_state = {
      'optimizer':
          torch.optim.SGD(
              model_params.parameters(),
              lr=hyperparameters.learning_rate,
              momentum=1 - hyperparameters.one_minus_beta1,
              weight_decay=hyperparameters.weight_decay,
              nesterov=False),
  }

  # Create learning rate schedule.
  target_setting_step_hint = int(0.75 * workload.step_hint)
  lr_schedule_fn = create_lr_schedule_fn(target_setting_step_hint,
                                         hyperparameters)

  # PyTorch's LambdaLR expects the lr_lambda fn to return a factor which will
  # be multiplied with the base lr, so we have to divide by it here.
  def _lr_lambda(step: int) -> float:
    return lr_schedule_fn(step).item() / hyperparameters.learning_rate

  optimizer_state['scheduler'] = LambdaLR(
      optimizer_state['optimizer'], lr_lambda=_lr_lambda)

  return optimizer_state

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
  elif workload_name =='cifar':
    return 128
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')

