"""Accumulate native per-batch BN moments in float64 at fixed learned weights."""
import torch
from torch import nn
from .unet_bn_statistics import reestimate_bn_buffers


def reestimate_bn_buffers_precise(model,image_batches):
    """Average the same native moments with less accumulation-order error.

    Momentum1 exposes each batch's native running mean/unbiased variance. BN
    still normalizes each batch exactly as in training. Only the short vectors
    are accumulated in float64; activations retain their original precision.
    This is an unweighted average of batch moments, not population variance.
    """
    bns=[m for m in model.modules() if isinstance(m,nn.BatchNorm2d)]
    totals={m:[torch.zeros_like(m.running_mean,dtype=torch.float64),torch.zeros_like(m.running_var,dtype=torch.float64)] for m in bns if m.track_running_stats}
    def observed():
        for batch in image_batches:
            yield batch
            # The caller has completed model(batch) before advancing us.
            for m,(mean,var) in totals.items():
                mean.add_(m.running_mean.to(torch.float64));var.add_(m.running_var.to(torch.float64))
    result=reestimate_bn_buffers(model,observed(),momentum=1.)
    with torch.no_grad():
        for m,(mean,var) in totals.items():
            m.running_mean.copy_((mean/result['batches']).to(m.running_mean.dtype))
            m.running_var.copy_((var/result['batches']).to(m.running_var.dtype))
    return result|dict(momentum=None,accumulator_dtype='float64',native_batch_moments=True,
        activations_precision_unchanged=True,population_variance=False)
