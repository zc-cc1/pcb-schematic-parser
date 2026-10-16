"""Explicit image-only BN estimation for controlled training diagnostics."""
import torch
from torch import nn


def fixed_batches(length, batch_size, reverse=False):
    """Reverse whole batches, preserving their membership and singleton tail."""
    if length<1 or batch_size<1:
        raise ValueError("Need positive dataset length and batch size")
    batches=[list(range(start,min(length,start+batch_size))) for start in range(0,length,batch_size)]
    return list(reversed(batches)) if reverse else batches


def reestimate_bn_buffers(model, image_batches, momentum=None):
    """Reset only BN buffers; learned parameters and module modes remain intact.

    Every input must be an NCHW image tensor. Other modules are in eval mode
    during estimation, so dropout cannot change the estimates. On failure the
    original buffers are restored. No optimizer or labels enter this function.
    """
    if momentum is not None and not 0<momentum<=1:
        raise ValueError("Momentum must be None or in (0,1]")
    bns=[m for m in model.modules() if isinstance(m,nn.BatchNorm2d)]
    if not bns or any(not m.track_running_stats for m in bns):
        raise ValueError("Need BatchNorm2d with tracked running statistics")
    iterator=iter(image_batches)
    try:first=next(iterator)
    except StopIteration:raise ValueError("No training image batches") from None
    modes={m:m.training for m in model.modules()}
    saved=[(m,m.momentum,m.running_mean.clone(),m.running_var.clone(),m.num_batches_tracked.clone()) for m in bns]
    def stream():
        yield first
        yield from iterator
    batches=images=0
    try:
        model.eval()
        for m in bns:m.reset_running_stats();m.momentum=momentum;m.train()
        with torch.inference_mode():
            for batch in stream():
                if not isinstance(batch,torch.Tensor) or batch.ndim!=4 or batch.shape[0]<1:
                    raise ValueError("Expected a nonempty NCHW image tensor, without targets")
                model(batch);batches+=1;images+=batch.shape[0]
    except BaseException:
        for m,_,mean,var,count in saved:
            m.running_mean.copy_(mean);m.running_var.copy_(var);m.num_batches_tracked.copy_(count)
        raise
    finally:
        for m,old_momentum,*_ in saved:m.momentum=old_momentum
        for m,training in modes.items():m.training=training
    return dict(BN_layers=len(bns),batches=batches,image_crops=images,momentum=momentum,
        gradient_updates=0,dropout_disabled_during_estimation=True)
