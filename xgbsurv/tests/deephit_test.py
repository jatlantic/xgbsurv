from xgbsurv.models.deephit_pycox_final import deephit_loss1_pycox, pycox_gradient_no_gamma, \
pycox_second_deriv_no_gamma
from xgbsurv.models.utils import transform_back
import numpy as np
import jax.numpy as jnp
from jax import grad, hessian
import pytest


# deephit loss 1

def test_deephit_loss_1(deephit_data):
    time, event, target, phi = deephit_data
    loss_own = deephit_loss1_pycox(target, phi)
    # loss from pycox: nll_pmf(*,reduction='sum')
    loss_precalculated = 31.7249 # enter value here
    assert np.allclose(loss_own, loss_precalculated)

#time, event, target, phi = deephit_data()
#loss_own = deephit_loss1_pycox(target, phi)



# gradient comparison with Jax, requires Jax loss
def relu(x):
    return (jnp.maximum(0,x))

def pycox_loss_no_gamma_jax(phi, y):

    time, event = transform_back(y)
    # pad phi as in pycox
    pad = jnp.zeros_like(phi[:,:1])
    phi = jnp.concatenate((phi, pad),axis=1)
    # create durations index
    bins = jnp.unique(time)
    idx_durations_part1 = (jnp.digitize(time, bins)-1).reshape(-1, 1)
    idx_durations_part2 = (jnp.digitize(time, bins)).reshape(-1, 1) # double check the -1
    n,k = phi.shape
    part1 = event.reshape(n, 1)*jnp.take_along_axis(phi, idx_durations_part1, axis=1)
    phi_exp = jnp.exp(phi)
    cumsum = jnp.cumsum(phi_exp, axis=1)
    part2 = (1.0-event).reshape(n, 1)*jnp.log(cumsum[:,[-1]]-jnp.take_along_axis(cumsum, idx_durations_part1, axis=1)) # 
    part3 = -jnp.log(cumsum[:,[-1]])
    result = jnp.sum(-(part1+part2+part3))
    return result


# grad
def test_deephit_gradient(deephit_data):
    time, event, target, phi = deephit_data
    target = target.reshape(phi.shape)
    target = target[:,0]
    grad_own = pycox_gradient_no_gamma(target, phi)
    # loss from pycox: nll_pmf(*,reduction='sum')
    # there is also the loss version without gamma
    grad_precalculated = grad(pycox_loss_no_gamma_jax, allow_int=True)(phi, target)
    assert grad_own.shape==grad_precalculated.shape
    assert np.allclose(grad_own, grad_precalculated)


# hessian
def test_deephit_hessian(deephit_data):
    time, event, target, phi = deephit_data
    target = target.reshape(phi.shape)
    target = target[:,0]
    hessian_own = pycox_second_deriv_no_gamma(target, phi)
    print(hessian_own)
    pyc_loss_hessian_jax = hessian(pycox_loss_no_gamma_jax)(phi, target)
    print(pyc_loss_hessian_jax.shape)
    n = phi.shape[1]
    d = {}
    for i in range(n):
        d[i] = np.diag(pyc_loss_hessian_jax[:,i,:,i])

    hessian_precalculated = np.array(list(d.values())).T
    print(hessian_precalculated)
    assert np.allclose(hessian_own, hessian_precalculated)


# deephit loss 2: rank loss

def test_deephit_loss_2(deephit_data):
    time, event, target, phi = deephit_data
    loss_own = deephit_loss1_pycox(target, phi)
    # loss from pycox: nll_pmf(*,reduction='sum')
    loss_precalculated = 31.7249 # enter value here
    assert np.allclose(loss_own, loss_precalculated)