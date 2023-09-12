import numpy as np
import torch
import math
from scipy.optimize import check_grad

from xgbsurv.models.utils import transform
from xgbsurv.tests.get_data_arrays import get_1d_array, get_2d_array
from xgbsurv.models.eh_final import eh_likelihood, eh_objective


PDF_PREFACTOR = 0.3989424488876037


def eh_calculation(linear_predictor, time, event):
    """Extended hazards loss from original paper Zhong et al."""
    assert isinstance(
        linear_predictor, torch.Tensor
    ), f"<linear_predictor> should be a Tensor, but is {type(linear_predictor)} instead."
    if torch.sum(event) == 0:
        raise RuntimeError("No events detected!")

    n_samples = len(event)

    h = 1.30 * math.pow(n_samples, -0.2)  ## or 1.59*n_samples^(-1/3)
    time = time.view(n_samples, 1)
    event = event.view(n_samples, 1)
    g1 = linear_predictor[:, 0].view(n_samples, 1)
    g2 = linear_predictor[:, 1].view(n_samples, 1)

    # R = g(Xi) + log(Oi)
    R = torch.add(g1, torch.log(time))

    S1 = (event * g2).sum() / n_samples
    S2 = -(event * R).sum() / n_samples

    # Rj - Ri
    rawones = torch.ones(1, n_samples)
    R1 = torch.mm(R, rawones)
    R2 = torch.mm(torch.t(rawones), torch.t(R))
    DR = R1 - R2

    # K[(Rj-Ri)/h]
    x = (DR / h)
    K =PDF_PREFACTOR * torch.exp(-0.5 * torch.pow(x, 2.0))
    Del = torch.mm(event, rawones)
    DelK = Del * K

    # (1/nh) *sum_j eventj * K[(Rj-Ri)/h]
    Dk = torch.sum(DelK, dim=0) / (
        n_samples * h
    )  ## Dk would be zero as learning rate too large!

    # log {(1/nh) * eventj * K[(Rj-Ri)/h]}
    log_Dk = torch.log(Dk)

    S3 = (torch.t(event) * log_Dk).sum() / n_samples

    # Phi((Rj-Ri)/h)
    ncdf = torch.distributions.normal.Normal(
        torch.tensor([0.0]), torch.tensor([1.0])
    ).cdf
    P = ncdf(DR / h)
    L = torch.exp(g2 - g1)
    LL = torch.mm(L, rawones)
    LP_sum = torch.sum(LL * P, dim=0) / n_samples
    Q = torch.log(LP_sum)

    S4 = -(event * Q.view(n_samples, 1)).sum() / n_samples

    S = S1 + S2 + S3 + S4
    S = -S
    return S

def get_eh_gradient(case="default"):
    linear_predictor, time, event = get_2d_array(case)
    linear_predictor_tensor, time_tensor, event_tensor = (
        torch.from_numpy(linear_predictor),
        torch.from_numpy(time),
        torch.from_numpy(event),
    )
    linear_predictor_tensor.requires_grad_()
    eh_loss_torch = eh_calculation(linear_predictor_tensor, time_tensor, event_tensor)
    eh_loss_torch.backward()
    eh_gradient_torch = linear_predictor_tensor.grad
    return eh_gradient_torch.numpy().flatten()


class TestEHGradients:
    abs_tol = 1e-6

    def test_default(self):
        linear_predictor, time, event = get_2d_array("default")
        y = np.vstack((transform(time, event), transform(time, event))).T
        n_events = np.sum(event)
        n_samples = time.shape[0]
        eh_gradient_computed = eh_objective(
                y,
                linear_predictor
            )[0]/n_events
        
        eh_torch_gradient = get_eh_gradient(case="default")
        assert np.allclose(eh_gradient_computed, eh_torch_gradient, atol=self.abs_tol)


    def test_first_five_zero(self):
        linear_predictor, time, event = get_2d_array("first_five_zero")
        y = np.vstack((transform(time, event), transform(time, event))).T
        n_events = np.sum(event)
        n_samples = time.shape[0]
        eh_gradient_computed = eh_objective(
                y,
                linear_predictor
            )[0]/n_events
        
        eh_torch_gradient = get_eh_gradient(case="first_five_zero")
        assert np.allclose(eh_gradient_computed, eh_torch_gradient, atol=self.abs_tol)


    def test_last_five_zero(self):
        linear_predictor, time, event = get_2d_array("last_five_zero")
        y = np.vstack((transform(time, event), transform(time, event))).T
        n_events = np.sum(event)
        n_samples = time.shape[0]
        eh_gradient_computed = eh_objective(
                y,
                linear_predictor
            )[0]/n_events
        
        eh_torch_gradient = get_eh_gradient(case="last_five_zero")
        assert np.allclose(eh_gradient_computed, eh_torch_gradient, atol=self.abs_tol)


    def test_high_event_ratio(self):
        linear_predictor, time, event = get_2d_array("high_event_ratio")
        y = np.vstack((transform(time, event), transform(time, event))).T
        n_events = np.sum(event)
        n_samples = time.shape[0]
        eh_gradient_computed = eh_objective(
                y,
                linear_predictor
            )[0]/n_events
        
        eh_torch_gradient = get_eh_gradient(case="high_event_ratio")
        assert np.allclose(eh_gradient_computed, eh_torch_gradient, atol=self.abs_tol)


    def test_low_event_ratio(self):
        linear_predictor, time, event = get_2d_array("low_event_ratio")
        y = np.vstack((transform(time, event), transform(time, event))).T
        n_events = np.sum(event)
        n_samples = time.shape[0]
        eh_gradient_computed = eh_objective(
                y,
                linear_predictor
            )[0]/n_events
        
        eh_torch_gradient = get_eh_gradient(case="low_event_ratio")
        assert np.allclose(eh_gradient_computed, eh_torch_gradient, atol=self.abs_tol)
