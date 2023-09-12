from math import log
import math
import torch
import numpy as np

import pytest
from xgbsurv.models.utils import transform
from xgbsurv.tests.get_data_arrays import get_1d_array
from xgbsurv.models.eh_aft_final import aft_likelihood

PDF_PREFACTOR = 0.3989424488876037

def aft_calculation(linear_predictor, time, event):
    """Accelerated Failure Time Loss."""
    assert isinstance(
        linear_predictor, torch.Tensor
    ), f"<linear_predictor> should be a Tensor, but is {type(linear_predictor)} instead."
    if torch.sum(event) == 0:
        raise RuntimeError("No events detected!")

    n_samples = len(event)

    if linear_predictor.dim() != 2:
        linear_predictor = linear_predictor[:, None]
        # .reshape(n_samples,1)

    h = 1.30 * math.pow(n_samples, -0.2)
    # h 1.304058*math.pow(n_samples,-0.2)  ## 1.304058*n_samples^(-1/5) or 1.587401*math.pow(n_samples,-0.333333) 1.587401*n_samples^(-1/3)
    time = time.view(n_samples, 1)
    event = event.view(n_samples, 1)

    # R = g(Xi) + log(Oi)
    R = torch.add(linear_predictor, torch.log(time))

    # Rj - Ri
    rawones = torch.ones([1, n_samples], dtype=linear_predictor.dtype)
    R1 = torch.mm(R, rawones)
    R2 = torch.mm(torch.t(rawones), torch.t(R))
    DR = R1 - R2

    # K[(Rj-Ri)/h]
    x = (DR / h)
    K =PDF_PREFACTOR * torch.exp(-0.5 * torch.pow(x, 2.0))
    Del = torch.mm(event, rawones)
    DelK = Del * K

    # (1/nh) *sum_j eventj * K[(Rj-Ri)/h]
    Dk = torch.sum(DelK, dim=0) / (n_samples * h)

    # log {(1/nh) * eventj * K[(Rj-Ri)/h]}
    log_Dk = torch.log(Dk)
    A = torch.t(event) * log_Dk / n_samples
    S1 = A.sum()

    ncdf = torch.distributions.normal.Normal(
        torch.tensor([0.0], dtype=linear_predictor.dtype),
        torch.tensor([1.0], dtype=linear_predictor.dtype),
    ).cdf
    P = ncdf(DR / h)
    CDF_sum = torch.sum(P, dim=0) / n_samples
    Q = torch.log(CDF_sum)
    S2 = -(event * Q.view(n_samples, 1)).sum() / n_samples

    S0 = -(event * torch.log(time)).sum() / n_samples

    S = S0 + S1 + S2
    S = -S
    return S


class TestAFTLoss:
    def test_default(self):
        linear_predictor, time, event = get_1d_array("default")

        # convert to torch tensor for external testing
        linear_predictor_tensor, time_tensor, event_tensor = (
            torch.from_numpy(linear_predictor),
            torch.from_numpy(time),
            torch.from_numpy(event),
        )
        aft_formula_computation = aft_calculation(
            linear_predictor_tensor, time_tensor, event_tensor
        )
        
        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        aft_loss = aft_likelihood(y, linear_predictor)/n_events

        assert np.allclose(
            aft_loss, aft_formula_computation, atol=1e-2
        ), f"Computed aft loss is {aft_loss} but formula yields {aft_formula_computation} for default data."

    def test_first_five_zero(self):
        linear_predictor, time, event = get_1d_array("first_five_zero")

        # convert to torch tensor for external testing
        linear_predictor_tensor, time_tensor, event_tensor = (
            torch.from_numpy(linear_predictor),
            torch.from_numpy(time),
            torch.from_numpy(event),
        )
        aft_formula_computation = aft_calculation(
            linear_predictor_tensor, time_tensor, event_tensor
        )

        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        aft_loss = aft_likelihood(y, linear_predictor)/n_events

        assert np.allclose(
            aft_loss, aft_formula_computation, atol=1e-2
        ), f"Computed aft loss is {aft_loss} but formula yields {aft_formula_computation} for edge case: first five zero events."

    def test_last_five_zero(self):
        linear_predictor, time, event = get_1d_array("last_five_zero")

        # convert to torch tensor for external testing
        linear_predictor_tensor, time_tensor, event_tensor = (
            torch.from_numpy(linear_predictor),
            torch.from_numpy(time),
            torch.from_numpy(event),
        )
        aft_formula_computation = aft_calculation(
            linear_predictor_tensor, time_tensor, event_tensor
        )
        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        aft_loss = aft_likelihood(y, linear_predictor)/n_events

        assert np.allclose(
            aft_loss, aft_formula_computation, atol=1e-2
        ), f"Computed aft loss is {aft_loss} but formula yields {aft_formula_computation} for edge case: last five zero events."

    def test_high_event_ratio(self):
        linear_predictor, time, event = get_1d_array("high_event_ratio")

        # convert to torch tensor for external testing
        linear_predictor_tensor, time_tensor, event_tensor = (
            torch.from_numpy(linear_predictor),
            torch.from_numpy(time),
            torch.from_numpy(event),
        )
        aft_formula_computation = aft_calculation(
            linear_predictor_tensor, time_tensor, event_tensor
        )

        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        aft_loss = aft_likelihood(y, linear_predictor)/n_events

        assert np.allclose(
            aft_loss, aft_formula_computation, atol=1e-2
        ), f"Computed aft loss is {aft_loss} but formula yields {aft_formula_computation} for edge case: high event ratio."

    def test_low_event_ratio(self):
        linear_predictor, time, event = get_1d_array("low_event_ratio")

        # convert to torch tensor for external testing
        linear_predictor_tensor, time_tensor, event_tensor = (
            torch.from_numpy(linear_predictor),
            torch.from_numpy(time),
            torch.from_numpy(event),
        )
        aft_formula_computation = aft_calculation(
            linear_predictor_tensor, time_tensor, event_tensor
        )

        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        aft_loss = aft_likelihood(y, linear_predictor)/n_events

        assert np.allclose(
            aft_loss, aft_formula_computation, atol=1e-2
        ), f"Computed aft loss is {aft_loss} but formula yields {aft_formula_computation} for edge case: low event ratio."

    def test_all_events(self):
        linear_predictor, time, event = get_1d_array("all_events")

        # convert to torch tensor for external testing
        linear_predictor_tensor, time_tensor, event_tensor = (
            torch.from_numpy(linear_predictor),
            torch.from_numpy(time),
            torch.from_numpy(event),
        )
        aft_formula_computation = aft_calculation(
            linear_predictor_tensor, time_tensor, event_tensor
        )

        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        aft_loss = aft_likelihood(y, linear_predictor)/n_events

        assert np.allclose(
            aft_loss, aft_formula_computation, atol=1e-2
        ), f"Computed aft loss is {aft_loss} but formula yields {aft_formula_computation} for edge case: all(100%) events."

    def test_no_events(self):
        linear_predictor, time, event = get_1d_array("no_events")
        y = transform(time, event)
        with pytest.raises(RuntimeError) as excinfo:
            aft_likelihood(y, linear_predictor)
        assert "No events detected!" in str(
            excinfo.value
        ), f"Events detected in data. Check data or the function <aft_negative_likelihood> to make sure data is processed correctly."
