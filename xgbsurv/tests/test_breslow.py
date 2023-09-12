from math import log

import numpy as np
import numpy.typing as npt

import pytest
from xgbsurv.models.utils import transform, transform_back
from xgbsurv.tests.get_data_arrays import get_1d_array
from xgbsurv.models.breslow_final import breslow_likelihood


def breslow_calculation(linear_predictor, time, event):
    if np.sum(event) == 0:
        raise RuntimeError("No events detected!")
    sorted_ix = np.argsort(time)
    linear_predictor_sorted = linear_predictor[sorted_ix]
    linear_predictor_sorted_exp = np.exp(linear_predictor_sorted)
    time_sorted = time[sorted_ix]
    event_sorted = event[sorted_ix].astype(int)
    ll = np.sum(linear_predictor_sorted[event_sorted.astype(bool)])
    previous_time = 0.0
    risk_set = np.sum(linear_predictor_sorted_exp)
    breslow_sum = 0.0
    breslow_count = 0.0

    for i in range(sorted_ix.shape[0]):
        current_linear_predictor = linear_predictor_sorted_exp[i]
        current_time = time_sorted[i]
        current_event = event_sorted[i]
        if current_time == previous_time:
            breslow_count += current_event
            breslow_sum += current_linear_predictor
        else:
            ll -= breslow_count * log(risk_set)
            risk_set -= breslow_sum
            breslow_count = current_event
            breslow_sum = current_linear_predictor
        previous_time = current_time

    if breslow_count:
        ll -= breslow_count * log(risk_set)

    return -ll / sorted_ix.shape[0]

def breslow_calculation_old(linear_predictor, time, event):
    """Breslow loss Moeschberger page 259."""
    nominator = []
    denominator = []
    for idx, t in enumerate(np.unique(time[event.astype(bool)])):
        nominator.append(np.exp(np.sum(np.where(t==time, linear_predictor, 0))))
    riskset = (np.outer(time,time)<=np.square(time)).astype(int)
    linear_predictor_exp = np.exp(linear_predictor)
    riskset = riskset*linear_predictor_exp
    uni, idx, counts = np.unique(time[event.astype(bool)], return_index=True, return_counts=True)
    denominator = np.sum(riskset[event.astype(bool)], axis=1)[idx]
    return -np.log(np.prod(nominator/(denominator**counts)))/time.shape[0]

class TestBreslowLoss:
    def test_default(self):
        linear_predictor, time, event = get_1d_array("default")

        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        
        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        breslow_loss = breslow_likelihood(y, linear_predictor)*n_events/n_samples

        assert np.allclose(
            breslow_loss, breslow_formula_computation, atol=1e-2
        ), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for default data."

    def test_first_five_zero(self):
        linear_predictor, time, event = get_1d_array("first_five_zero")
        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)

        breslow_loss = breslow_likelihood(y, linear_predictor)*n_events/n_samples


        assert np.allclose(
            breslow_loss, breslow_formula_computation, atol=1e-2
        ), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: first five zero events."

    def test_last_five_zero(self):
        linear_predictor, time, event = get_1d_array("last_five_zero")

        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        breslow_loss = breslow_likelihood(y, linear_predictor)*n_events/n_samples


        assert np.allclose(
            breslow_loss, breslow_formula_computation, atol=1e-2
        ), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: last five zero events."

    def test_high_event_ratio(self):
        linear_predictor, time, event = get_1d_array("high_event_ratio")

        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        breslow_loss = breslow_likelihood(y, linear_predictor)*n_events/n_samples


        assert np.allclose(
            breslow_loss, breslow_formula_computation, atol=1e-2
        ), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: high event ratio."

    def test_low_event_ratio(self):
        linear_predictor, time, event = get_1d_array("low_event_ratio")

        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        breslow_loss = breslow_likelihood(y, linear_predictor)*n_events/n_samples


        assert np.allclose(
            breslow_loss, breslow_formula_computation, atol=1e-2
        ), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: low event ratio."

    def test_all_events(self):
        linear_predictor, time, event = get_1d_array("all_events")

        breslow_formula_computation = breslow_calculation(linear_predictor, time, event)
        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        breslow_loss = breslow_likelihood(y, linear_predictor)*n_events/n_samples


        assert np.allclose(
            breslow_loss, breslow_formula_computation, atol=1e-2
        ), f"Computed Breslow loss is {breslow_loss} but formula yields {breslow_formula_computation} for edge case: all(100%) events."

    def test_no_events(self):
        linear_predictor, time, event = get_1d_array("no_events")
        y = transform(time, event)
        with pytest.raises(RuntimeError) as excinfo:
            breslow_likelihood(y, linear_predictor)
        assert "No events detected!" in str(
            excinfo.value
        ), "Events detected in data. Check data or the function <breslow_likelihood> to make sure data is processed correctly."
