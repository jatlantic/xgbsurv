import numpy as np
from scipy.optimize import check_grad

from xgbsurv.models.utils import transform
from xgbsurv.tests.get_data_arrays import get_1d_array
from xgbsurv.models.eh_ah_final import ah_likelihood, ah_objective


class TestAFTGradients:
    tolerance = 1e-6

    def test_default(self):
        linear_predictor, time, event = get_1d_array("default")
        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        x = linear_predictor
        diff = check_grad(
            lambda x: ah_likelihood(y, x)/n_events,
            lambda x: ah_objective(y, x)[0]/n_events,
            linear_predictor,
        )
        assert diff < self.tolerance

    def test_first_five_zero(self):
        linear_predictor, time, event = get_1d_array("first_five_zero")
        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        x = linear_predictor
        diff = check_grad(
            lambda x: ah_likelihood(y, x)/n_events,
            lambda x: ah_objective(y, x)[0]/n_events,
            linear_predictor,
        )
        assert diff < self.tolerance

    def test_last_five_zero(self):
        linear_predictor, time, event = get_1d_array("last_five_zero")
        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        x = linear_predictor
        diff = check_grad(
            lambda x: ah_likelihood(y, x)/n_events,
            lambda x: ah_objective(y, x)[0]/n_events,
            linear_predictor,
        )
        assert diff < self.tolerance

    def test_high_event_ratio(self):
        linear_predictor, time, event = get_1d_array("high_event_ratio")
        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        x = linear_predictor
        diff = check_grad(
            lambda x: ah_likelihood(y, x)/n_events,
            lambda x: ah_objective(y, x)[0]/n_events,
            linear_predictor,
        )
        assert diff < self.tolerance

    def test_low_event_ratio(self):
        linear_predictor, time, event = get_1d_array("low_event_ratio")
        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        x = linear_predictor
        diff = check_grad(
            lambda x: ah_likelihood(y, x)/n_events,
            lambda x: ah_objective(y, x)[0]/n_events,
            linear_predictor,
        )
        assert diff < self.tolerance

    def test_all_events(self):
        linear_predictor, time, event = get_1d_array("all_events")
        y = transform(time, event)
        n_events = np.sum(event)
        n_samples = time.shape[0]
        x = linear_predictor
        diff = check_grad(
            lambda x: ah_likelihood(y, x)/n_events,
            lambda x: ah_objective(y, x)[0]/n_events,
            linear_predictor,
        )
        assert diff < self.tolerance
