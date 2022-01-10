from typing import List, Any, Tuple

import theano
import theano.tensor as tt
import pymc3 as pm
import numpy as np
import pandas as pd
import arviz as az

from pymc3.theanof import floatX, intX
from scipy import stats
from scipy.special import beta as beta_f


def betaln(a, b):
    return tt.gammaln(a) + tt.gammaln(b) - tt.gammaln(a + b)


def beta_geom_llh(x, a, b):
    return betaln(a + 1, x + b - 1) - betaln(a, b)


def censored_beta_geom_llh(x, a, b):
    return betaln(a, b + x) - betaln(a, b)


def censored_beta_geom_pdf(x, a, b):
    return beta_f(a, b + x) / beta_f(a, b)


def beta_geom_pdf(x, a, b):
    return beta_f(a + 1, x + b - 1) / beta_f(a, b)


class SPShiftedBetaGeometric(stats.rv_continuous):
    def _pdf(self, x, t_max, a, b):
        if x >= t_max:
            return censored_beta_geom_pdf(x, a, b)
        return beta_geom_pdf(x, a, b)


class ShiftedBetaGeometric(pm.Discrete):
    """
    Pymc implementation of:
    Fader, Peter and Hardie, Bruce, How to Project Customer Retention (May 2006).
    Available at SSRN: https://ssrn.com/abstract=801145.
    or http://dx.doi.org/10.2139/ssrn.801145
    """

    def __init__(self, a, b, surviving, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = tt.as_tensor_variable(floatX(a))
        self.b = tt.as_tensor_variable(floatX(b))
        self.surviving = tt.as_tensor_variable(intX(surviving))
        self.mode = 1

    def logp(self, value):
        idx = tt.arange(1, value.shape[0] + 1, 1, dtype='int32')
        return tt.sum(value * beta_geom_llh(idx, self.a, self.b)) + self.surviving * censored_beta_geom_llh(
            value.shape[0], self.a, self.b)


def fit_sbg_model(cohort_matrix: pd.DataFrame, price: float = 1, all_users: List[int] = None,
                  min_cohort_size: int = 40, periods: int = 52, true_alpha=None, true_beta=None,
                  **sampler_kwargs: Any) -> Tuple[
    az.InferenceData, pm.Model]:
    """
    Fits the SBG model for customer LTV
    :param cohort_matrix: the retention matrix by cohort, see `generate_synthetic_cohort_matrix` for an example
    :param price: the price users pay for each subscription renewal, by default unitary price
    :param periods: the number of periods to forecast
    :param all_users: optional, adds a conversion rate component to the model if you have unconverted
        users using your product (e.g. "free users" or "free trials")
    :param sampler_kwargs: extra kwargs to be passed to the pymc sampler
    :param min_cohort_size: minimum size for a cohort to be considered in the fitting process
    :return: az.InferenceData of the sbg model
    """

    model = pm.Model()
    with model:
        alpha = pm.HalfCauchy('alpha', 0.05)  # pm.HalfFlat('alpha')#pm.HalfCauchy('beta', 1)
        beta = pm.HalfCauchy('beta', 0.05)  # pm.HalfFlat('beta')

        can_fit = False
        starting_payers = []
        for i in range(cohort_matrix.shape[0]):
            effective_data = cohort_matrix.values[i][~pd.isna(cohort_matrix.values[i])]
            starting_payers.append(effective_data[0])
            if len(effective_data) >= 4 and max(effective_data) > min_cohort_size:
                can_fit = True
                # Consider only cohorts with 4 FULL renewal periods and at least 40 (or min_cohort_size) ppl
                ShiftedBetaGeometric(
                    f'obs_{i}_cohort',
                    a=alpha,
                    b=beta,
                    observed=np.abs(np.diff(effective_data[:-1])),
                    surviving=effective_data[-2]
                )

        if not all_users:
            conversion_rate = pm.Deterministic('conversion_rate', tt.ones(1))
            pm.Deterministic('conversion_rate_by_cohort', tt.ones(cohort_matrix.shape[0]))

        else:
            if not all_users:
                all_users = starting_payers
            if not len(all_users) == cohort_matrix.shape[0]:
                raise Exception('If provided, free trials must be provided for each cohort')
            conversion_rate = pm.Uniform('conversion_rate', 0, 1)

            conversion_rate_alpha = pm.HalfCauchy('cohort_conversion_rate_alpha', 0.5)

            cohort_conversion_rate = pm.Beta('conversion_rate_by_cohort',
                                             alpha=conversion_rate_alpha,
                                             beta=conversion_rate_alpha * (1 / conversion_rate - 1),
                                             shape=len(all_users))

            pm.Binomial('conversion_rate_obs', n=all_users, p=cohort_conversion_rate, observed=starting_payers)

        survival_probabilities, _ = theano.scan(
            fn=lambda t: tt.switch(tt.gt(t, 1), sbg_survival_rate(t - 1, alpha, beta), tt.ones(1)),
            sequences=tt.arange(start=1, stop=periods + 1, step=1),
        )

        survival_curve = tt.cumprod(survival_probabilities)
        pm.Deterministic('survival_curve', survival_curve)
        pm.Deterministic('ltv', tt.cumsum(survival_curve * conversion_rate * price))

        if true_alpha and true_beta:
            survival_probabilities, _ = theano.scan(
                fn=lambda t: tt.switch(tt.gt(t, 1), sbg_survival_rate(t - 1, true_alpha, true_beta), tt.ones(1)),
                sequences=tt.arange(start=1, stop=periods + 1, step=1),
            )

            survival_curve = tt.cumprod(survival_probabilities)
            pm.Deterministic('true_survival_curve', survival_curve)
            pm.Deterministic('true_ltv', tt.cumsum(survival_curve * conversion_rate * price))

        if not can_fit:
            raise Exception(
                f'Not enough data for fitting, you need 1 cohort with 4 observation periods and >{min_cohort_size}ppl')

        data = pm.sample(return_inferencedata=True, **sampler_kwargs)

        samples = pm.fast_sample_posterior_predictive(data.posterior)

    return az.concat(data, az.from_pymc3(posterior_predictive=samples)), model


def sbg_survival_rate(x, a, b):
    """
    Probability that a customer alive at time x-1 is still alive at time x
    """
    return (b + x - 1) / (a + b + x - 1)


def _weighted_cohort_line(cohort_matrix):
    z = {}
    for idx, cohort_line in enumerate(cohort_matrix.values):
        for cohort_period, population in enumerate(cohort_line[idx:]):
            if not pd.isna(population):
                z.setdefault(cohort_period, [])
                z[cohort_period].append((population / cohort_line[~pd.isna(cohort_line)].max(), population))
    out = []
    for k, v in z.items():
        tw = 0
        tv = []
        for pct, weight in v:
            tv.append(pct * weight)
            tw += weight
        out.append(sum(tv) / tw)
    return np.array(out)


def compute_empirical_ltv(cohort_matrix: pd.DataFrame, price: float = 1, conversion_rate: float = 1,
                          periods: int = 52) -> Tuple[np.array, np.array, np.array]:
    """
    :param conversion_rate: optional, if the model was fit using it.
    :param cohort_matrix: the retention matrix by cohort, see `generate_synthetic_cohort_matrix` for an example
    :param price: the price your users pays for each renewal, remember to consider taxes
    :param periods: the maximum number of computed periods returned by the function
    :return: Returns three np.arrays containing lower, median and upper credible intervals for the empirical LTV
    """
    weighted_cohort = _weighted_cohort_line(cohort_matrix)
    empirical_ltv = np.cumsum(weighted_cohort * price) * conversion_rate
    return empirical_ltv[:periods]
