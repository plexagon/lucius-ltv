from typing import List, Any, Tuple

import pytensor
import pytensor.tensor as tt
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

from pymc import floatX, intX
from scipy import stats
from scipy.special import beta as beta_f

from pytensor.tensor.random.op import RandomVariable


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


class SPShiftedBetaGeometric(stats.rv_discrete):
    def _argcheck(self, a, b, t_max):
        return (a > 0) & (b > 0) & (t_max > 0)

    def _pmf(self, x, a, b, t_max):
        return np.where(x == 0, 0, np.where(x >= t_max, censored_beta_geom_pdf(x, a, b), beta_geom_pdf(x, a, b)))


class ShiftedBetaGeometricRV(RandomVariable):
    name: str = "ShiftedBetaGeometricRV"
    ndim_supp: int = 0
    ndims_params: List[int] = [0, 0, 0]
    dtype: str = "int64"
    _print_name: Tuple[str, str] = ("SBG", "\\operatorname{SBG}")

    @classmethod
    def rng_fn(cls, rng, a, b, t_max, size) -> np.ndarray:
        return SPShiftedBetaGeometric().rvs(a, b, t_max, random_state=rng, size=size)


sbgrv = ShiftedBetaGeometricRV()


class ShiftedBetaGeometric(pm.Discrete):
    """
    Pymc implementation of:
    Fader, Peter and Hardie, Bruce, How to Project Customer Retention (May 2006).
    Available at SSRN: https://ssrn.com/abstract=801145.
    or http://dx.doi.org/10.2139/ssrn.801145
    """

    rv_op = sbgrv

    @classmethod
    def dist(cls, a, b, surviving, *args, **kwargs):
        a = tt.as_tensor_variable(floatX(a))
        b = tt.as_tensor_variable(floatX(b))
        surviving = tt.as_tensor_variable(intX(surviving))
        return super().dist([a, b, surviving], *args, **kwargs)

    def logp(value, a, b, surviving):
        idx = tt.arange(1, value.shape[0] + 1, 1, dtype='int32')
        return tt.sum(value * beta_geom_llh(idx, a, b)) + surviving * censored_beta_geom_llh(
            value.shape[0], a, b)


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
        alpha = pm.HalfCauchy('alpha', 0.1)
        beta = pm.HalfCauchy('beta', 0.1)

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

        survivals = [1]
        for i in range(1, periods):
            survivals.append(sbg_survival_rate(i, alpha, beta))

        survival_curve = tt.cumprod(survivals)
        pm.Deterministic('survival_curve', survival_curve)
        pm.Deterministic('ltv', tt.cumsum(survival_curve * conversion_rate * price))

        if true_alpha and true_beta:
            true_survivals = [1]
            for i in range(1, periods):
                true_survivals.append(sbg_survival_rate(i, true_alpha, true_beta))

            true_survival_curve = tt.cumprod(true_survivals)
            pm.Deterministic('true_survival_curve', true_survival_curve)
            pm.Deterministic('true_ltv', tt.cumsum(true_survival_curve * conversion_rate * price))

        if not can_fit:
            raise Exception(
                f'Not enough data for fitting, you need 1 cohort with 4 observation periods and >{min_cohort_size}ppl')

        data = pm.sample(return_inferencedata=True, **sampler_kwargs)
        data.extend(pm.sample_posterior_predictive(data))

    return data, model


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
