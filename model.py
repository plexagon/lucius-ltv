import theano.tensor as tt
import pymc3 as pm
import numpy as np
import pandas as pd
from pymc3.theanof import floatX, intX
import arviz as az


def betaln(a, b):
    return tt.gammaln(a) + tt.gammaln(b) - tt.gammaln(a + b)


def beta_geom_llh(x, a, b):
    return betaln(a + 1, x + b - 1) - betaln(a, b)


def censored_beta_geom_llh(x, a, b):
    return betaln(a, b + x) - betaln(a, b)


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
            value.shape[0] + 1, self.a, self.b)


def fit_sbg_model(cohort_matrix, verbose=False, min_cohort_size=40):
    model = pm.Model()
    with model:
        beta = pm.HalfCauchy('beta', 0.1)
        alpha = pm.HalfCauchy('alpha', 0.1)
        can_fit = False
        for i in range(cohort_matrix.shape[0]):
            effective_data = cohort_matrix.values[i][~pd.isna(cohort_matrix.values[i])]
            if len(effective_data) > 5 and max(effective_data) > min_cohort_size:
                can_fit = True
                # Consider only cohorts with 4 FULL renewal periods and at least 40 ppl
                # (first one is FT->R, last is incomplete -> we need 6 total)
                ShiftedBetaGeometric(
                    f'obs_{i}_cohort',
                    a=alpha,
                    b=beta,
                    observed=np.abs(np.diff(effective_data[:-1])),
                    surviving=effective_data[-2]
                )
        if not can_fit:
            print(
                'Not enough data for fitting, you need at least 1 cohort with 5 full observation periods and >40ppl (excl the first one)')
            raise Exception('Not enough data')
        data = pm.sample(progressbar=verbose, return_inferencedata=True, target_accept=0.9,
                         compute_convergence_checks=False)
        samples = pm.sample_posterior_predictive(data.posterior, progressbar=verbose)
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


def compute_empirical_ltv(cohort_matrix, subscription_price, conversion_to_pay=1, periods=52):
    weighted_cohort = _weighted_cohort_line(cohort_matrix)
    empirical_ltv = np.cumsum(weighted_cohort * subscription_price) * conversion_to_pay
    return empirical_ltv


def compute_predicted_ltv(idata, subscription_price_after_tax, conversion_to_pay=1, periods=52):
    a = idata['posterior']['alpha'][0]
    b = idata['posterior']['beta'][0]
    history = [np.ones_like(a)]
    for t in range(1, periods):
        history.append(history[-1] * sbg_survival_rate(t, a, b))
    q = np.array(history) * subscription_price_after_tax * conversion_to_pay
    if not len(q) == periods:
        print(len(q), periods)
        raise Exception('')
    predicted_low = np.cumsum(np.percentile(q=5, a=q, axis=-1))
    predicted_mid = np.cumsum(np.percentile(q=50, a=q, axis=-1))
    predicted_high = np.cumsum(np.percentile(q=95, a=q, axis=-1))
    return predicted_low, predicted_mid, predicted_high
