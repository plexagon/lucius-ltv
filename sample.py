import matplotlib.pyplot as plt
import numpy as np
import arviz as az

from lucius_ltv.utils import generate_synthetic_cohort_matrix
from lucius_ltv.model import fit_sbg_model, compute_empirical_ltv
from lucius_ltv.plots import plot_ltv, plot_cohort_matrix_retention, plot_conversion_rate


def main():
    cohorts = 8
    max_obs_period = 10
    true_alpha = 0.7
    true_beta = 1.8
    true_conversion_rate = 1.3
    cohort_sizes = 100

    cohort_sizes = np.random.normal(cohort_sizes, 0., size=cohorts).astype(int)
    alphas = np.random.normal(true_alpha, 0., size=cohorts)
    betas = np.random.normal(true_beta, 0., size=cohorts)
    conversion_rate = np.random.normal(true_conversion_rate, 0.1, size=cohorts)  # Conversion to Free Trial

    cohort_matrix = generate_synthetic_cohort_matrix(
        cohort_sizes=list(cohort_sizes),
        alphas=list(alphas),
        betas=list(betas),
        max_obs_period=max_obs_period
    )

    inference_data, model = fit_sbg_model(
        cohort_matrix,
        all_users=list((cohort_sizes * conversion_rate).astype(int)),  # Optional, if you have free trials
        progressbar=True,
        periods=52,
        true_alpha=true_alpha,
        true_beta=true_beta,
        target_accept=0.9
    )

    conversion_rate = np.median(inference_data['posterior']['conversion_rate'][0])

    empirical_ltv = compute_empirical_ltv(cohort_matrix=cohort_matrix, price=1, conversion_rate=conversion_rate)

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_ltv(empirical_ltv, inference_data=inference_data, ax=ax)
    fig.suptitle(f'Lifetime value')
    ax.legend()
    ax.grid()

    plot_cohort_matrix_retention(cohort_matrix, 'Cohort Retention')

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_conversion_rate(inference_data, ax=ax)
    fig.suptitle(f'Conversion Rate')
    ax.grid()

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 12))
    az.plot_posterior(inference_data, var_names=("alpha",),
                      ref_val=true_alpha,
                      ax=ax1)
    az.plot_posterior(inference_data, var_names=("beta",),
                      ref_val=true_beta,
                      ax=ax2)
    az.plot_posterior(inference_data, var_names=("conversion_rate",),
                      ref_val=1/true_conversion_rate,
                      ax=ax3)
    fig.suptitle(f'True v Recovered values')

    plt.show()


if __name__ == '__main__':
    main()
