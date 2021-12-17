import matplotlib as plt

from utils import generate_synthetic_cohort_matrix
from model import fit_sbg_model, compute_predicted_ltv, compute_empirical_ltv
from plots import plot_empirical_ltv, plot_cohort_matrix


def main():
    cohort_matrix = generate_synthetic_cohort_matrix(
        cohort_sizes=[100, 80, 120, 130, 85, 68],
        alphas=[1.2, 1.1, 1.3, 1.25, 1.18, 0.93],
        betas=[2.2, 2.1, 2.3, 2.25, 2.18, 2.4],
        max_obs_period=10
    )

    idata, model = fit_sbg_model(cohort_matrix)

    predicted_ltv_lmh = compute_predicted_ltv(idata, 9.99)
    empirical_ltv = compute_empirical_ltv(cohort_matrix, 9.99)
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_empirical_ltv(empirical_ltv, predicted_ltv_lmh=predicted_ltv_lmh, ax=ax)
    ax.grid()

    plot_cohort_matrix(cohort_matrix)


if __name__ == '__main__':
    main()
