import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import arviz as az


def plot_ltv(empirical_ltv, inference_data=None, hdi_prob=.95, extra_label_text='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
    if inference_data:
        az.plot_hdi(x=np.arange(52), hdi_prob=hdi_prob, smooth=False, y=inference_data['posterior']['ltv'], ax=ax)
        curve_m = np.median(np.median(inference_data['posterior']['ltv'], axis=0), axis=0)
        ax.plot(curve_m, 'k', linestyle='dashed', alpha=0.5,
                label=f'Median ltv: {curve_m[len(curve_m) - 1].round(2)}')

    if 'true_ltv' in inference_data['posterior'].keys():
        curve_m = np.median(np.median(inference_data['posterior']['true_ltv'], axis=0), axis=0)
        ax.plot(curve_m, 'k', alpha=0.5,
                label=f'True ltv: {curve_m[len(curve_m) - 1].round(2)}')

    ax.plot(empirical_ltv, 'o',
            label=f'{extra_label_text}Empirical @{len(empirical_ltv)} periods: {empirical_ltv[len(empirical_ltv) - 1].round(2)}')
    return ax


def plot_conversion_rate(inference_data, hdi_prob=.95, extra_label_text='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))

    conversion_rate_by_cohort = inference_data['posterior']['conversion_rate_by_cohort']
    az.plot_hdi(x=np.arange(conversion_rate_by_cohort.shape[-1]), hdi_prob=hdi_prob, smooth=False,
                y=conversion_rate_by_cohort, ax=ax)
    curve_m = np.median(np.median(inference_data['posterior']['conversion_rate_by_cohort'], axis=0), axis=0)
    ax.plot(curve_m, 'k-', alpha=0.3,
            label=f'{extra_label_text}Median Conversion Rate: {np.median(curve_m).round(2)}')
    ax.plot(curve_m, 'ko', alpha=0.8,
            label=f'{extra_label_text}Median Conversion Rate: {np.median(curve_m).round(2)}')


def plot_cohort_matrix_retention(cohort_matrix, title=''):
    cohort_size = cohort_matrix.max(axis=1)
    retention_matrix = cohort_matrix.divide(cohort_size, axis=0)

    with sns.axes_style("white"):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharey='all', gridspec_kw={'width_ratios': [1, 11]})
        sns.heatmap(retention_matrix.iloc[:, :-1],
                    mask=retention_matrix.iloc[:, :-1].isnull(),
                    annot=True,
                    fmt='.0%',
                    cmap='RdYlGn',
                    ax=ax[1])
        ax[1].set_title(title, fontsize=12)
        ax[1].set(xlabel='# of periods',
                  ylabel='')

        # cohort size
        cohort_size_df = pd.DataFrame(cohort_size).rename(columns={0: 'cohort_size'})
        white_cmap = mcolors.ListedColormap(['white'])
        sns.heatmap(cohort_size_df,
                    annot=True,
                    cbar=False,
                    fmt='g',
                    cmap=white_cmap,
                    ax=ax[0])

        fig.tight_layout()
