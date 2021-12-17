import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


def plot_empirical_ltv(empirical_ltv, predicted_ltv_lmh=None, label='', ax=None):
    if ax == None:
        fig, ax = plt.subplots()
    if predicted_ltv_lmh is not None:
        curve_l, curve_m, curve_h = predicted_ltv_lmh
        err = ((curve_h[len(curve_h) - 1].round(2) - curve_l[len(curve_l) - 1].round(2)) / 2) if curve_l[
                                                                                                     len(curve_l) - 1].round(
            2) > 0 else 0
        ax.plot(curve_l, 'r.', alpha=0.5)
        ax.plot(curve_m, 'k', linestyle='dashed', alpha=0.5,
                label=f'Median ltv: {curve_m[len(curve_m) - 1].round(2)} +/- {np.round(err, 2)}')
        ax.plot(curve_h, 'r.', alpha=0.5)
        ax.fill_between(range(len(curve_h)), curve_h, curve_l, alpha=0.05)
    ax.plot(empirical_ltv, 'o',
            label=f'{label} Empirical @{len(empirical_ltv)} periods: {empirical_ltv[len(empirical_ltv) - 1].round(2)}')
    return ax


def plot_cohort_matrix(cohort_matrix, title=''):
    cohort_size = cohort_matrix.max(axis=1)
    retention_matrix = cohort_matrix.divide(cohort_size, axis=0)

    with sns.axes_style("white"):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharey=True, gridspec_kw={'width_ratios': [1, 11]})
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
