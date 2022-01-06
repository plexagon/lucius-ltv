from typing import List, Union

import numpy as np
import pandas as pd


def generate_synthetic_cohort_matrix(cohort_sizes: List[int], alphas: List[float], betas: List[float],
                                     max_obs_period: Union[int, float] = float('inf')) -> pd.DataFrame:
    cohorts = len(cohort_sizes)
    assert cohorts < max_obs_period

    cohort_matrix = []
    for cohort_number in range(cohorts):
        rand_data = np.random.geometric(
            np.random.beta(alphas[cohort_number],
                           betas[cohort_number],
                           size=cohort_sizes[cohort_number])
        )
        cohort = {
            i: 0
            for i in range(
                cohort_number + 1,
                min(max_obs_period, max(rand_data)) + 1
            )
        }
        for individual_death in rand_data:
            for alive_period in range(cohort_number + 1, min(max_obs_period, individual_death + cohort_number) + 1):
                cohort[alive_period] += 1
        cohort_matrix.append(cohort)

    return pd.DataFrame(cohort_matrix)

