import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import os

# Add parent directory to path to import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    STEM_LOGIC_SUBJECTS,
    LIFE_SCIENCES_SUBJECTS,
    HUMANITIES_SOCIAL_SUBJECTS
)


def get_subjects():
    # Use imported subject category constants
    return STEM_LOGIC_SUBJECTS, LIFE_SCIENCES_SUBJECTS, HUMANITIES_SOCIAL_SUBJECTS


if __name__ == "__main__":
    fp_rc0 = "results/suppobj_rc-1.csv"
    fp_rc1 = "results/suppobj_rcTrue.csv"
    df_incorrect = pd.read_csv(fp_rc0)
    df_correct = pd.read_csv(fp_rc1)

    stem_logic_subjects, life_sciences_subjects, humanities_social_subjects = get_subjects()
    # df_correct = df_correct[df_correct["subject"].isin(humanities_social_subjects)]
    # df_incorrect = df_incorrect[df_incorrect["subject"].isin(humanities_social_subjects)]

    hashes_incorrect = df_incorrect["hash_key"].unique()
    hashes_correct = df_correct["hash_key"].unique()
    # print(f"{len(hashes_incorrect)=}")
    # print(f"{len(hashes_correct)=}")
    # quit()
    overlapping_hashes = set(hashes_incorrect) & set(hashes_correct)
    df_incorrect = df_incorrect[df_incorrect["hash_key"].isin(overlapping_hashes)]
    df_correct = df_correct[df_correct["hash_key"].isin(overlapping_hashes)]
    # print(f"{len(df_incorrect)=}")
    # print(f"{len(df_correct)=}")
    # quit()

    df_incorrect.sort_index(inplace=True)
    df_correct.sort_index(inplace=True)

    kl_keys = [col for col in df_incorrect.columns if col.startswith("total_kl_")]
    kl_keys.sort()

    skip = 5
    for i in range(1, 120, skip):
        key = f"total_kl_k{i}"
        keys = [f"total_kl_k{j}" for j in range(i, i + skip)]
        s_incorrect = df_incorrect[keys].mean(axis=1).values * 1000
        s_correct = df_correct[keys].mean(axis=1).values * 1000

        # s_incorrect = df_incorrect[key].values * 1000
        # s_correct = df_correct[key].values * 1000
        assert len(s_incorrect) == len(s_correct)
        nan_idxs = np.isnan(s_incorrect) | np.isnan(s_correct)
        s_incorrect = s_incorrect[~nan_idxs]
        s_correct = s_correct[~nan_idxs]
        assert len(s_incorrect) == len(s_correct)

        # M_both = np.mean(s_incorrect) + np.mean(s_correct)
        # SD_both = np.std(s_incorrect.tolist() + s_correct.tolist())

        M_incorrect = np.mean(s_incorrect)
        SE_incorrect = np.std(s_incorrect) / np.sqrt(len(s_incorrect))
        M_correct = np.mean(s_correct)
        SE_correct = np.std(s_correct) / np.sqrt(len(s_correct))
        t, p = stats.ttest_rel(s_incorrect, s_correct)
        print(
            f"{key} | Correct M = {M_correct:.1f} ({SE_correct:.1f}), Incorrect M = {M_incorrect:.1f} ({SE_incorrect:.1f}) | t = {t:.3f} | p = {p:.3f}"
        )
