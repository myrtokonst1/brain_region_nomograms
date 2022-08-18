import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.regression.linear_model import OLS
from tabulate import tabulate

from constants.data_paths import data_path
from constants.ukb_table_column_names import participant_id, latest_age_cn, intercranial_cn, sex_cn

PGS = pd.read_csv(f'{data_path}/PGS_AV.all_score', delim_whitespace=True)
thresholds = ["1e-05", "0.0001", "0.001", "0.01", "0.05", "0.1", "0.2", "0.4", "0.5", "0.75", "1"]
slope = 'Slope'
x_range = 'Range'
p_value = 'P-Value'
r_squared = 'R-Squared'

column_names = ['Threshold', slope, x_range, p_value, r_squared]


def perform_ols_prs_regression(df, brain_region):
    # Merge both
    Dataset = pd.merge(PGS, df, left_on='IID', right_on=participant_id)
    Dataset = Dataset[Dataset[latest_age_cn].notna()]
    # Put the data into the correct X/Y structures
    Dataset["ones"] = np.ones(len(Dataset))

    R2 = np.zeros(11)
    PV = np.ones(11)

    for volume in brain_region:
        i = 0
        Y = Dataset[[volume.get_column_name()]]

        for thresh in thresholds:
            X = Dataset[[thresh, intercranial_cn, sex_cn, latest_age_cn, "ones"]]
            reg_res = OLS(Y, X).fit()
            PV[i] = reg_res.pvalues[thresh]
            R2[i] = reg_res.rsquared

            i = i+1

        print(PV)
        print(R2)

        # Plotting
        plt.figure()
        plt.title(f"R2 for models of PGS and covariates for {volume}")
        plt.ylim([0.3,0.5])
        plt.ylabel("R2")
        plt.xlabel("Threshold")
        plt.bar(thresholds, R2,facecolor='#B2C1DC', edgecolor='#32415D', linewidth=0.8)
        plt.savefig(f'saved_graphs/R2_for_PGS_{volume.get_name()}_top.png',
                    dpi=600)

        plt.show()


def perform_prs_regression(df, sex, brain_region):
    df["ones"] = np.ones(len(df))

    number_of_thresholds = len(thresholds)

    slopes = np.ones(number_of_thresholds)
    ranges = np.ones(number_of_thresholds)
    p_values = np.ones(number_of_thresholds)
    r_squareds = np.ones(number_of_thresholds)

    for i in range(number_of_thresholds):
        threshold = thresholds[i]
        X = df[[threshold,
                intercranial_cn, sex_cn,
                latest_age_cn,
                "ones",
                "Genetic.PC.1","Genetic.PC.2","Genetic.PC.3",
               "Genetic.PC.4","Genetic.PC.5","Genetic.PC.6","Genetic.PC.7",
               "Genetic.PC.8","Genetic.PC.9","Genetic.PC.10"
            ]]
        Y = df[brain_region.get_column_name()]

        reg_res = OLS(Y, X).fit()

        slopes[i] =  np.round(reg_res.params[threshold], decimals=2)
        ranges[i] =  np.round(np.max(df[threshold]) - np.min(df[threshold]), decimals=2)
        p_values[i] =  reg_res.pvalues[threshold]
        r_squareds[i] = np.round(reg_res.rsquared, decimals=2)

    plt.figure()
    plt.title(f"R2 for models of PGS and covariates for {brain_region}")
    plt.ylim([np.min(r_squareds) - 0.01, np.max(r_squareds) + 0.01])
    plt.ylabel("R2")
    plt.xlabel("Threshold")
    plt.bar(thresholds, r_squareds, facecolor='#B2C1DC', edgecolor='#32415D', linewidth=0.8)
    plt.savefig(f'saved_graphs/R2_for_PGS_{brain_region.get_name()}_top_TEST.png',
                dpi=600)

    plt.show()

    # create and print table with thresholds and data and print for latex
    transition_matrix = np.matrix([thresholds, slopes, ranges, p_values, r_squareds]).T
    # table = tabulate(transition_matrix, column_names)
    table = pd.DataFrame(transition_matrix,
                 columns=column_names)
    print(sex, brain_region)
    print(table.to_latex(index=False))

    # return pd.DataFrame.from_dict(snps_for_brain_region)

