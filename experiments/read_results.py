"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plotting utility for the results with random sampling of distributions within 
the Wasserstein ball. The results are saved in the following CSV files.
Cost:
- double_integrator_beta.csv
- double_integrator_bimodal_gaussian.csv
Constraint violations:
- double_integrator_viol_beta.csv
- double_integrator_viol_bimodal_gaussian.csv

Copyright Andrea Martin (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#colors = ['#3D6BB6', '#94A5C4', '#CCD1DD', '#CFB9B7', '#614947', '#4F2020']
colors = np.array(sns.color_palette(None, n_colors=8))[[0, 1, 2, 3, 4, 7]]


def melt_and_add_category(df):
    # Melt the DataFrame to reshape it
    melted_df = df.melt(id_vars=df.columns[-1], var_name='Method', value_name='Value')
    # Rename the last column to 'Category'
    melted_df.rename(columns={df.columns[-1]: 'Category'}, inplace=True)
    return melted_df


def plot_bars(filename, ylabels):
    # Create scatter plot
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(2, figsize=(10, 6))

    # Plot both cost and violations
    for lab, (ax, affix) in zip(ylabels, zip(axs, ['', '_viol'])):

        # Read the data from the CSV file into a pandas DataFrame
        df = pd.read_csv(filename.split(".")[0] + affix + ".csv", header=None)
        # df = df.drop(columns=1, axis=1)
        # Rename the columns
        df = df.rename(columns={0: "DRInC", 1: "Empirical", 2: "Robust",
                                3: "LQG", 4: "DRLQG",
                                5: "Wasserstein distance"})

        # Round the Wasserstein distance to the nearest 0.02
        # The data should be much closer than that
        for r in df.index:
            df.iloc[r, -1] = 0.02 * round(df.iloc[r, -1] / 0.02)

        # Melt the DataFrame to reshape it
        melted_df = melt_and_add_category(df)

        # Create the box plot using Seaborn
        sns.boxplot(x='Method', y='Value', hue='Category',
                    data=melted_df, ax=ax, palette=colors)

        # Set plot labels and legend
        ax.set_ylabel(lab)
        ax.set_ylim(8 if affix == '' else None, None, auto=True)
        ax.legend(title='Wasserstein distance (ordered)',
                  ncol=round(np.max(df.iloc[:, -1].to_numpy() / 0.02)),
                  columnspacing=1.2)

    # Only set xlabel at the bottom
    axs[0].set_xlabel('')


def plot_scatter(filename, ylabels):
    # Define a threshold value for filtering
    threshold_value = 0.14

    # Define colors and markers for each column
    labels = ['DRInC', 'Empirical', 'Robust', 'LQG', 'DRLQG']
    markers = ['o', '<', 's', '^', 'D']
    ticks = [0.015] + list(np.arange(0.02, 0.14, 0.02))

    # Create scatter plot
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(2, figsize=(10, 6))
    # Plot both cost and violations
    for lab, (ax, affix) in zip(ylabels, zip(axs, ['', '_viol'])):
        # Read data
        df = pd.read_csv(filename.split(".")[0] + affix + ".csv")
        # Filter rows based on the value of the last column
        df = df[df[df.columns[-1]] < threshold_value]
        # Drop the second column
        # df.drop(df.columns[1], axis=1, inplace=True)

        # Plot scatter and trendline
        for i, column in enumerate(df.columns[:-1]):
            sns.scatterplot(x=df[df.columns[-1]], y=df[column], color=colors[i],
                            marker=markers[i], label=labels[i], ax=ax)

            # Add trend line
            sns.regplot(x=df[df.columns[-1]], y=df[column], scatter=False,
                        color=colors[i], order=3, ax=ax,
                        line_kws={'linewidth': 1})

        # Set plot labels and legend
        ax.set_xscale('log')
        ax.set_xticks(ticks)
        ax.set_xticklabels(["{:.2f}".format(tick) for tick in ticks])
        ax.set_ylabel(lab)
        ax.set_xlabel('Wasserstein distance')
        ax.set_ylim(-6, None, auto=True)
        ax.legend(ncol=len(labels), columnspacing=0.8,
                  loc="lower right")

    # Only set xlabel at the bottom
    axs[0].set_xlabel('')
