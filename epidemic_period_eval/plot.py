import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

class Plot:
    def __init__(self):
        pass

    def plot_period(self):
        column_names = ['model','n_weeks_ahead','sensitivity','specificity','precision','npv','accuracy']
        period_df = pd.read_csv('MetricsResult/epidemicPeriod_diff.csv', header=None, names=column_names)

        # create plot for Accuracy in predicting epidemic period
        sns.set(style="white")
        plt.figure(figsize=(10, 6))  # Set the figure size
        lineplot = sns.lineplot(data=period_df, x='n_weeks_ahead', y='accuracy', hue='model', marker='o')
        plt.title('Accuracy in predicting epidemic period')
        plt.xlabel('Number of Weeks Ahead')
        plt.ylabel('Accuracy')
        plt.legend(title_fontsize='10', labelspacing=1.2, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=len(period_df['model'].unique()))
        plt.xticks(range(5))

        # save plot
        plot_result_dir = f"PlotResult"
        if not os.path.exists(plot_result_dir):
            os.makedirs(plot_result_dir)
        plt.savefig(f"{plot_result_dir}/Period.png")


    def plot_period_sensitivity(self):
        column_names = ['model','n_weeks_ahead','sensitivity','specificity','precision','npv','accuracy']
        period_df = pd.read_csv('MetricsResult/epidemicPeriod_diff.csv', header=None, names=column_names)

        df_melted = period_df.melt(id_vars=['model', 'n_weeks_ahead'], value_vars=['sensitivity', 'specificity', 'precision', 'npv', 'accuracy'])

        # create plot for Accuracy in predicting epidemic period
        g = sns.FacetGrid(df_melted, col="n_weeks_ahead", col_wrap=3, height=4)
        custom_palette = ['#FF5733', '#33FF57', '#3357FF', '#FF33FB', '#57FF33', '#FF3357', '#57FFC3', '#C357FF', '#F3FF33', '#333FFF']
        g.map_dataframe(sns.barplot, x='variable', y='value', hue='model', palette=custom_palette, ci=None)
        g.add_legend( loc='lower right')
        g.set_axis_labels("", "Value")
        g.set_titles("Weeks Ahead: {col_name}")

        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
        plt.tight_layout()

        plot_result_dir = f"PlotResult"
        if not os.path.exists(plot_result_dir):
            os.makedirs(plot_result_dir)
        plt.savefig(f"{plot_result_dir}/Period_sensitivity.png")



    def plot_seasonal(self):
        column_names = ['model','n_weeks_ahead','sensitivity','specificity','precision','npv','accuray']
        period_df = pd.read_csv('MetricsResult/epidemicSeasonal_diff.csv', header=None, names=column_names)

        # create plot for Accuracy in predicting epidemic period
        sns.set(style="white")
        plt.figure(figsize=(10, 6))  # Set the figure size
        lineplot = sns.lineplot(data=period_df, x='n_weeks_ahead', y='accuray', hue='model', marker='o')
        # Adding titles and labels
        plt.title('Accuracy in predicting epidemic period season')
        plt.xlabel('Number of Weeks Ahead')
        plt.ylabel('Accuracy')
        # Enhance the legend
        plt.legend(title_fontsize='10', labelspacing=1.2, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=len(period_df['model'].unique()))
        plt.xticks(range(5))
        # save plot
        plot_result_dir = f"PlotResult"
        if not os.path.exists(plot_result_dir):
            os.makedirs(plot_result_dir)
        plt.savefig(f"{plot_result_dir}/Season.png")

    def plot_peak(self,mode):
        column_names = ["model", "n_weeks_ahead", "strict_peak_week_accuracy","strict_peak_rate_accuracy","loose_peak_week_accuracy","loose_peak_rate_accuracy"]
        period_df = pd.read_csv('MetricsResult/epidemicPeak_diff.csv', header=None, names=column_names)

        if mode == "strict":
            y_week = 'strict_peak_week_accuracy'
            y_rate = 'strict_peak_rate_accuracy'
        else:
            y_week = 'loose_peak_week_accuracy'
            y_rate = 'loose_peak_rate_accuracy'

        # create plot for Accuracy in predicting peak week and peak date
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Plot 1: peak week date
        sns.lineplot(data=period_df, x='n_weeks_ahead', y=y_week, hue='model', marker='o', ax=ax1)
        ax1.set_title('Accuracy in predicting epidemic peak week with strict standard')
        ax1.set_xlabel(' ')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(5))

        # Plot 2: peak rate
        sns.lineplot(data=period_df, x='n_weeks_ahead', y=y_rate, hue='model', marker='o', ax=ax2)
        ax2.set_title('Accuracy in predicting epidemic peak rate with strict standard')
        ax2.set_xlabel(' ')
        ax2.set_xticks(range(5))

        # Enhance the legend on the first subplot, hide the legend on the second subplot
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', title_fontsize='13', labelspacing=1.2, bbox_to_anchor=(0.5, 0.05), ncol=len(period_df['model'].unique()))
        ax1.get_legend().remove()
        ax2.get_legend().remove()

        # Display the plot
        plt.tight_layout()

        # save plot
        plot_result_dir = f"PlotResult"
        if not os.path.exists(plot_result_dir):
            os.makedirs(plot_result_dir)
        plt.savefig(f"{plot_result_dir}/Peak_week_rate_{mode}.png")






if __name__ == "__main__":
    p = Plot()
    p.plot_period_sensitivity()
    p.plot_seasonal()
    p.plot_period()
    p.plot_peak(mode = "strict")
    p.plot_peak(mode = "loose")

