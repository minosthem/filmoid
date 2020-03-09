from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd

from utils import utils


def conv_collaborative(properties):
    user_ids = utils.load_from_pickle(properties["output_folder"], "user_ids.pickle_{}".
                                      format(properties["dataset"]))
    for user_id in user_ids:
        path = join(utils.app_dir, properties["output_folder"],
                    "results_pearson_{}_20-similar".format(properties["dataset"]))
        filename = join(path, "Predictions_{}.csv".format(user_id))
        df = pd.read_csv(filename)
        del df['Unnamed: 0']
        df['rating'] = (df['rating'] <= 3).astype(int)
        df['prediction'] = (df['prediction'] <= 3).astype(int)
        df.to_csv(filename, sep=',')


def visualize_classifier(properties, models):
    path = join(utils.app_dir, properties["output_folder"], "results")
    captions = ["Experiment1", "Experiment2", "Experiment3"]
    for model in models:
        model_path = join(path, model)
        df = pd.read_csv(join(model_path, "Results.csv"), index_col=0, header=0)
        utils.visualize(df=df, output_folder=properties["output_folder"], results_folder="results", folder_name=model,
                        filename="microf_{}.png".format(model), captions=captions)
    best_df = pd.read_csv(join(path, "Best_Results.csv"), index_col=0, header=0)
    captions = ["Micro F-Measure"]
    utils.visualize(df=best_df, output_folder=properties["output_folder"], results_folder="results", folder_name=None,
                        filename="microf_best.png", captions=captions)


def get_micro_f(filename):
    df = pd.read_csv(filename, index_col=0, header=0)
    return df.loc[5, "result"]


def visualize_grouped_bar_chart(folder_path, models):
    total_df = pd.DataFrame(columns=["classifier", "Experiment1", "Experiment2", "Experiment3"])
    row = 0
    for model in models:
        path = join(folder_path, model)
        files = listdir(path)
        microfs = []
        for file in files:
            if file.startswith("Exp1"):
                microfs.append(get_micro_f(filename=join(path, file)))
            elif file.startswith("Exp2"):
                microfs.append(get_micro_f(filename=join(path, file)))
            else:
                microfs.append(get_micro_f(filename=join(path, file)))
        total_df.loc[row] = [model, microfs[0], microfs[1], microfs[2]]
        row += 1
        # plot(folder_path, total_df, model=model)
    total_df.to_csv(join(folder_path, "Results.csv"), sep=",")
    plot(folder_path, total_df)
    best_df = pd.DataFrame(columns=["classifier", "Micro F-Measure"])
    row_counter = 0
    for index, row in total_df.iterrows():
        classifier, micro1, micro2, micro3 = row
        best = max([micro1, micro2, micro3])
        best_df.loc[row_counter] = [classifier, best]
        row_counter += 1
    best_df.to_csv(join(folder_path, "Best_Results.csv"), sep=",")


def plot(folder_path, total_df, model=None):
    # Setting the positions and width for the bars
    pos = list(range(len(total_df["Experiment1"])))
    width = 0.10
    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10, 5))
    # Create a bar with Experiment 1 data,
    # in position pos,
    plt.bar(pos,
            # using df['Experiment1'] data,
            total_df["Experiment1"],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#566D7E',
            # with label the first value in classifier
            label=total_df["classifier"][0])
    # Create a bar with Experiment 2 data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos],
            # using df["Experiment2"] data,
            total_df["Experiment2"],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#C7A317',
            # with label the second value in classifier
            label=total_df["classifier"][0])
    # Create a bar with Experiment 3 data,
    # in position pos + some width buffer,
    plt.bar([p + width * 2 for p in pos],
            # using df["Experiment3"] data,
            total_df["Experiment3"],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#7E587E',
            # with label the third value in classifier
            label=total_df["classifier"][0])
    # Set the y axis label
    ax.set_ylabel("Micro F-Measure")
    # Set the chart's title
    ax.set_title("Micro F-Measure for all models & experiments")
    # Set the position of the x ticks
    ax.set_xticks([p + 1.5 * width for p in pos])
    # Set the labels for the x ticks
    ax.set_xticklabels(total_df["classifier"])
    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos) - width, max(pos) + width * 4)
    microfs = list(total_df["Experiment1"])
    microfs = microfs + list(total_df["Experiment2"])
    microfs = microfs + list(total_df["Experiment3"])
    plt.ylim([0, max(microfs) + 0.3])
    # Adding the legend and showing the plot
    plt.legend(["Experiment1", "Experiment2", "Experiment3"], loc='upper left')
    plt.grid()
    filename = "total_plot_{}.png".format(model) if model else "total_plot.png"
    plt.savefig(join(folder_path, filename))
    plt.close('all')
    # plt.show()


if __name__ == '__main__':
    program_properties = utils.load_properties()
    # conv_collaborative(utils.load_properties())
    folder = join(utils.app_dir, program_properties["output_folder"], "results")
    if program_properties["models"]["content-based"] and program_properties["models"]["collaborative"]:
        models_list = program_properties["models"]["content-based"] + program_properties["models"]["collaborative"]
    elif program_properties["models"]["content-based"]:
        models_list = program_properties["models"]["content-based"]
    else:
        models_list = program_properties["models"]["collaborative"]
    # visualize_grouped_bar_chart(folder_path=folder, models=models_list)
    visualize_classifier(properties=program_properties, models=models_list)
