from os.path import join
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


if __name__ == '__main__':
    conv_collaborative(utils.load_properties())
