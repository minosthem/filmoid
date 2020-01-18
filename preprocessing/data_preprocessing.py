import pandas as pd


def read_csv(files):
    datasets = {}
    for name, file in files.items():
        datasets[name] = pd.read_csv(file)
    return datasets


def preprocessing_collaborative(datasets):
    pass


def preprocessing_content_based(datasets):
    vectors = [[]]
    pass