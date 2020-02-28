from enum import Enum


class Methods(Enum):
    """
    Recommendation system methods
    """
    collaborative = "collaborative"
    content_based = "content-based"


class ContentBasedModels(Enum):
    """
    Machine Learning models implemented following the content based recommendation system method
    """
    random = "random"
    naive = "naive"
    knn = "knn"
    rf = "rf"
    dnn = "dnn"


class CollaborativeModels(Enum):
    """
    Models implemented using the collaborative technique for recommendation systems
    """
    kmeans = "kmeans"
    pearson = "pearson"


class MetricNames(Enum):
    """
    The name of the metrics used to evaluate the models
    """
    macro_precision = "macro_precision"
    micro_precision = "micro_precision"
    macro_recall = "macro_recall"
    micro_recall = "micro_recall"
    macro_f = "macro_f"
    micro_f = "micro_f"


class MetricCaptions(Enum):
    """
    Metric captions used in visualization
    """
    macro_prec = "macro-prec"
    micro_prec = "micro-prec"
    macro_recall = "macro-recall"
    micro_recall = "micro-recall"
    macro_f = "macro-f"
    micro_f = "micro-f"


class Datasets(Enum):
    """
    The folder names and dataset names from MovieLens.
    Dev dataset is s smaller one for testing purposes - small or latest should be used.
    """
    small = "small"
    latest = "latest"
    dev = "dev"
    ml_dev = "ml-dev"
    ml_latest = "ml-latest"
    ml_latest_small = "ml-latest-small"


class PreprocessKind(Enum):
    """
    Use different dataset (ratings or test_recommendation). Applies only in the content based method
    """
    train = "train"
    recommend = "recommend"


class MetricKind(Enum):
    """
    Use different model for validation or test. When testing the model, the best model from the k-fold cross-validation
    is used
    """
    validation = "validation"
    test = "test"


class Classification(Enum):
    """
    Different classification for the content-based method
    """
    binary = "binary"
    multi = "multi"


class AggregationStrategy(Enum):
    """
    Word embeddings aggregation strategy
    """
    average = "avg"
    max = "max"


class ResultStatus(Enum):
    """
    The status of a results. Possible values: success / failure
    """
    success = "success"
    failure = "failure"
