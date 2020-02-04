from enum import Enum


class Methods(Enum):
    collaborative = "collaborative"
    content_based = "content-based"


class ContentBasedModels(Enum):
    random = "random"
    naive = "naive"
    knn = "knn"
    rf = "rf"
    dnn = "dnn"


class CollaborativeModels(Enum):
    kmeans = "kmeans"


class MetricNames(Enum):
    macro_precision = "macro_precision"
    micro_precision = "micro_precision"
    macro_recall = "macro_recall"
    micro_recall = "micro_recall"
    macro_f = "macro_f"
    micro_f = "micro_f"


class MetricCaptions(Enum):
    macro_prec = "macro-prec"
    micro_prec = "micro-prec"
    macro_recall = "macro-recall"
    micro_recall = "micro-recall"
    macro_f = "macro-f"
    micro_f = "micro-f"


class Datasets(Enum):
    small = "small"
    latest = "latest"
    dev = "dev"
    ml_dev = "ml-dev"
    ml_latest = "ml-latest"
    ml_latest_small = "ml-latest-small"


class PreprocessKind(Enum):
    train = "train"
    recommend = "recommend"


class MetricKind(Enum):
    validation = "validation"
    test = "test"


class Classification(Enum):
    binary = "binary"
    multi = "multi"


class AggregationStrategy(Enum):
    average = "avg"
    max = "max"
