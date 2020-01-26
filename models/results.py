from os.path import join, exists
from os import mkdir, getcwd


def write_results_to_file(properties, fold, classifier, conf_matrix, results):
    """
    Writes micro/macro precision, recall and F-Measure for current fold into a file

    Args:
        properties(dict): properties loaded from yaml file. Used so as to get the output folder
        fold (str): the name of the folder to be created to store the results. For cross-validation is the fold with the
        index while for the testing the name is test_results.
        classifier (str): the name of the classifier
        conf_matrix (ConfusionMatrix): the confusion matrix of the current fold or of the test dataset
        results (dict): dictionary with keys the classifiers' names and values a list with the metrics per fold

    Returns:
        dict: the results dictionary
    """
    macro_precision, micro_precision, macro_recall, micro_recall, macro_f, micro_f = calc_results(properties,
                                                                                                  conf_matrix)
    measure_tuples = macro_precision, micro_precision, macro_recall, micro_recall, macro_f, micro_f
    if classifier not in results:
        results[classifier] = []
    results[classifier].append(measure_tuples)
    output_folder_path = join(getcwd(), properties["output_folder"],
                              "results_{}_{}".format(properties["dataset"], properties["classification"]))
    if not exists(output_folder_path):
        mkdir(output_folder_path)
    fold_path = join(output_folder_path, fold)
    if not exists(fold_path):
        mkdir(fold_path)
    filename = "Result_test_" + classifier + ".txt"
    result_file = join(fold_path, filename)
    with open(result_file, 'w') as f:
        conf_matrix = "Confusion Matrix: " + str(conf_matrix) + "\n"
        macro_precision = "Macro Precision: " + str(macro_precision) + "\n"
        micro_precision = "Micro Precision: " + str(micro_precision) + "\n"
        macro_recall = "Macro Recall: " + str(macro_recall) + "\n"
        micro_recall = "Micro Recall: " + str(micro_recall) + "\n"
        macro_f = "Macro F-Measure: " + str(macro_f) + "\n"
        micro_f = "Micro F-Measure: " + str(micro_f) + "\n"
        f.write(conf_matrix)
        f.write(macro_precision)
        f.write(micro_precision)
        f.write(macro_recall)
        f.write(micro_recall)
        f.write(macro_f)
        f.write(micro_f)
    return results


def calc_results(properties, confusion_matrix):
    """
    Function that calculates micro/macro precision, recall and F-Measure of the specific fold

    Args:
        properties(dict): properties loaded from yaml file. Used so as to get the output folder
        confusion_matrix (ConfusionMatrix): the confusion matrix of the current fold or of the test dataset

    Returns:
        tuple: a tuple with micro/macro average for precision, recall and F-Measure
    """
    labels = [0, 1] if properties["classification"] == "binary" else [1, 2, 3, 4, 5]
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    list_precisions = []
    list_recalls = []
    list_fmeasures = []
    for label in labels:
        true_positive = confusion_matrix[label][label] if properties["classification"] == "binary" else \
            confusion_matrix[label - 1][label - 1]
        false_positive = 0
        false_negative = 0
        for other_label in labels:
            if other_label != label:
                fp_position = confusion_matrix[other_label][label] if properties["classification"] == "binary" else \
                    confusion_matrix[other_label - 1][label - 1]
                fn_position = confusion_matrix[label][other_label] if properties["classification"] == "binary" else \
                    confusion_matrix[label - 1][other_label - 1]
                false_positive = false_positive + fp_position
                false_negative = false_negative + fn_position
        print(label, " true_positive ", true_positive, " false_negative ", false_negative, " false_positive ",
              false_positive)
        if (true_positive + false_positive) == 0:
            precision = 0
        else:
            precision = true_positive / (true_positive + false_positive)
        if (true_positive + false_negative) == 0:
            recall = 0
        else:
            recall = true_positive / (true_positive + false_negative)
        if (precision + recall) == 0:
            fmeasure = 0
        else:
            fmeasure = 2 * ((precision * recall) / (precision + recall))

        total_true_positives = total_true_positives + true_positive
        total_false_negatives = total_false_negatives + false_negative
        total_false_positives = total_false_positives + false_positive
        list_precisions.append(precision)
        list_recalls.append(recall)
        list_fmeasures.append(fmeasure)
    macro_precision = sum(list_precisions) / len(list_precisions)
    micro_precision = total_true_positives / (total_true_positives + total_false_positives)
    macro_recall = sum(list_recalls) / len(list_recalls)
    micro_recall = total_true_positives / (total_true_positives + total_false_negatives)
    macro_f = 2 * ((macro_precision * macro_recall) / (macro_precision + macro_recall))
    micro_f = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall))

    print("macro_precision: ", macro_precision, " macro_recall: ", macro_recall, " macro_f: ", macro_f,
          " micro_precision: ", micro_precision, " micro_recall: ", micro_recall, " micro_f: ", micro_f)
    return macro_precision, micro_precision, macro_recall, micro_recall, macro_f, micro_f


def visualize():
    pass
