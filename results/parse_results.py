from os import getcwd
from os.path import join


def calc_avg_fold_metrics_content_based(properties, directory):
    output_folder = properties["output_folder"]
    base_path = join(getcwd(), output_folder, directory)
    fold_num = properties["cross-validation"]
    metric_names = ["macro-precision", "micro-precision", "macro-recall", "micro-recall", "macro-f", "micro-f"]
    metrics = {}
    for metric_name in metric_names:
        metrics[metric_name] = {}
    best_models = {}
    for fold in range(fold_num):
        fold_dir = "fold_{}".format(fold)
        fold_dir_path = join(base_path, fold_dir)
        for model in properties["models"]["content-based"]:
            if model not in metrics[metric_names[0]]:
                best_models[model] = -1
                for metric_name in metric_names:
                    metrics[metric_name][model] = []
            filename = "Result_test_{}.txt".format(model)
            file_path = join(fold_dir_path, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Macro Precision:"):
                        macro_precision = float(line.split(":")[1].strip())
                        metrics[metric_names[0]][model].append(macro_precision)
                    elif line.startswith("Micro Precision:"):
                        micro_precision = float(line.split(":")[1].strip())
                        metrics[metric_names[1]][model].append(micro_precision)
                    elif line.startswith("Macro Recall:"):
                        macro_recall = float(line.split(":")[1].strip())
                        metrics[metric_names[2]][model].append(macro_recall)
                    elif line.startswith("Micro Recall:"):
                        micro_recall = float(line.split(":")[1].strip())
                        metrics[metric_names[3]][model].append(micro_recall)
                    elif line.startswith("Macro F-Measure:"):
                        macro_f = float(line.split(":")[1].strip())
                        metrics[metric_names[4]][model].append(macro_f)
                    elif line.startswith("Micro F-Measure:"):
                        micro_f = float(line.split(":")[1].strip())
                        metrics[metric_names[5]][model].append(micro_f)
            best_models[model] = (metrics[properties["metric_best_model"]][model], fold)
    avg_metrics = {}
    for model in properties["models"]["content-based"]:
        avg_metrics[model] = {}
        for metric_name in metric_names:
            metric = metrics[metric_name][model]
            avg = sum(metric) / len(metric)
            avg_metrics[model][metric_name] = avg
            print("Showing results for model {} with configuration {}".format(model, properties[model]))
            print("Average results for {} for model {} is: {}".format(metric_name, model, avg))
    return avg_metrics, best_models
