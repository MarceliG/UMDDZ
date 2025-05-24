import os


class FilePath:
    current_path = os.getcwd()
    # directories
    data_path = os.path.join(current_path, "data")
    datasets = os.path.join(data_path, "datasets")
    datasets_skin_cancer = os.path.join(datasets, "datasets_skin_cancer")
    dataset_preprocessed_skin_cancer = os.path.join(datasets, "processed_skin_cancer")
    models = os.path.join(data_path, "models")
    model_vit_skin_cancer = os.path.join(models, "vit_skin_cancer")
    plots = os.path.join(data_path, "plots")
    results = os.path.join(data_path, "results")
    evaluation = os.path.join(results, "evaluation")
    predictions = os.path.join(results, "predictions")
    classification_report = os.path.join(results, "classification_reports")

    # files
    classification_report_path = os.path.join(classification_report, "classification_report.csv")

    # Create neccesery folders
    for directory in [
        data_path,
        models,
        model_vit_skin_cancer,
        datasets_skin_cancer,
        dataset_preprocessed_skin_cancer,
        plots,
        results,
        evaluation,
        predictions,
        classification_report,
    ]:
        if not os.path.exists(directory):
            os.makedirs(directory)
