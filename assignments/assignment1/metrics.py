def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    fp = 0
    fn = 0
    tp = 0
    tn = 0

    for pred_y, y in zip(prediction, ground_truth):
        if y:
            if pred_y: tp += 1
            else: fp += 1
        else:
            if pred_y: fn += 1
            else: tn += 1

    eps = 1e-9 # for numerical stability, taken from https://github.com/fastai/fastai/blob/ba64dd4bc4a5f9e3ea9294b4cf2e7c4928d70483/fastai/metrics.py#L18
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / len(prediction)

    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return (prediction == ground_truth).sum() / len(prediction)
