import numpy as np
import pandas as pd


def tensor_to_numpy(batch_tensor):
    return batch_tensor.cpu().detach().numpy()[0, 0, :, :]

def logits_to_prediction(logits, pred_threshold=0.5):
    pred = (logits > pred_threshold) * 1
    return pred

def sample_logits_and_labels(sample, model, DEVICE):

    data = sample['image']
    data = data[None, :] # add dimension to make it a batch

    target = sample['labels']
    target = target[None, :] # add dimension to make it a batch

    data, target = data.to(DEVICE), target.to(DEVICE)
    output = model(data)

    labels = tensor_to_numpy(target)
    logits = tensor_to_numpy(output)

    return labels, logits

def classification_cases(labels, pred):

    true_pos = (labels == 1) & (pred == 1)
    true_neg = (labels == 0) & (pred == 0)
    false_pos = (labels == 0) & (pred == 1)
    false_neg = (labels == 1) & (pred == 0)

    # check that all pixels are in one of the categories
    n_pixels = np.prod(labels.shape)
    assert true_pos.sum() + true_neg.sum() + false_neg.sum() + false_pos.sum() == n_pixels

    return true_pos, true_neg, false_pos, false_neg

def prediction_metrics(true_pos, true_neg, false_pos, false_neg):
    metrics = {'n_pixels': np.prod(true_pos.shape), 
            'true_pos': true_pos.sum(), 'true_neg': true_neg.sum(),
            'false_pos': false_pos.sum(), 'false_neg': false_neg.sum()}

    metrics_df = pd.DataFrame.from_dict({0: metrics}, orient='index')
    metrics_df['n_building'] = metrics_df['true_pos'] + metrics_df['false_neg']
    metrics_df['building_cover'] = metrics_df['n_building'] / metrics_df['n_pixels']
    metrics_df['n_union'] = metrics_df['true_pos'] + metrics_df['false_pos'] + metrics_df['false_neg']
    metrics_df['jaccard'] = metrics_df['true_pos'] / metrics_df['n_union'] # TODO: case where union is 0
    metrics_df['dice'] = 2*metrics_df['true_pos'] / (2*metrics_df['true_pos'] + metrics_df['false_pos'] + metrics_df['false_neg']) # TODO: case without any building / building prediction
    metrics_df['accuracy'] = (metrics_df['true_pos'] + metrics_df['true_neg']) / metrics_df['n_pixels']

    return metrics_df

def compute_true_false_classifications_for_sample_and_model(sample, model, DEVICE):
    
    # Compute model predictions
    labels, logits = sample_logits_and_labels(sample, model, DEVICE)
    pred = logits_to_prediction(logits, pred_threshold=0.5)
    true_pos, true_neg, false_pos, false_neg = classification_cases(labels, pred)
    
    # Translate into true / false positives / negatives:
    classes = np.zeros(true_pos.shape)
    classes[true_neg] = 0
    classes[false_pos] = 1
    classes[false_neg] = 2
    classes[true_pos] = 3
    
    return classes