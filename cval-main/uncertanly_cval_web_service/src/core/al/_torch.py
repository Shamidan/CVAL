import collections
import math
from typing import List

import torch

from uncertanly_cval_web_service.src.core.models import BBoxScores, DetectionSamplingOnPremise


def mean(x):
    return sum(x) / len(x)


def least_confidence(prob_dist, num_labels=2):
    if len(prob_dist) == 0:
        prob_dist = torch.full((1, num_labels), 1 / num_labels)
    simple_least_conf = torch.max(prob_dist, dim=1).values
    normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels - 1))
    return normalized_least_conf


def new_entropy(prob_dist, num_labels, w):
    if len(prob_dist) == 0:
        prob_dist = torch.full((1, num_labels), 1 / num_labels)
    log_probs = w * prob_dist * torch.log2(prob_dist)
    raw_entropy = -torch.sum(log_probs, dim=1)
    normalized_entropy = raw_entropy / math.log2(num_labels)
    return normalized_entropy


def margin_confidence(prob_dist, num_labels=2, w=None):
    if len(prob_dist) == 0:
        prob_dist = torch.full((1, num_labels), 1 / num_labels)
    sorted_probs, _ = torch.sort(prob_dist, descending=True, dim=1)
    difference = sorted_probs[:, 0] - sorted_probs[:, 1]
    margin_conf = 1 - difference
    return margin_conf


def ratio_confidence(prob_dist, num_labels=2, w=None):
    if len(prob_dist) == 0:
        prob_dist = torch.full((1, num_labels), 1 / num_labels)
    sorted_probs, _ = torch.sort(prob_dist, descending=True, dim=1)
    ratio_conf = sorted_probs[:, 1] / sorted_probs[:, 0]
    return ratio_conf


def p_2_custom(prob_dist, num_labels=2, w=None):
    if len(prob_dist) == 0:
        prob_dist = torch.full((1, num_labels), 1 / num_labels)
    return 2 - 1 * prob_dist[0]


def probability(prob_dist, num_labels, w=None):
    if len(prob_dist) == 0:
        prob_dist = torch.tensor([[0, 1]], dtype=torch.float32)
    return prob_dist[:, 0]


bbox_selection_policy = collections.namedtuple('bbox_selection_policy', 'func name')
bbox_min = bbox_selection_policy(func=min, name='min')
bbox_max = bbox_selection_policy(func=max, name='max')
bbox_mean = bbox_selection_policy(func=mean, name='mean')
bbox_sum = bbox_selection_policy(func=sum, name='sum')

selection_strategy = collections.namedtuple('selection_strategy', 'func name')
s_probability = selection_strategy(func=probability, name='probability')
s_least = selection_strategy(func=least_confidence, name='least')
s_entropy = selection_strategy(func=new_entropy, name='entropy')
s_margin = selection_strategy(func=margin_confidence, name='margin')
s_ratio = selection_strategy(func=ratio_confidence, name='ratio')
s_custom = selection_strategy(func=p_2_custom, name='cval_custom')

sorting = collections.namedtuple('sorting', 'func name')
sort_min = sorting(func=min, name='ascending')
sort_max = sorting(func=max, name='descending')

dict_strategy = {
    s_probability.name: s_probability.func,
    s_least.name: s_least.func,
    s_entropy.name: s_entropy.func,
    s_margin.name: s_margin.func,
    s_ratio.name: s_ratio.func,
    s_custom.name: s_custom.func,
}

dict_selection = {
    bbox_min.name: bbox_min.func,
    bbox_max.name: bbox_max.func,
    bbox_mean.name: bbox_mean.func,
    bbox_sum.name: bbox_sum.func,
}


def fa2(func, preds: List[BBoxScores], w):
    list_value = []
    for pred in preds:
        list_value.append(pred.probabilities)
    prob_tensor = torch.tensor(list_value, dtype=torch.float32)
    return dict_strategy[func](prob_tensor, len(w), torch.tensor(w, dtype=torch.float32))


def fb(func, preds):
    return dict_selection[func](preds)


def fc(func, preds):
    if func == 'ascending':
        return sorted(preds, key=lambda x: x[1], reverse=False)
    else:
        return sorted(preds, key=lambda x: x[1], reverse=True)


def al(input_json: DetectionSamplingOnPremise):
    numofsamples = input_json.num_of_samples
    bboxselectionpolicy = input_json.bbox_selection_policy
    selectionstrategy = input_json.selection_strategy
    sortstrateg = input_json.sort_strategy
    frame = input_json.frames
    w = input_json.probs_weights
    a2 = [
        (x[0], fb(bboxselectionpolicy, x[1])) for x in
        [(f.frame_id, fa2(selectionstrategy, f.predictions, w)) for f in frame]
    ]
    return fc(sortstrateg, a2)[:numofsamples]
