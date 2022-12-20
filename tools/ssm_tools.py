from libfmp.c4 import *
from libfmp.c3 import *

import numpy as np

from tools.processing_tools_final import normalize_feature_sequence_z
from tools.feature_extraction_tools import ExtractFeatureMatrix

def compute_ssm(s, window_size, overlap_perc):
    feat_Mat = ExtractFeatureMatrix(s, window_size, perc_overlap=overlap_perc)

    F_set = normalize_feature_sequence_z(np.array(feat_Mat["allfeatures"]))

    S = compute_sm_dot(F_set, F_set)

    return S

def compute_novelty(S, kernel_size):
    nov_ssm = compute_novelty_ssm(S, L = kernel_size)

    return nov_ssm

def novelty_event_cost(s, window_size, kernel_size, overlap_perc):
    S = compute_ssm(s, window_size, overlap_perc)
    nov_ssm = compute_novelty(S, kernel_size)

    return nov_ssm

def perioric_event_cost(s, window_size, overlap_perc):
    S = compute_ssm(s, window_size, overlap_perc)
    per_ssm = np.sum(S, axis=0)

    return per_ssm


