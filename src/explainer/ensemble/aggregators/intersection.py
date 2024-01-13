import copy
import sys
from abc import ABC

from src.core.explainer_base import Explainer
from src.dataset.instances.graph import GraphInstance
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
import numpy as np

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset


class ExplanationIntersection(ExplanationAggregator):

    def aggregate(self, instance, explanations):
        label = self.oracle.predict(instance)

        edge_freq_matrix = np.zeros_like(instance.data)
        for exp in explanations:
            if self.oracle.predict(exp) != label:
                edge_freq_matrix = edge_freq_matrix * exp.data

        adj = np.where(edge_freq_matrix > 0, 1, 0)

        cf_candidate = GraphInstance(id=instance.id,
                             label=1-instance.label,
                             data=adj,
                             node_features=instance.node_features)
        
        for manipulator in instance._dataset.manipulators:
            manipulator._process_instance(cf_candidate)
        
        return cf_candidate