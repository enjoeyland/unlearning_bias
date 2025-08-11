from .base import BaseModel
from .dpo import DpoModel
from .grad_ascent import GradAscentModel, GradAscentKDModel
from .in_process import RegularizationModel, AdversarialRepresentationModel
from .layerwise_analysis import LayerwiseAnalyzerModel
from .influence_func import ReviewWrongAnswerModel
from .resample import ResampleModel
from .generation import GenerationModel, PredictionDatasetSaver

class ModelFactory:
    def __init__(self, cfg):
        self.cfg = cfg
        self._model_builders = {
            'dpo': DpoModel,
            'grad_ascent': GradAscentModel,
            'grad_ascent_kd': GradAscentKDModel,
            'regularization': RegularizationModel,
            'representation': AdversarialRepresentationModel,
            'layerwise_analysis': LayerwiseAnalyzerModel,
            'review_wrong': ReviewWrongAnswerModel,
            'resample': ResampleModel,
            'find_prediction_dataset': PredictionDatasetSaver,
        }
    def get_model_class(self):
        method_name = self.cfg.method.name
        if method_name in self._model_builders:
            return self._model_builders[method_name]
        elif self.cfg.task.name in ["adult", "compas"] and self.cfg.task.task_type == "CAUSAL_LM":
            # zero_shot, finetune_zero_shot, negtaskvector
            return GenerationModel
        return BaseModel
    
    def create_model(self):
        model_builder = self.get_model_class()
        return model_builder(self.cfg)