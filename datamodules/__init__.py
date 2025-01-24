from .base import BaseDataModule, BaseSampler, CombinedDataModule
from .stereoset import StereoSetDataModule
from .civil_comments import CivilCommentsDataModule
from .crows_pairs import CrowsPairsDataModule
from .adult import AdultDataModule
from .compas import CompasDataModule

class DataModuleFactory:
    def __init__(self, module, cfg, tokenizer):
        self.module = module
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.task = cfg.task.name
    
    def create_datamodule(self, task):
        if "combined" in task:
            datamodules = []
            for target, data_path in zip(self.cfg.task.targets, self.cfg.task.data_paths):
                self.cfg.task.data_path = data_path
                datamodules.append(self.create_datamodule(target))
            return CombinedDataModule(self.module, self.cfg, self.tokenizer, datamodules)

        if task == "stereoset":
            return StereoSetDataModule(self.module, self.cfg, self.tokenizer)
        elif task == "civil_comments":
            return CivilCommentsDataModule(self.module, self.cfg, self.tokenizer)
        elif task == "crows_pairs":
            return CrowsPairsDataModule(self.module, self.cfg, self.tokenizer)
        elif task == "adult":
            return AdultDataModule(self.module, self.cfg, self.tokenizer)
        elif task == "compas":
            return CompasDataModule(self.module, self.cfg, self.tokenizer)
        else:
            raise NotImplementedError(f"Task {task} not implemented.")

    @staticmethod
    def configure_callbacks_monitor(cfg):
        task = cfg.task.name
        method = cfg.method.name
        
        monitor = None
        mode = "min"
        filename = "best"

        if task in ["stereoset", "crows_pairs"] or ("combined" in task and ("stereoset" in cfg.task.targets or "crows_pairs" in cfg.task.targets)):
            if method == "dpo":
                ...
            elif cfg.method.fit_target == "forget":
                monitor = "valid/bias_score"
                mode = "max"
                filename = "ppl={valid/ppl/dataloader_idx_0:.2f}-bias_score={valid/bias_score:.4f}"
        
        elif task == "adult" or task == "compas":
            if method in ["grad_ascent", "grad_ascent_kd"]:
                monitor = "valid/equal_opportunity"
                mode = "min"
            else:
                monitor = "valid/accuracy"
                mode = "max"
            filename = "acc={valid/accuracy:.3f}-eo={valid/equal_opportunity:.4f}-spd={valid/spd:.4f}"
        else:
            print(f"Task {task} is not setup for callbacks.")  


        if method == "finetune":
            if cfg.method.fit_target == "forget":
                filename = f"forget_{filename}"
            elif cfg.method.fit_target == "retain":
                if hasattr(cfg.data, "retain_multiplier") and not cfg.data.retain_multiplier:
                    filename = f"retain{cfg.data.retain_multiplier}_{filename}"
                else:
                    filename = f"retain_{filename}"
        elif method == "grad_ascent":
            if cfg.method.fit_target == "without_retain":
                filename = f"without_retain_{filename}"
            elif cfg.method.fit_target == "with_retain":
                filename = f"with_retain_{filename}"
        
        return monitor, mode, filename