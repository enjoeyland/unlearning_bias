import json
import torch

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.functional.text import perplexity

from datamodules import BaseDataModule
from metrics.metric_base import MetricHandler
from metrics.text import Perplexity
from datamodules.preference import MultiPromptDataset, SingleDataCollator, ConcatDataCollator
from utils import get_absolute_path

_BIAS_TYPE = ["race-color","socioeconomic", "gender", "disability", "nationality", "sexual-orientation", "physical-appearance", "religion", "age"]

@dataclass
class CrowsPairsData:
    id: int
    stereotype: str
    antistereotype: str
    bias_type: int
    historically_disadvantaged_group: int

class BiasScoreDerivation(Metric, MetricHandler):
    stereo_ppl: torch.Tensor
    antistereo_ppl: torch.Tensor

    def __init__(self, ignore_index=-100):
        import warnings
        warnings.warn("This class is deprecated. No mathmetical way to calculate the bias score. Use T-Test instead.", DeprecationWarning)
        
        super().__init__()
        self.ignore_index = ignore_index
        self.add_state("stereo_ppl", default=[], dist_reduce_fx="cat")
        self.add_state("antistereo_ppl", default=[], dist_reduce_fx="cat")
    
    def update(self, preds, target, sent_type):
        if sent_type == "stereotype":
            stereo_ppl = []
            for i in range(len(preds)):
                stereo_ppl.append(perplexity(preds[i].unsqueeze(0), target[i].unsqueeze(0), ignore_index=self.ignore_index))
            stereo_ppl = torch.tensor(stereo_ppl, device=preds.device)
            self.stereo_ppl.append(stereo_ppl)
        elif sent_type == "antistereotype":
            antistereo_ppl = []
            for i in range(len(preds)):
                antistereo_ppl.append(perplexity(preds[i].unsqueeze(0), target[i].unsqueeze(0), ignore_index=self.ignore_index))
            antistereo_ppl = torch.tensor(antistereo_ppl, device=preds.device)
            self.antistereo_ppl.append(antistereo_ppl)
    
    def compute(self):
        stereo_ppl = torch.tensor([])
        antistereo_ppl = torch.tensor([])

        if len(self.stereo_ppl) != 0:
            stereo_ppl = dim_zero_cat(self.stereo_ppl)
        if len(self.antistereo_ppl) != 0:
            antistereo_ppl = dim_zero_cat(self.antistereo_ppl)
        
        if len(self.stereo_ppl) != len(self.antistereo_ppl):
            if len(self.stereo_ppl) != 0:
                return stereo_ppl.mean()
            if len(self.antistereo_ppl) != 0:
                return antistereo_ppl.mean()

        assert len(stereo_ppl) == len(antistereo_ppl) > 0, \
            f"stereo_ppl: {len(stereo_ppl)}, antistereo_ppl: {len(antistereo_ppl)}"
        
        bias_score = (stereo_ppl < antistereo_ppl).float().mean()
        bias_score = torch.abs(bias_score - 0.5)
        return bias_score

    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        return self(outputs.logits[:, :-1], batch["labels"][:, 1:], batch["prompt_field"])

    def on_epoch_end(self, split, *args, **kwargs):
        bias_score = self.compute()
        self.reset()
        return bias_score

# TODO: 현재 dpo는 CrwosPair에서만 됌. 다른 데이터셋에서도 사용할 수 있도록 수정 필요
class CrowsPairsDataModule(BaseDataModule):
    def __init__(self, module, cfg, tokenizer):
        super().__init__(module, cfg)
        self.tokenizer = tokenizer
        self.cache_dir = cfg.cache_dir
        self.data_path = get_absolute_path(cfg.task.data_path)
        self.is_pairwised = cfg.method.name == "dpo"
        self.max_length = 64

        self.metrics["_valid"].update({"ppl": Perplexity(ignore_index=-100)})
        self.metrics["_test"].update({"ppl": Perplexity(ignore_index=-100)})
        
        if self.is_pairwised:
            self.collate_fn = ConcatDataCollator(tokenizer, mlm=False)
        else:
            self.collate_fn = SingleDataCollator(tokenizer, mlm=False)
            self.metrics["_valid"].update({"bias_score": BiasScoreDerivation()})
            self.metrics["_test"].update({"bias_score": BiasScoreDerivation()})

    def prepare_data(self) -> None:
        if Path(self.data_path).exists():
            return

        print("Preparing CrowS-Pairs dataset...")
        dataset = load_dataset("nyu-mll/crows_pairs", cache_dir=self.cache_dir, trust_remote_code=True)
        
        max_length = 0
        data = []
        for item_data in dataset["test"]:
            entry = CrowsPairsData(
                stereotype=item_data["sent_more"],
                antistereotype=item_data["sent_less"],
                bias_type=item_data["bias_type"],
                id=item_data["id"],
                historically_disadvantaged_group=item_data["stereo_antistereo"]^1,
            )
            data.append(asdict(entry))

            max_length = max(max_length, len(entry.stereotype.split()), len(entry.antistereotype.split()))    
        print(f"Max length: {max_length}") # 40

        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def setup(self, stage: str):
        data = load_dataset(
            "json",
            data_files=self.data_path,
            cache_dir=self.cache_dir,
        )["train"]

        self.datatsets = defaultdict(list)

        if self.is_pairwised:
            if stage == "fit":
                # split data into train and validate
                data = data.train_test_split(test_size=0.1, seed=42)
                self.datasets["train"].append(MultiPromptDataset(data["train"], self.tokenizer, prompt_fields=["stereotype", "antistereotype"], split='train', max_length=self.max_length))
                self.datasets["valid"].append(MultiPromptDataset(data["test"], self.tokenizer, prompt_fields=["stereotype", "antistereotype"], split='valid', max_length=self.max_length))
            elif stage == "validate":
                self.datasets["valid"].append(MultiPromptDataset(data, self.tokenizer, prompt_fields=["antistereotype", "stereotype"], split='valid', max_length=self.max_length))
            elif stage == "test":
                self.datasets["test"].append(MultiPromptDataset(data, self.tokenizer, prompt_fields=["antistereotype", "stereotype"], split='test', max_length=self.max_length))
        else:
            if stage == "fit" or stage == "validate":
                self.datasets["valid"].append(MultiPromptDataset(data, self.tokenizer, prompt_fields=["stereotype"], split='valid', max_length=self.max_length))
                self.datasets["valid"].append(MultiPromptDataset(data, self.tokenizer, prompt_fields=["antistereotype"], split='valid', max_length=self.max_length))
            elif stage == "test":
                self.datasets["test"].append(MultiPromptDataset(data, self.tokenizer, prompt_fields=["stereotype"], split='test', max_length=self.max_length))
                self.datasets["test"].append(MultiPromptDataset(data, self.tokenizer, prompt_fields=["antistereotype"], split='test', max_length=self.max_length))

if __name__ == "__main__":
    # PYTHONPATH=$(pwd) python datamodules/crews_pairs.py
    import unittest
    from omegaconf import OmegaConf

    class TestCrowsPairsDataModule(unittest.TestCase):
        """CrowsPairsDataModule 대한 유닛 테스트"""

        def setUp(self):
            """테스트 전에 호출되어 테스트 환경을 설정"""
            self.cfg = {
                "training": {"per_device_batch_size": 4},
                "cache_dir": get_absolute_path(".cache"),
                "task": {"data_path": "data/crows_pairs.json"},
                "data": {"num_workers": 4},
                "method": {"name": "dpo"},
            }
            self.cfg = OmegaConf.create(self.cfg)
            self.dm = CrowsPairsDataModule(self.cfg, tokenizer=None)

        def test_prepare_data(self):
            """prepare_data 메서드가 올바르게 실행되는지 테스트"""
            try:
                self.dm.prepare_data()
            except Exception as e:
                self.fail(f"prepare_data 실행 중 오류 발생: {e}")

        def test_setup(self):
            """setup 메서드가 올바르게 실행되는지 테스트"""
            try:
                self.dm.setup('test')
            except Exception as e:
                self.fail(f"setup 실행 중 오류 발생: {e}")

        def test_test_dataloader(self):
            """test_dataloader 메서드가 올바르게 실행되는지 테스트"""
            self.dm.setup('test')
            try:
                dls = self.dm.test_dataloader()
                self.assertIsInstance(dls, list, "test_dataloader가 list 객체를 반환해야 함")
                for dl in dls:
                    self.assertIsInstance(dl, DataLoader, "test_dataloader가 DataLoader 객체를 반환해야 함")
            except Exception as e:
                self.fail(f"test_dataloader 실행 중 오류 발생: {e}")
    
    unittest.main()