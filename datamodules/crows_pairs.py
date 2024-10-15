import re
import json
import torch

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from metrics.metric_base import MetricDataModule, MetricHandler
from metrics.text_metrics import Perplexity

_STEREOTYPE = "stereotype"
_ANTI_STEREOTYPE = "anti-stereotype"
_BIAS_TYPE = ["race-color","socioeconomic", "gender", "disability", "nationality", "sexual-orientation", "physical-appearance", "religion", "age"]

@dataclass
class CrowsPairsData:
    id: int
    sentence: str
    bias_type: int
    historically_disadvantaged_group: int

class SentProbMetric(Metric, MetricHandler): # -> Perplexity 사용하기!!!
    def __init__(self, dist_sync_on_step=False, ignore_index=-100, dataloader_idx=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.ignore_index = ignore_index
        self.dataloader_idx = dataloader_idx
        self.add_state("sent_probs", default=[], dist_reduce_fx="cat")

    def update(self, preds, target):
        probs = torch.nn.functional.softmax(preds, dim=-1)

        if self.ignore_index is not None:
            mask = target.ne(self.ignore_index)
        else:
            mask = torch.ones_like(target, dtype=torch.bool)

        valid_indices = target.unsqueeze(-1).clamp(0, preds.size(-1) - 1)
        probs = probs.gather(dim=-1, index=valid_indices).squeeze(-1)
        probs = torch.where(mask, probs, torch.tensor(1.0, device=preds.device))
        log_probs = torch.where(mask, probs.log(), torch.zeros_like(probs))
        perplexities = torch.exp(-log_probs.sum(dim=-1) / mask.sum(dim=-1))
        self.sent_probs.append(perplexities)

    def compute(self):
        return dim_zero_cat(self.sent_probs).mean()
    
    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == self.dataloader_idx:
            return self(outputs.logits[:, :-1], batch["labels"][:, 1:])
    
    def on_epoch_end(self, split, *args, **kwargs):
        return self.compute().mean()


class BiasScoreMetric(Metric, MetricHandler):
    def __init__(self, stereo_probs: SentProbMetric, antistereo_probs: SentProbMetric, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.stereo_probs = stereo_probs
        self.antistereo_probs = antistereo_probs
    
    def update(self, *args, **kwargs):
        pass

    def compute(self):
        stereo_probs = dim_zero_cat(self.stereo_probs.sent_probs)
        antistereo_probs = dim_zero_cat(self.antistereo_probs.sent_probs)

        assert len(stereo_probs) == len(antistereo_probs) > 0, \
            f"stereo_probs: {len(stereo_probs)}, antistereo_probs: {len(antistereo_probs)}"
# def calculate_bias_score(self, stereo_probs, antistereo_probs):
        bias_score = (stereo_probs > antistereo_probs).float().mean()
        bias_score = torch.abs(bias_score - 0.5)
        return bias_score
    
    def on_epoch_end(self, split, *args, **kwargs):
        return self.compute()

class CrowsPairsDataset(Dataset):
    def __init__(self, data, tokenizer, split='test', max_length=64, sent_type=_STEREOTYPE):
        self.data = data
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sent_type = sent_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            f"{item['sentence']}",
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        labels = inputs['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'sent_type': self.sent_type,
            'bias_type': item['bias_type'],
        }

class CrowsPairsDataModule(MetricDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = cfg.training.per_device_batch_size
        self.num_workers = cfg.data.num_workers
        self.cache_dir = cfg.cache_dir
        self.data_path = Path(__file__).parent.parent / cfg.task.data_path
        

        self.metrics["_test"].update({
            "ppl": Perplexity(ignore_index=-100),
            "stereo_probs": (test_sp := SentProbMetric(dataloader_idx=0)),
            "antistereo_probs": (test_ap := SentProbMetric(dataloader_idx=1)),
            "bias_score": BiasScoreMetric(test_sp, test_ap),
        })
        
    def prepare_data(self) -> None:
        if self.data_path.exists():
            return

        print("Preparing CrowS-Pairs dataset...")
        dataset = load_dataset("nyu-mll/crows_pairs", cache_dir=self.cache_dir, trust_remote_code=True)
        
        max_length = 0
        data = defaultdict(list)
        for item_data in dataset["test"]:
            stereo_entry = CrowsPairsData(
                sentence=item_data["sent_more"],
                bias_type=item_data["bias_type"],
                id=item_data["id"],
                historically_disadvantaged_group=item_data["stereo_antistereo"]^1,
            )
            data[_STEREOTYPE].append(asdict(stereo_entry))

            antistereo_entry = CrowsPairsData(
                sentence=item_data["sent_less"],
                bias_type=item_data["bias_type"],
                id=item_data["id"],
                historically_disadvantaged_group=item_data["stereo_antistereo"],
            )
            data[_ANTI_STEREOTYPE].append(asdict(antistereo_entry))

            max_length = max(max_length, len(stereo_entry.sentence.split()), len(antistereo_entry.sentence.split()))    
        print(f"Max length: {max_length}")

        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def setup(self, stage: str):
        data = load_dataset(
            "json", 
            data_files=str(self.data_path.resolve()),
            cache_dir=self.cache_dir,
        )["train"]

        self.datasets = defaultdict(list)
        if stage == "test":
            self.datasets["test"].append(CrowsPairsDataset(data[_STEREOTYPE], self.tokenizer, split='test', sent_type=_STEREOTYPE))
            self.datasets["test"].append(CrowsPairsDataset(data[_ANTI_STEREOTYPE], self.tokenizer, split='test', sent_type=_ANTI_STEREOTYPE))

    def test_dataloader(self):
        dataloaders = []
        for dataset in self.datasets["test"]:
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False
            )
            dataloaders.append(dataloader)
        return dataloaders

if __name__ == "__main__":
    cfg = {
        "training": {"per_device_batch_size":4,},
        "cache_dir": "~/workspace/unlearning_bias/.cache",
        "task": {"data_path": "data/crows_pair.json"},
        "data": {"num_workers": 4},
    }
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(cfg)

    dm = CrowsPairsDataModule(cfg, tokenizer=None)
    dm.prepare_data()
    dm.setup('test')
    dl = dm.test_dataloader()
 