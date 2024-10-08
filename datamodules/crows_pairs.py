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
from torch.nn import ModuleDict, ModuleList

from metric_logging import MetricDataModule

_STEREOTYPE = "stereotype"
_ANTI_STEREOTYPE = "anti-stereotype"
_BIAS_TYPE = ["race-color","socioeconomic", "gender", "disability", "nationality", "sexual-orientation", "physical-appearance", "religion", "age"]

@dataclass
class CrowsPairsData:
    id: int
    sentence: str
    bias_type: int
    historically_disadvantaged_group: int

class SentProbMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("sent_probs", default=[], dist_reduce_fx="cat")

    def update(self, outputs, batch):
        # self.sent_probs.append(-outputs.loss)

        # log_likelihood = - outputs.loss * batch["input_ids"].size(1)
        # self.sent_probs.append(log_likelihood)
# def calculate_sent_probs(self, model, batch):
#     outputs = model(**batch)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size, seq_len, num_labels)
        labels = batch['labels']  # (batch_size, seq_len)
        input_ids = batch['input_ids']

        sentence_log_probs = []
        for i in range(input_ids.size(0)): # iterate over batch
            label_ids = labels[i]
            sentence_log_prob = 0.0
            for j in range(label_ids.size(0)): # iterate over tokens
                if label_ids[j] != -100:
                    token_prob = log_probs[i, j, label_ids[j]].item()
                    sentence_log_prob += token_prob
            sentence_log_probs.append(sentence_log_prob)
        
        self.sent_probs.append(torch.tensor(sentence_log_probs))


    def compute(self):
        return dim_zero_cat(self.sent_probs) 

class BiasScoreMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("stereo_probs", default=[], dist_reduce_fx="cat")
        self.add_state("antistereo_probs", default=[], dist_reduce_fx="cat")

    def update(self, sent_type, sent_probs):
        if sent_type == _STEREOTYPE:
            self.stereo_probs.append(sent_probs)
        elif sent_type == _ANTI_STEREOTYPE:
            self.antistereo_probs.append(sent_probs)
        
    def compute(self):
        stereo_probs = dim_zero_cat(self.stereo_probs)
        antistereo_probs = dim_zero_cat(self.antistereo_probs)

        assert len(stereo_probs) == len(antistereo_probs) > 0, \
            f"stereo_probs: {len(stereo_probs)}, antistereo_probs: {len(antistereo_probs)}"
# def calculate_bias_score(self, stereo_probs, antistereo_probs):
        bias_score = (stereo_probs > antistereo_probs).float().mean()
        bias_score = torch.abs(bias_score - 0.5)
        return bias_score
        

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
        self.metrics = ModuleDict({
            "test": ModuleList([
                ModuleDict({"sent_probs": SentProbMetric()}), 
                ModuleDict({"sent_probs": SentProbMetric()}), 
                ModuleDict({"bias_score": BiasScoreMetric()})
            ])
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
 