import json

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from metric_logging import MetricDataModule

@dataclass
class CivilCommentsData:
    text: str
    identity_attack: float
    sexual_explicit: float
    insult: float
    toxicity: float
    obscene: float
    severe_toxicity: float
    threat: float


class CivilCommentsDataset(Dataset):
    def __init__(self, data, tokenizer, split='train', max_length=512):
        self.data = data
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            f"{item['text']}",
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
            'labels': labels.squeeze()
        }

class CivilCommentsDataModule(MetricDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = cfg.training.per_device_batch_size
        self.num_workers = cfg.data.num_workers
        self.cache_dir = cfg.cache_dir
        self.data_path = Path(__file__).parent.parent / cfg.task.data_path
        self.max_length = cfg.data.max_length

    def prepare_data(self) -> None:
        if self.data_path.exists():
            return

        print("Preparing CivilComments dataset...")

        dataset = load_dataset("google/civil_comments", cache_dir=self.cache_dir)
        
        data = defaultdict(list)
        for item_data in dataset["train"]:
            item_entry = CivilCommentsData(
                text=item_data["text"],
                identity_attack=item_data["identity_attack"],
                sexual_explicit=item_data["sexual_explicit"],
                insult=item_data["insult"],
                toxicity=item_data["toxicity"],
                obscene=item_data["obscene"],
                severe_toxicity=item_data["severe_toxicity"],
                threat=item_data["threat"],
            )
            if item_entry.identity_attack > 0.5 or item_entry.sexual_explicit > 0.5:
                data["social_bias"].append(asdict(item_entry))

        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def setup(self, stage: str):
        data = load_dataset(
            "json", 
            data_files=str(self.data_path.resolve()),
            cache_dir=self.cache_dir,
        )["train"]
        
        self.datasets = {}
        if stage == "fit":
            self.datasets["train"] = CivilCommentsDataset(data["social_bias"], self.tokenizer, split='train', max_length=self.max_length)

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

if __name__ == "__main__":
    cfg = {
        "training": {"per_device_batch_size":4,},
        "cache_dir": "~/workspace/unlearning_bias/.cache",
        "task": {"data_path": "data/civil_comments.json"},
        "data": {
            "max_length": 512,
            "num_workers": 4,
        },
    }
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(cfg)

    dm = CivilCommentsDataModule(cfg, tokenizer=None)
    dm.prepare_data()
    dm.setup('fit')
    dl = dm.train_dataloader()
 