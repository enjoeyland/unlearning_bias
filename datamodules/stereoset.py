
import json
import lightning as L

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets

@dataclass
class StereoSetData:
    context: str
    sentence: str
    bias_type: str
    label: str
    id: str
    target: str       


class StereoSetDataset(Dataset):
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
            f"{item['context']} {item['sentence']}",
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

class StereoSetDataModule(L.LightningDataModule):
    ANTI_STEREOTYPE = "anti-stereotype"
    STEREOTYPE = "stereotype"
    UNRELATED = "unrelated"
    
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = cfg.training.per_device_batch_size
        self.num_workers = cfg.data.num_workers
        self.cache_dir = cfg.cache_dir
        self.data_path = Path(__file__).parent.parent / cfg.task.data_path


    def prepare_data(self) -> None:
        if self.data_path.exists():
            return

        print("Preparing Stereoset dataset...")
        label_name = ["anti-stereotype", "stereotype", "unrelated"]

        dataset_inter = load_dataset("McGill-NLP/stereoset", "intersentence", cache_dir=self.cache_dir)
        dataset_intra = load_dataset("McGill-NLP/stereoset", "intrasentence", cache_dir=self.cache_dir)
        
        data = defaultdict(list)
        for item_data in concatenate_datasets([dataset_inter["validation"], dataset_intra["validation"]]):
            for i in range(len(item_data["sentences"]["sentence"])):
                item_entry = StereoSetData(
                    context=item_data["context"],
                    sentence=item_data["sentences"]["sentence"][i],
                    bias_type=item_data["bias_type"],
                    label=item_data["sentences"]["gold_label"][i],
                    id=item_data["id"],
                    target=item_data["target"],
                )
                data[label_name[item_entry.label]].append(asdict(item_entry))

        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def setup(self, stage=None):
        data = load_dataset(
            "json", 
            data_files=str(self.data_path.resolve()),
            cache_dir=self.cache_dir,
        )["train"]

        self.datasets = defaultdict(list)
        if stage == 'fit' or stage is None:
            self.datasets["train"] = StereoSetDataset(data[self.STEREOTYPE], self.tokenizer, split='train')

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

if __name__ == "__main__":
    cfg = {
        "batch_size": 32,
        "num_workers": 4,
        "cache_dir": "~/unlearning_bias/.cache",
        "data_path": "data/stereoset.json"
    }
    import argparse
    cfg = argparse.Namespace(**cfg)

    dm = StereoSetDataModule(cfg, tokenizer=None)
    dm.prepare_data()
    dm.setup()
    dl = dm.train_dataloader()
 