import json

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizer

from metric_logging import MetricDataModule

@dataclass
class StereoSetData:
    context: str
    sentence: str
    bias_type: str
    label: str
    id: str
    target: str       


class StereoSetDataset(Dataset):
    def __init__(self, data, tokenizer: PreTrainedTokenizer, split='train', max_length=128):
        self.data = data
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = item['sentence']
        if "BLANK" not in item['context']:
            prompt = f"{item['context']} {prompt}"

        inputs = self.tokenizer(
            prompt,
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

class StereoSetDataModule(MetricDataModule):
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
        
        max_length = 0
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
                max_length = max(max_length, len(item_entry.context.split()) + len(item_entry.sentence.split()))
        print(f"Max length: {max_length}")

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
            self.datasets["train"] = StereoSetDataset(data[self.STEREOTYPE], self.tokenizer, split='train', max_length=256)


    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )

if __name__ == "__main__":
    cfg = {
        "training": {"per_device_batch_size":4,},
        "cache_dir": "~/workspace/unlearning_bias/.cache",
        "task": {"data_path": "data/stereoset.json"},
        "data": {"num_workers": 4},
    }
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(cfg)

    dm = StereoSetDataModule(cfg, tokenizer=None)
    dm.prepare_data()
    dm.setup('fit')
    dl = dm.train_dataloader()
 