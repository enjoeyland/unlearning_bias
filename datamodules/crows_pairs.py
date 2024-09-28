import json
import torch
import lightning as L

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


@dataclass
class CrowsPairsData:
    id: int
    sentence: str
    bias_type: int
    historically_disadvantaged_group: int

def calculate_sent_prob(pl_module, batch, outputs):
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    labels = batch['labels']

    valid_token_mask = labels != -100
    sentence_log_probs = (log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1) * valid_token_mask).sum(dim=1)
    # sentence_log_probs = []
    # for i in range(input_ids.size(0)):
    #     label_ids = labels[i]
    #     sentence_log_prob = 0.0
    #     for j in range(label_ids.size(0)):
    #         if label_ids[j] != -100:
    #             token_prob = log_probs[i, j, label_ids[j]].item()
    #             sentence_log_prob += token_prob

    #     sentence_log_probs.append(sentence_log_prob)

    return sentence_log_probs

def calculate_bias(stereotype_prob, anti_stereotype_prob):
    bias_score = (stereotype_prob > anti_stereotype_prob).float().mean()
    return torch.abs(bias_score - 0.5)

class CrowsPairsDataset(Dataset):
    def __init__(self, data, tokenizer, split='test', max_length=512):
        self.data = data
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

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
            'labels': labels.squeeze()
        }

class CrowsPairsDataModule(L.LightningDataModule):
    ANTI_STEREOTYPE = "anti-stereotype"
    STEREOTYPE = "stereotype"
    BIAS_TYPE = ["race-color","socioeconomic", "gender", "disability", "nationality", "sexual-orientation", "physical-appearance", "religion","age"]

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

        print("Preparing CrowS-Pairs dataset...")
        dataset = load_dataset("nyu-mll/crows_pairs", cache_dir=self.cache_dir, trust_remote_code=True)
        
        data = defaultdict(list)
        for item_data in dataset["test"]:
            item_entry = CrowsPairsData(
                sentence=item_data["sent_more"],
                bias_type=item_data["bias_type"],
                id=item_data["id"],
                historically_disadvantaged_group=item_data["stereo_antistereo"]^1,
            )
            data[self.STEREOTYPE].append(asdict(item_entry))

            item_entry = CrowsPairsData(
                sentence=item_data["sent_less"],
                bias_type=item_data["bias_type"],
                id=item_data["id"],
                historically_disadvantaged_group=item_data["stereo_antistereo"],
            )
            data[self.ANTI_STEREOTYPE].append(asdict(item_entry))


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
            self.datasets["test"].append(CrowsPairsDataset(data[self.STEREOTYPE], self.tokenizer, split='test', max_length=self.max_length))
            self.datasets["test"].append(CrowsPairsDataset(data[self.ANTI_STEREOTYPE], self.tokenizer, split='test', max_length=self.max_length))

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
        "data": {
            "max_length": 512,
            "num_workers": 4,
        },
    }
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(cfg)

    dm = CrowsPairsDataModule(cfg, tokenizer=None)
    dm.prepare_data()
    dm.setup('test')
    dl = dm.test_dataloader()
 