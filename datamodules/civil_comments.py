import json

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from datamodules import BaseDataModule
from metrics.text import Perplexity
from utils import get_absolute_path

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
    def __init__(self, data, tokenizer, split='train', max_length=256):
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

class CivilCommentsDataModule(BaseDataModule):
    def __init__(self, module, cfg, tokenizer):
        super().__init__(module, cfg)
        self.tokenizer = tokenizer
        self.cache_dir = cfg.cache_dir
        self.data_path = get_absolute_path(cfg.task.data_path)
        self.metrics["_train"].update({
            "ppl": Perplexity()
        })

    def prepare_data(self) -> None:
        if self.data_path.exists():
            return

        print("Preparing CivilComments dataset...")

        dataset = load_dataset("google/civil_comments", cache_dir=self.cache_dir)
        
        max_length = 0
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
                max_length = max(max_length, len(item_entry.text.split()))
        print(f"Max length: {max_length}")

        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def setup(self, stage: str):
        data = load_dataset(
            "json", 
            data_files=str(self.data_path.resolve()),
            cache_dir=self.cache_dir,
        )["train"]
        
        if stage == "fit":
            self.datasets["train"].append(CivilCommentsDataset(data["social_bias"], self.tokenizer, split='train'))

if __name__ == "__main__":
    # PYTHONPATH=$(pwd) python datamodules/civil_comments.py
    import unittest
    from omegaconf import OmegaConf

    class TestCivilCommentsDataModule(unittest.TestCase):
        """CivilCommentsDataModule에 대한 유닛 테스트"""

        def setUp(self):
            """테스트 전에 호출되어 테스트 환경을 설정"""
            self.cfg = {
                "training": {"per_device_batch_size": 4},
                "cache_dir": get_absolute_path(".cache"),
                "task": {"data_path": "data/civil_comments.json"},
                "data": {"num_workers": 4},
            }
            self.cfg = OmegaConf.create(self.cfg)
            self.dm = CivilCommentsDataModule(self.cfg, tokenizer=None)

        def test_prepare_data(self):
            """prepare_data 메서드가 올바르게 실행되는지 테스트"""
            try:
                self.dm.prepare_data()
            except Exception as e:
                self.fail(f"prepare_data 실행 중 오류 발생: {e}")

        def test_setup(self):
            """setup 메서드가 올바르게 실행되는지 테스트"""
            try:
                self.dm.setup('fit')
            except Exception as e:
                self.fail(f"setup 실행 중 오류 발생: {e}")

        def test_train_dataloader(self):
            """train_dataloader 메서드가 올바르게 실행되는지 테스트"""
            self.dm.setup('fit')
            try:
                dl = self.dm.train_dataloader()
                self.assertIsInstance(dl, DataLoader, "train_dataloader가 DataLoader 객체를 반환해야 함")
            except Exception as e:
                self.fail(f"train_dataloader 실행 중 오류 발생: {e}")
    
    unittest.main()

 