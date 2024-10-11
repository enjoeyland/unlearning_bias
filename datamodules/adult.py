import json

from pathlib import Path
from dataclasses import dataclass, asdict
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from datamodules import BaseDataModule
from metrics.classification_metrics import BinaryAccuracy

@dataclass
class AdultData:
    age: int
    capital_gain: float
    capital_loss: float
    education: str # Education level: the higher, the more educated the person.
    final_weight: int
    hours_worked_per_week: int
    marital_status: str
    native_country: str
    occupation: str # Job of the person
    race: str
    relationship: str
    is_male: bool
    workclass: str
    over_threshold: int # 1 for income >= 50k$, 0 otherwise.


class AdultDataset(Dataset):
    def __init__(self, data, tokenizer, split='train', max_length=128):
        self.data = data
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        text = f"age is {item['age']}, capital gain is {item['capital_gain']}, capital loss is {item['capital_loss']}, education is {item['education']}, final weight is {item['final_weight']}, hours worked per week is {item['hours_worked_per_week']}, marital status is {item['marital_status']}, native country is {item['native_country']}, occupation is {item['occupation']}, race is {item['race']}, relationship is {item['relationship']}, gender is {'male' if item['is_male'] else 'female'}, workclass is {item['workclass']}"

        inputs = self.tokenizer(
            text, 
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': tensor(item['over_threshold']),
        }

class AdultDataModule(BaseDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = cfg.training.per_device_batch_size
        self.num_workers = cfg.data.num_workers
        self.cache_dir = cfg.cache_dir
        self.data_path = {}
        for split in ["train", "valid"]:
            self.data_path[split] = Path(__file__).parent.parent / cfg.task.data_path[split]
        
        self.num_classes = 2 # income >= 50k$ or not
        
        self.metrics.update({
            "accuracy": BinaryAccuracy(),
        })

    def prepare_data(self) -> None:
        if self.data_path["train"].exists() and self.data_path["valid"].exists():
            return

        print("Preparing Adult dataset")

        encode_dataset = load_dataset("mstz/adult", "encoding", cache_dir=self.cache_dir)["train"]
        dataset = load_dataset("mstz/adult", "income", cache_dir=self.cache_dir)
        
        education_dict = {item['encoded_value']: item['original_value'] for item in encode_dataset}
        for split in dataset:
            data = []
            for item_data in dataset[split]:
                item_entry = AdultData(
                    age=item_data["age"],
                    capital_gain=item_data["capital_gain"],
                    capital_loss=item_data["capital_loss"],
                    education=education_dict[item_data["education"]-1],
                    final_weight=item_data["final_weight"],
                    hours_worked_per_week=item_data["hours_worked_per_week"],
                    marital_status=item_data["marital_status"],
                    native_country=item_data["native_country"],
                    occupation=item_data["occupation"],
                    race=item_data["race"],
                    relationship=item_data["relationship"],
                    is_male=item_data["is_male"],
                    workclass=item_data["workclass"],
                    over_threshold=item_data["over_threshold"],
                )
                split = "train" if split == "train" else "valid"
                data.append(asdict(item_entry))
            with open(self.data_path[split], "w") as f:
                json.dump(data, f, indent=2)

    def setup(self, stage: str):
        data = load_dataset(
            "json", 
            data_files={
                "train": str(self.data_path["train"].resolve()),
                "valid": str(self.data_path["valid"].resolve())
            },
            cache_dir=self.cache_dir,
        )
        
        if stage == "fit":
            self.datasets["train"].append(AdultDataset(data["train"], self.tokenizer, split='train'))
            self.datasets["valid"].append(AdultDataset(data["valid"], self.tokenizer, split='valid'))
        
        elif stage == "validate":
            self.datasets["valid"].append(AdultDataset(data["valid"], self.tokenizer, split='valid'))


if __name__ == "__main__":
    # PYTHONPATH=$(pwd) python datamodules/adult.py
    import unittest
    from omegaconf import OmegaConf

    class TestAdultDataModule(unittest.TestCase):
        """AdultDataModule에 대한 유닛 테스트"""

        def setUp(self):
            """테스트 전에 호출되어 테스트 환경을 설정"""
            self.cfg = {
                "training": {"per_device_batch_size": 4},
                "cache_dir": Path(__file__).parent.parent / ".cache",
                "task": {"data_path": {"train": "data/adult_train.json", "valid": "data/adult_valid.json"}},
                "data": {"num_workers": 4},
            }
            self.cfg = OmegaConf.create(self.cfg)
            self.dm = AdultDataModule(self.cfg, tokenizer=None)

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
 