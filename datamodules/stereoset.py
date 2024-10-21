import json

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizer

from datamodules import BaseDataModule
from metrics.text import Perplexity

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

class StereoSetDataModule(BaseDataModule):
    ANTI_STEREOTYPE = "anti-stereotype"
    STEREOTYPE = "stereotype"
    UNRELATED = "unrelated"
    label_name = ["anti-stereotype", "stereotype", "unrelated"]
    
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = cfg.training.per_device_batch_size
        self.num_workers = cfg.data.num_workers
        self.cache_dir = cfg.cache_dir
        self.data_path = Path(__file__).parent.parent / cfg.task.data_path
        self.metrics["_train"].update({
            "ppl": Perplexity(ignore_index=-100),
        })

    def prepare_data(self) -> None:
        if self.data_path.exists():
            return

        print("Preparing Stereoset dataset...")

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
                data[self.label_name[item_entry.label]].append(asdict(item_entry))
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

        if stage == "fit":
            self.datasets["train"].append(StereoSetDataset(data[self.STEREOTYPE], self.tokenizer, split='train', max_length=256))

if __name__ == "__main__":
    # PYTHONPATH=$(pwd) python datamodules/stereoset.py
    import unittest
    from omegaconf import OmegaConf

    class TestStereoSetDataModule(unittest.TestCase):
        """StereoSetDataModule에 대한 유닛 테스트"""

        def setUp(self):
            """테스트 전에 호출되어 테스트 환경을 설정"""
            self.cfg = {
                "training": {"per_device_batch_size": 4},
                "cache_dir": Path(__file__).parent.parent / ".cache",
                "task": {"data_path": "data/stereoset.json"},
                "data": {"num_workers": 4},
            }
            self.cfg = OmegaConf.create(self.cfg)
            self.dm = StereoSetDataModule(self.cfg, tokenizer=None)

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