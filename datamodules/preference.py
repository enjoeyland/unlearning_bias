from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, DataCollatorForLanguageModeling


class MultiPromptDataset(Dataset):
    def __init__(self, data, tokenizer: PreTrainedTokenizerBase, prompt_fields=[], split='train', max_length=128):
        self.data = data
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_fields = prompt_fields
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        inputs = {}
        for field in self.prompt_fields:
            prompt = item[field]
            
            x = self.tokenizer(
                prompt,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )

            inputs[field] = {
                "input_ids": x['input_ids'].squeeze(),
                "attention_mask": x['attention_mask'].squeeze(),
            }
        
        for field in item.keys():
            if field not in self.prompt_fields:
                inputs[field] = item[field]         
        return inputs


class MultiPromptDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm: bool = True, **kwargs):
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
            **kwargs
        )

    def __call__(self, batch):
        print("batch:", batch)
        inputs = {}
        for field in batch[0].keys():
            inputs[field] = [x[field] for x in batch]
            if isinstance(inputs[field][0], dict):
                inputs[field] = self.data_collator(inputs[field])
        return inputs

if __name__ == "__main__":
    # PYTHONPATH=$(pwd) python datamodules/preference.py
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", cache_dir=".cache/", clean_up_tokenization_spaces=True)
    data = [
        {"stereotype": "The doctor is a man.", "antistereotype": "The doctor is a woman.", "bias_type": "gender"},
        {"stereotype": "The nurse is a woman.", "antistereotype": "The nurse is a man.", "bias_type": "gender"},
    ]
    dataset = MultiPromptDataset(data, tokenizer, prompt_fields=["stereotype", "antistereotype"],max_length=10)
    collator = MultiPromptDataCollator(tokenizer, mlm=False)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    for batch in dataloader:
        print(batch)
        break
