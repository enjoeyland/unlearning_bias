import json
from collections import Counter
print("importing...", end=" ")
from datasets import load_dataset
print("Done")

print("Loading dataset...", end=" ")
dataset = load_dataset("google/civil_comments", cache_dir="~/unlearning_bias/.cache")
print("Done")

cnt = Counter()
with open("civil_comments_social_bias.jsonl", "w") as f:
    for d in dataset["train"]:
        if d['identity_attack'] > 0.5:
            json.dump(d, f)
            f.write("\n")
            cnt["identity_attack"] += 1
        elif d['sexual_explicit'] > 0.5:
            json.dump(d, f)
            f.write("\n")
            cnt["sexual_explicit"] += 1
print(cnt)