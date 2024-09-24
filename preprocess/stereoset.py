from pathlib import Path
import argparse
import json
import jsonlines
from datasets import load_dataset


def main(args):
    if not Path(f"stereoset_{args.data_name}.jsonl").exists():
        print("Loading dataset...", end=" ")
        datset = load_dataset(args.data_path, args.data_name, cache_dir=args.cache_dir)
        print("Done")


        with open(f"stereoset_{args.data_name}.jsonl", "w") as f:
            for d in datset["validation"]:
                for i in range(len(d["sentences"]["sentence"])):
                    data = {
                        "context": d["context"],
                        "sentence": d["sentences"]["sentence"][i],
                        "label": d["sentences"]["gold_label"][i],
                        "id": d["id"],
                        "target": d["target"],
                        "bias_type": d["bias_type"],
                    }

                    json.dump(data, f)
                    f.write("\n")

    label_name = ["anti-stereotype", "stereotype", "unrelated"]
    with jsonlines.open(f"stereoset_{args.data_name}.jsonl", "r") as reader:
        dataset = list(reader)            

    for label, label_name in enumerate(['anti-stereotype', 'stereotype', 'unrelated']):
        with open(f"stereoset_{args.data_name}_{label_name}.jsonl", "w") as f:
            for d in dataset:
                if d["label"] == label:
                    json.dump(d, f)
                    f.write("\n")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cache_dir", type=str, default="~/unlearning_bias/.cache")
    argparser.add_argument("--data_path", type=str, default="McGill-NLP/stereoset")
    argparser.add_argument("--data_name", type=str, default="intersentence")
    args = argparser.parse_args()

    for data_name in ["intersentence", "intrasentence"]:
        args.data_name = data_name
        main(args)