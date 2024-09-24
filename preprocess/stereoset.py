import argparse
import json

from collections import Counter
print("importing...", end=" ")
from datasets import load_dataset, concatenate_datasets
print("Done")

def main(args):
    ANTI_STEREOTYPE = 0
    STEREOTYPE = 1
    UNRELATED = 2
    
    label_name = ["anti-stereotype", "stereotype", "unrelated"]

    cnt_stereoset = Counter()
    with open(f"stereoset_{label_name[ANTI_STEREOTYPE]}.jsonl", "w") as f_antistereotype, \
            open(f"stereoset_{label_name[STEREOTYPE]}.jsonl", "w") as f_stereotype, \
            open(f"stereoset_{label_name[UNRELATED]}.jsonl", "w") as f_unrelated:
        
        print("Loading dataset...", end=" ")
        dataset_inter = load_dataset(args.data_path, "intersentence", cache_dir=args.cache_dir)
        dataset_intra = load_dataset(args.data_path, "intrasentence", cache_dir=args.cache_dir)
        print("Done")


        for d in concatenate_datasets([dataset_inter["validation"], dataset_intra["validation"]]):
            for i in range(len(d["sentences"]["sentence"])):

                data = {
                    "context": d["context"],
                    "sentence": d["sentences"]["sentence"][i],
                    "bias_type": d["bias_type"],
                    "label": d["sentences"]["gold_label"][i],
                    "id": d["id"],
                    "target": d["target"],
                }


                if data["label"] == ANTI_STEREOTYPE:
                    json.dump(data, f_antistereotype)
                    f_antistereotype.write("\n")
                elif data["label"] == STEREOTYPE:
                    json.dump(data, f_stereotype)
                    f_stereotype.write("\n")
                    cnt_stereoset[d["bias_type"]] += 1
                elif data["label"] == UNRELATED:
                    json.dump(data, f_unrelated)
                    f_unrelated.write("\n")
    print(cnt_stereoset)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cache_dir", type=str, default="~/unlearning_bias/.cache")
    argparser.add_argument("--data_path", type=str, default="McGill-NLP/stereoset")
    args = argparser.parse_args()

    main(args)