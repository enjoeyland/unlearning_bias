import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", cache_dir=".cache/")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", cache_dir=".cache/", clean_up_tokenization_spaces=True)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def sentence_tsne(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        latent_values = model(**inputs, output_hidden_states=True).hidden_states[-1].squeeze(0)
        latent_values_np = latent_values.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    latent_2d = tsne.fit_transform(latent_values_np)
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6)
    plt.title("t-SNE Visualization of Latent Space")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("images/t-sne.png")
    plt.clf()

def multi_sentence_tsne(texts, colors):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)

    for text, color in zip(texts, colors):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            latent_values = model(**inputs, output_hidden_states=True).hidden_states[-1].squeeze(0)
            print("Fitting t-SNE for", text)
            latent_2d = tsne.fit_transform(latent_values.cpu().numpy())
            plt.scatter(latent_2d[:, 0], latent_2d[:, 1], label=text, color=color, alpha=0.6)

    plt.title("Comparison of Latent Space via t-SNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.savefig("images/t-sne_comparison.png")
    plt.clf()

# sentence_tsne("The doctor is a man. The nurse is a woman.", model, tokenizer)
# multi_sentence_tsne(
#     ["The doctor is a man.", "The docter is a woman.", "The nurse is a man.", "The nurse is a woman."],
#     ["red", "blue", "green", "yellow"],
# )

# PYTHONPATH=$(pwd) python sandbox/t-sne.py
from tqdm import tqdm
from datamodules import CrowsPairsDataModule
from omegaconf import OmegaConf

def crows_pairs_tsne(): # 30분 걸림..
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cfg = {
        "training": {"per_device_batch_size": 20},
        "cache_dir": ".cache",
        "task": {"data_path": "data/crows_pairs.json"},
        "data": {"num_workers": 4},
    }
    cfg = OmegaConf.create(cfg)


    dm = CrowsPairsDataModule(cfg, tokenizer)
    dm.setup("fit")
    stereo_dataloader, anti_stereo_dataloader = dm.val_dataloader()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    sentence_vectors = []
    labels = []
    colors = {"stereotype": "red", "anti-stereotype": "blue"}
    for dataloader in [stereo_dataloader, anti_stereo_dataloader]:
        print(dataloader.dataset.sent_type)
        with torch.no_grad():
            for inputs in tqdm(dataloader):
                latent_values = model(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), output_hidden_states=True).hidden_states[-1]
                latent_values = latent_values * inputs["attention_mask"].unsqueeze(-1).to(device)
                sentence_vectors.append(latent_values.mean(dim=1).cpu())
                labels.extend([dataloader.dataset.sent_type] * inputs["input_ids"].size(0))
    
    latent_2d = tsne.fit_transform(torch.cat(sentence_vectors).numpy())
    for label in set(labels):
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(latent_2d[indices, 0], latent_2d[indices, 1], color=colors[label], alpha=0.6)

    plt.title("t-SNE Visualization of Crows Pairs Dataset")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(["Stereotype", "Anti-Stereotype"])
    plt.savefig("images/t-sne_crows_pairs.png")
    plt.clf()

crows_pairs_tsne()