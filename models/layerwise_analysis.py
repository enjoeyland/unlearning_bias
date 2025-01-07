import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseModel

class LayerwiseAnalyzerModel(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.layerwise_token_tracker = None
        self.layer_indices = [*range(0, 24, 2), 24 - 1]

    def forward(self, input_ids, attention_mask=None, labels=None, **inputs):
        with self.layerwise_token_tracker as tracker:
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            tracker.layerwise_probability_plot(input_ids=input_ids)
            # tracker.scatter_plot(input_ids=input_ids, top_k=3)
            # tracker.heatmap_plot(input_ids=input_ids, top_k=3)
        return outputs
    
    def configure_model(self):
        super().configure_model()
        self.layerwise_token_tracker = LayerwiseTokenTracker(self.tokenizer, self.model, self.layer_indices)


class LayerwiseTokenTracker:
    def __init__(self, tokenizer, model, hook_layers: list[int]):
        self.tokenizer = tokenizer
        self.model = model
        self.hook_layers = hook_layers
        self.hooks = []
        self.probabilities_per_layer = []  # Stores probabilities for all tokens across layers
        self._version_counter = {}


    def __enter__(self):
        layers = self._find_layers_module(self.model)
        print("number of layers:", len(layers))
        for layer in self.hook_layers:
            self.hooks.append(layers[layer].register_forward_hook(self.hook_fn))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.probabilities_per_layer = []
        for hook in self.hooks:
            hook.remove()
    
    def _find_layers_module(self, model):
        """
        Recursively find the first attribute named 'layers' in the model.
        """
        current_model = model
        while not hasattr(current_model, "decoder"):
            submodules = [attr for attr in dir(current_model) if isinstance(getattr(current_model, attr), torch.nn.Module)]
            if not submodules:
                raise AttributeError("No 'layers' attribute found in the model.")
            current_model = getattr(current_model, submodules[0])  # Dive into the first submodule
        return current_model.decoder.layers


    def hook_fn(self, module, input, output):
        """
        Hook function to capture the hidden states from specific layers.
        """
        with torch.no_grad():
            # Final FC layer
            if isinstance(output, tuple): # (hidden_states, key_value_states)
                output = output[0]
            fc_layer = self.model.get_output_embeddings()

            logits = fc_layer(output)  # Logits from hidden states

            # Get probabilities for the entire sequence
            probabilities = torch.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
            self.probabilities_per_layer.append(probabilities[0].tolist())  # Shape: (seq_len, vocab_size)


    def _save_with_versioning(self, directory, filename, extension="png"):
        """
        Save the plot with sequential versioning.
        Starts at version 0 for a new process.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create directory if it doesn't exist

        # Initialize the version counter for this filename if not already done
        if filename not in self._version_counter:
            self._version_counter[filename] = 0  # Start from version 0 for new process

        version = self._version_counter[filename]  # Get the current version
        base_filepath = os.path.join(directory, f"{filename}_v{version}.{extension}")

        # Save the plot
        plt.savefig(base_filepath)
        print(f"Saved: {base_filepath}")

        # Increment the version for next save
        self._version_counter[filename] += 1

    def scatter_plot(self, input_ids, top_k=5):
        """
        Plot the probability changes across layers for each token in the sentence.
        """
        
        # Convert to numpy for easier processing
        num_layers = len(self.probabilities_per_layer)
        input_ids = input_ids[0].tolist()
        seq_len = input_ids.index(self.tokenizer.pad_token_id)-1 if self.tokenizer.pad_token_id in input_ids else len(input_ids)

        fig, axs = plt.subplots(seq_len, 1, figsize=(10, 2 * seq_len), sharex=True)
        # axs = [axs] if seq_len == 1 else axs  # Ensure axs is a list even for a single token
        fig.subplots_adjust(hspace=0.5)

        for i in range(seq_len):
            token_id = input_ids[i+1]
            probs_matrix = np.array([layer_probs[i] for layer_probs in self.probabilities_per_layer])  # (num_layers, vocab_size)
            max_indices = np.argsort(-probs_matrix[-1])[:top_k]  # Get top-k tokens from the last layer
            tokens = self.tokenizer.convert_ids_to_tokens(max_indices)
            input_token = self.tokenizer.convert_ids_to_tokens(token_id)

            for j, token_name in zip(max_indices, tokens):
                axs[i].plot(range(num_layers), probs_matrix[:, j], label=f"'{token_name}'")

            axs[i].set_title(f"Token '{input_token}' Probability Changes")
            axs[i].set_ylabel("Probability")
            axs[i].legend()

        axs[-1].set_xlabel("Layer")
        plt.suptitle("Layerwise Probability Changes for All Tokens in the Sentence")
        self._save_with_versioning("images", "layerswise_token_scatter", extension="png")
        plt.clf()
    
    def heatmap_plot(self, input_ids, top_k=5):
        """
        Plot a heatmap showing the probability distribution for all tokens across layers.
        """
        num_layers = len(self.probabilities_per_layer)
        input_ids = input_ids[0].tolist()
        seq_len = input_ids.index(self.tokenizer.pad_token_id)-1 if self.tokenizer.pad_token_id in input_ids else len(input_ids)

        fig, axs = plt.subplots(seq_len, 1, figsize=(10, 2 * seq_len), sharex=True)

        for i in range(seq_len):
            token_id = input_ids[i+1]

            probs_matrix = np.array([layer_probs[i] for layer_probs in self.probabilities_per_layer])  # (num_layers, vocab_size)
            max_indices = np.argsort(-probs_matrix[-1])[:top_k]  # Top-k tokens from the final layer
            token_name = self.tokenizer.convert_ids_to_tokens(token_id)

            heatmap_data = probs_matrix[:, max_indices]  # Shape: (num_layers, top_k)
            tokens = self.tokenizer.convert_ids_to_tokens(max_indices)

            sns.heatmap(heatmap_data.T, ax=axs[i], cmap="YlGnBu", cbar=False, annot=True, fmt=".2f", xticklabels=[f"Layer {idx}" for idx in self.hook_layers], yticklabels=tokens)
            axs[i].set_title(f"Probability Heatmap for Token '{token_name}'")
            axs[i].set_ylabel("Top Tokens")

        axs[-1].set_xlabel("Layers")
        plt.tight_layout()
        self._save_with_versioning("images", "layerswise_token_heatmap", extension="png")
        plt.clf()

    def layerwise_probability_plot(self, input_ids):
        """
        Create a single heatmap showing the probability changes across layers for all tokens in the input sequence.
        """
        num_layers = len(self.probabilities_per_layer)
        input_ids = input_ids[0].tolist()
        seq_len = input_ids.index(self.tokenizer.pad_token_id)-1 if self.tokenizer.pad_token_id in input_ids else len(input_ids)

        # Create a matrix where each row corresponds to one layer and each column to a token
        token_probs = np.zeros((num_layers, seq_len))

        # Populate the matrix with the top-k probabilities of the actual input tokens
        for layer_idx, layer_probs in enumerate(self.probabilities_per_layer):
            for token_idx in range(seq_len):
                token_id = input_ids[token_idx+1]
                token_probs[layer_idx, token_idx] = layer_probs[token_idx][token_id]  # Probability for actual token

        # Convert token IDs to readable token strings
        tokens = [self.tokenizer.convert_ids_to_tokens(tok_id) for tok_id in input_ids[1:seq_len+1]]

        # Plot the heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(np.flipud(token_probs), annot=False, cmap="YlGnBu", xticklabels=tokens, yticklabels=[f"Layer {idx}" for idx in self.hook_layers[::-1]])
        plt.title("Layer-wise Probability Changes for Input Tokens")
        plt.xlabel("Tokens")
        plt.ylabel("Layers")
        self._save_with_versioning("images", "layerswise_label_token_heatmap", extension="png")
        plt.clf()



# input_text = "Hello, how are you?"
# inputs = tokenizer(input_text, return_tensors="pt")
# layer_indices = [0, 2, 4]  # Hook layers

# with LayerwiseTokenTracker(tokenizer, model, layer_indices) as tracker:
#     tracker.model(**inputs)

# # Visualize with scatter plot
# tracker.scatter_plot(inputs, top_k=3)

# # Visualize with heatmap
# tracker.heatmap_plot(inputs, top_k=3)

# # Visualize probability for actual input tokens across layers
# tracker.layerwise_probability_plot(inputs)