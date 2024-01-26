# adapted from https://github.com/TeunvdWeij/output_control/blob/main/src/model.py
# which was itself adapted from https://github.com/nrimsky/LM-exp/blob/main/sycophancy/sycophancy_steering.ipynb.

from interventions import AddIntervention, AblationIntervention

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.last_hidden_state = None
        self.interventions = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.last_hidden_state = output[0]
        self.output_before_adding = output
        for intervention in self.interventions:
            output = (intervention.apply(output[0]), output[1])
        self.output_after_adding = output
        return output

    def add_intervention(self, intervention):
        self.interventions.append(intervention)

    def reset(self):
        self.last_hidden_state = None
        self.interventions = []


class Llama2Helper():
    def __init__(
        self, model, tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer)

        self.n_layers = len(self.model.model.layers)

    def forward(self, x):
        return self.model(x)

    def generate_text(self, prompt, do_sample=False, temperature=.0, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(
            inputs.input_ids.to(self.model.device),
            do_sample=do_sample,
            temperature=temperature,
            # max_length=max_length,
        )
        return self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def get_logits(self, tokens):
        with torch.no_grad():
            return self.model(tokens.to(self.model.device)).logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].last_hidden_state

    def set_add_activations(self, layer, activations, target_pos):
        self.model.model.layers[layer].add_intervention(
            AddIntervention(activations, target_pos)
        )
    
    def set_ablate_activations(self, layer, activations, target_pos):
        self.model.model.layers[layer].add_intervention(
            AblationIntervention(activations, target_pos)
        )

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()