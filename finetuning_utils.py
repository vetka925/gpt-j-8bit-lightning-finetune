import torch 

from gpt_quant_modules import FrozenBNBEmbedding, FrozenBNBLinear


def add_all_adapters(model, adapter_dim=2):
    assert adapter_dim > 0

    for module in model.modules():
        if isinstance(module, FrozenBNBLinear):
            module.adapter = torch.nn.Sequential(
                torch.nn.Linear(module.in_features, adapter_dim, bias=False),
                # torch.nn.Dropout(p=0.1),
                torch.nn.Linear(adapter_dim, module.out_features, bias=False),
            )
            torch.nn.init.zeros_(module.adapter[1].weight)
        elif isinstance(module, FrozenBNBEmbedding):
            module.adapter = torch.nn.Sequential(
                torch.nn.Embedding(module.num_embeddings, adapter_dim),
                torch.nn.Linear(adapter_dim, module.embedding_dim, bias=False),
            )
            torch.nn.init.zeros_(module.adapter[1].weight)

def add_attention_adapters(model, adapter_dim=2):
    assert adapter_dim > 0

    for name, module in model.named_modules():
        if isinstance(module, FrozenBNBLinear):
            if "attn" in name:
                print("Adding adapter to", name)
                module.adapter = torch.nn.Sequential(
                        torch.nn.Linear(module.in_features, adapter_dim, bias=False),
                        torch.nn.Linear(adapter_dim, module.out_features, bias=False)
                        ) 
                torch.nn.init.zeros_(module.adapter[1].weight)

