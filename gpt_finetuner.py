from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.nn.functional import cross_entropy
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import transformers
from bitsandbytes.optim import Adam8bit

from gpt_quant_modules import GPTJBlock, GPTJForCausalLM



transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock 

@dataclass
class FinetunerConfig():
    lr: float = 1e-3
    batch_size: int = 1
    warmup_steps: int = 0
    num_epochs: int = 1
    adapter_dim: int = 2
    classification: bool = False

class GPTJ8bitFineTuner(pl.LightningModule):
    def __init__(self, model_name, model_post_init_func, fine_tuning_config, train_dataset, val_dataset=None):
        super().__init__()
        self.model = GPTJForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        self.config = fine_tuning_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if model_post_init_func:
            model_post_init_func(self.model)

    def forward(self, input_ids, attention_masks):
        return self.model.forward(
                            input_ids=input_ids,
                            attention_mask=attention_masks
                            )

    def common_step(self, batch, batch_idx):
        input_ids, attention_masks, loss_mask = batch
        
        out = self(
                    input_ids=input_ids,
                    attention_masks=attention_masks
                    )
        
        
        logits = out.logits[loss_mask.roll(shifts=-1, dims=2)]
        labels = input_ids[loss_mask]
        loss = cross_entropy(logits, labels)
        preds = None
        if self.config.classification:
            preds = torch.argmax(logits, dim=1)
        
        return loss, preds, labels


    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.common_step(batch, batch_idx)
        self.log('val_loss', loss)
        if self.config.classification:
            trues = torch.sum(preds == labels).cpu()
            total = len(labels)
            return loss, trues, total
        return loss, None, None
    
    def validation_epoch_end(self, validation_step_outputs):
        self.log('val_epoch_loss', np.mean([e[0].cpu() for e in validation_step_outputs]))
        if self.config.classification:
            self.log('val_epoch_accuracy', np.sum([e[1] for e in validation_step_outputs]) /  np.sum([e[2] for e in validation_step_outputs]))


    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, 
                                                   batch_size=self.config.batch_size, 
                                                  shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        if self.val_dataset:
            val_dataloader = torch.utils.data.DataLoader(self.val_dataset, 
                                                    batch_size=self.config.batch_size, 
                                                    shuffle=True)
            return val_dataloader

    # def configure_optimizers(self):
    #     optimizer = Adam8bit(self.model.parameters(), lr=self.config.lr)
    #     scheduler = transformers.get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=self.config.warmup_steps,
    #         num_training_steps=len(self.train_dataset),
    #     )
    #     scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 2}
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = Adam8bit(self.model.parameters(), lr=self.config.lr)

        return optimizer


