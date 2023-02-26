from torch.utils.data import Dataset
import pandas as pd
import torch


def balance_dataframe(df, column_label):
    count_classes = df[column_label].value_counts()
    min_class = min(count_classes)
    balanced_df = pd.DataFrame()
    
    for class_index, _ in count_classes.items():
        class_df = df[df[column_label] == class_index]
        balanced_class_df = class_df.sample(min_class)
        balanced_df = balanced_df.append(balanced_class_df)
    return balanced_df

def prompt_tokenize(prompt, completion, tokenizer, max_len, truncation=True, padding=True):
    prompt_toks =  tokenizer.encode(prompt)
    
    completion_toks = tokenizer.encode(completion)
    if truncation:
        prompt_toks = prompt_toks[:max_len - len(completion_toks)]
    sample = torch.tensor(prompt_toks + completion_toks, dtype=int).unsqueeze(0)
    loss_mask = torch.zeros((1, sample.shape[1]), dtype=bool)
    loss_mask[:, list(range(len(prompt_toks), len(prompt_toks) + len(completion_toks)))] = True
    attention_mask = torch.ones(sample.shape, dtype=int)
    if padding:
        pad_zeros = torch.nn.ConstantPad1d((0, max_len - sample.shape[1]), 0)
        pad_eos = torch.nn.ConstantPad1d((0, max_len - sample.shape[1]), tokenizer.pad_token_id)
        
        sample = pad_eos(sample)
        loss_mask = pad_zeros(loss_mask)
        attention_mask = pad_zeros(attention_mask)
    return sample, attention_mask, loss_mask


class PromptDataset(Dataset):

    @staticmethod
    def create_prompt(text):
        prompt =  f''' Classify the following messages into one of the following categories: [Hate Speech], [Offensive language], [Neutral]

Message: {text}

Category: '''
        return prompt


    def __init__(self, data_df, tokenizer, max_prompt_len=100, truncation=True, padding=True):
        self.df = data_df
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.truncation = truncation
        self.padding = padding

    def __getitem__(self, idx):
        
        data = self.df.iloc[idx]
        prompt = data['prompt']
        completion = data['completion']
        input_ids, attention_mask, loss_mask = prompt_tokenize(prompt, completion, self.tokenizer, self.max_prompt_len, self.truncation, self.padding)
        return  input_ids, attention_mask, loss_mask
    
    def __len__(self):
        return len(self.df)
    