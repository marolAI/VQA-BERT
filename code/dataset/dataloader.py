import numpy as np
import torch 

from utilities.data_utils import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer


def qaloader(dataroot, batch_size, num_workers, split):
    if split == 'train':
        print('[DATASET] Preparing %s dataset...'%split)
        train_dset = VQADataset(dataroot, split)
        train_loader = DataLoader(train_dset, 
                                  sampler=RandomSampler(train_dset),
                                  shuffle=True,
                                  batch_size=batch_size, 
                                  num_workers=num_workers)
        print('[DATASET] Preparing %s dataset...DONE'%split)
        return train_loader
    elif split == 'val':
        print('[DATASET] Preparing %s dataset...'%split)
        val_dset = VQADataset(dataroot, split)
        eval_loader =  DataLoader(val_dset, 
                                  sampler=SequentialSampler(val_dset),
                                  batch_size=batch_size, 
                                  num_workers=num_workers)
        print('[DATASET] Preparing %s dataset...'%split)
        return eval_loader

class VQADataset(Dataset):
    def __init__(self, dataroot, split, num_ans=1000):
        super(VQADataset, self).__init__()
        assert split in ['train', 'val']

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.num_ans_candidates = num_ans
        self.entries = load_dataset(dataroot, split)
        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=14):
        for entry in self.entries:
            encoded_dict = self.tokenizer.encode_plus(
                                                      entry['question'], # Question to encode.
                                                      add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                                      max_length = max_length,  # Pad & truncate all sentences.
                                                      padding='max_length',
                                                      truncation=True,
                                                      return_attention_mask = True, # Construct attn. masks.
                                                      return_tensors = 'pt',  #Return pytorch tensors.
                                            )
        
            entry['bert_ids'] = np.array(encoded_dict['input_ids'], np.int32).ravel()
            entry['bert_mask'] = np.array(encoded_dict['attention_mask'], np.int32).ravel()

    def tensorize(self):
        for entry in self.entries:
            qids = torch.from_numpy(np.array(entry['bert_ids']))
            q_mask = torch.from_numpy(np.array(entry['bert_mask']))
            entry['bert_ids'] = qids
            entry['bert_mask'] = q_mask

            answer = entry['answer']
            answers_scores = np.array(answer['answers_scores'], dtype=np.float32)
            answers_indices = np.array(answer['answers_indices'], dtype=np.float32)
            if answers_indices.size:
                answers_indices = torch.from_numpy(answers_indices)
                answers_scores = torch.from_numpy(answers_scores)
                entry['answer']['answers_indices'] = answers_indices
                entry['answer']['answers_scores'] = answers_scores
            else:
                entry['answer']['answers_indices'] = None
                entry['answer']['answers_scores'] = None
           

    def __getitem__(self, index):
        entry = self.entries[index]
        q_ids = entry['bert_ids']
        q_mask = entry['bert_mask']

        answer = entry['answer']
        answers_indices = answer['answers_indices']
        answers_scores = answer['answers_scores']
        return q_ids, q_mask, answers_indices, answers_scores

    def __len__(self):
        return len(self.entries)