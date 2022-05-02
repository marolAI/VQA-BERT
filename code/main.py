import torch

from dataset.dataloader import qaloader
from model.base_model import BertBaseModel
from train.train_baseline import train
from utilities.train_utils import set_seed
from utilities.config import *

def main():
    """
        This is the main function for training Bert baseline model for VQA
    """
    #set seed for reproducibility
    set_seed(seed) 
    #load train and val dataset into memory
    train_dataloader = qaloader(dataroot, batch_size, num_workers, train) 
    val_dataloader = qaloader(dataroot, batch_size, num_workers, val)
    #instantiate Bert baseline model
    bert_base_model = BertBaseModel(D_in=q_embedding_size, H=num_hid, D_out=num_answers, dropout=drop, freeze_bert=False)
    #if vqa_bert_model_path_resume is not empty then resume the last save model
    if vqa_bert_model_path_resume != '':
        bert_base_model.load_state_dict(torch.load(vqa_bert_model_path_resume))
    #train Bert baseline model for VQA for 4 epochs 
    train(bert_base_model.cuda(), train_dataloader, val_dataloader, answer_vocab, epochs=4, evaluation=False)

if __name__ == '__main__':
    main()
    