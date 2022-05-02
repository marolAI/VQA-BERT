import torch
import os 

# General parameters
seed = 42
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                    

#data_parameters 
dataroot = 'data'
batch_size = 16
num_workers = 2
train = 'train'
val = 'val'
                

# model_parameters 
q_embedding_size = 1024
drop = 0.5
num_hid = 768
num_answers = 1000
freeze_bert = False
                    

# training_parameters 
epochs = 4
lr = 5e-5
eps = 1e-8
warmup_steps = 0
use_evalai = False
resume_epoch = 0
vqa_bert_model_path_resume = ''
snapshots_dir = 'outputs/snapshots'
outputs_dir = 'outputs/preprocessed'
answer_vocab = os.path.join(outputs_dir, 'answers_top%s_vocab.txt' %num_answers)
                        
