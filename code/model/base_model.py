import torch
import torch.nn as nn

from transformers import  BertModel

class BertBaseModel(nn.Module):
    """
        Class for language-alone baseline model using Bert Model for VQA task.
    """
    def __init__(self, D_in, H, D_out, dropout, freeze_bert=False):
        """
        @param bert: a BertModel object
        @param classifier: a torch.nn.Module classifier
        @param freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertBaseModel, self).__init__()

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.fc = nn.Linear(D_in, H)
        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(H, D_out)
        )
        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param input_ids (torch.Tensor): an input tensor with shape (batch_size, max_length)
        @param attention_mask (torch.Tensor): a tensor that hold attention mask information with shape (batch_size, max_length)
        @return logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[:, 0]
        #Compute question representation for the last hidden state from Bert
        q_repr = self.fc(last_hidden_state_cls)
        # Feed input to classifier to compute logits
        logits = self.classifier(q_repr)
        return logits
