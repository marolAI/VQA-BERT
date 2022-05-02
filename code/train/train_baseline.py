import time
import torch 
import numpy as np 
import os 
from torch.nn.utils import clip_grad_norm_

from utilities.answer_utils import EvalAIAnswerPreprocessor
from utilities.train_utils import compute_loss, compute_VQA_accuracy, compute_VQAEvalAI_accuracy, get_scheduler, get_optimizer
from utilities.config import *


os.makedirs(snapshots_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)


def train(model, train_dataloader, val_dataloader, answer_vocab, epochs=4, evaluation=False):
    """
        Training loop for the Bert baseline model for VQA task.
    """
    # Start training loop
    print("Start training...\n")

    optimizer = get_optimizer(model, lr)
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps=len(train_dataloader) * epochs)
    evalai_answer_processor = EvalAIAnswerPreprocessor()

    best_eval_acc = 0

    for epoch_i in range(epochs):
        # =======================================
        # Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10}| {'Train Acc':^12} |{'Val Acc':^9} | {'Elapsed time':^9}")
        print("-"*70)
        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()
        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss,total_acc, batch_acc, batch_counts = 0, 0, 0, 0, 0
        # Put the model into the training mode
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            q_ids, q_mask, answers_indices, answers_scores = tuple(t.cuda() for t in batch)
            # Zero out any previously calculated gradients
            optimizer.zero_grad()
            # Perform a forward pass. This will return logits.
            logits = model(q_ids, q_mask)
            # Compute loss and accumulate the loss values
            loss = compute_loss(answers_scores, logits, method='bce_with_logits')
            batch_loss += loss.item()
            total_loss += loss.item()
            # Perform a backward pass to calculate gradients
            loss.backward()
            
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()
            
            # Compute accuracy and accumulate the accuracy values
            if use_evalai:
                acc = compute_VQAEvalAI_accuracy(answers_indices, logits, answer_vocab, evalai_answer_processor)
            else:
                acc = compute_VQA_accuracy(answers_scores, logits)
            batch_acc += acc
            total_acc += acc
            
            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch
                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7}| {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^12} |{'-':^9} |{time_elapsed:^9.2f}")
                # Reset batch tracking variables
                batch_loss, batch_acc, batch_counts = 0, 0, 0
                t0_batch = time.time()
        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader.dataset)
        avg_train_acc = total_acc / len(train_dataloader.dataset)
        print("-"*70)

        # =======================================
        # Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {avg_train_acc:^12.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")

        with open(os.path.join(snapshots_dir, 'train-log-epoch-{:02}.txt').format(epoch_i+1), 'w') as f:
            f.write(str(epoch_i+1) + '\t'
                    + str(avg_train_loss) + '\t'
                    + str(avg_train_acc.detach().cpu().numpy()))
            
        with open(os.path.join(snapshots_dir, 'eval-log-epoch-{:02}.txt').format(epoch_i+1), 'w') as f:
            f.write(str(epoch_i+1) + '\t'
                    + str(val_loss) + '\t'
                    + str(val_accuracy.detach().cpu().numpy()))

        if val_accuracy > best_eval_acc:
            torch.save(model.state_dict(), os.path.join(snapshots_dir, 'vqa_bert_best_model.pth'))
            best_eval_acc = val_accuracy
    
    print("Training complete!")

def evaluate(model, val_dataloader, answer_vocab, evalai_answer_processor):
    """
    After the completion of each training epoch, measure the model's performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    # Tracking variables
    val_accuracy = []
    val_loss = []
    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        q_ids, q_mask, answers_indices, answers_scores = tuple(t.cuda() for t in batch)
        # Compute logits
        with torch.no_grad():
            logits = model(q_ids, q_mask)
        # Compute loss
        loss = compute_loss(answers_scores, logits, method='bce_with_logits')
        val_loss.append(loss.item())
        if use_evalai:
            acc = compute_VQAEvalAI_accuracy(answers_indices, logits, answer_vocab, evalai_answer_processor)
        else:
            acc = compute_VQA_accuracy(answers_scores, logits)
        val_accuracy.append(acc)
    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    return val_loss, val_accuracy