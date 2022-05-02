import torch
import numpy as np
import random

from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F


def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value)


def compute_loss(answer, pred, method):
    """
    answer = [batch, 1000]
    pred = [batch, 1000]
    """
    if method == "bce_with_logits":
        loss = F.binary_cross_entropy_with_logits(
            pred, answer, reduction="mean"
        ) * answer.size(1)
    elif method == "soft_cross_entropy":
        nll = -F.log_softmax(pred, dim=1)
        loss = (
            (nll * answer).sum(dim=1).mean()
        )  # this is worse than binary_cross_entropy_with_logits
    elif method == "KL_divergence":
        pred = F.softmax(pred, dim=1)
        kl = ((answer / (pred + 1e-12)) + 1e-12).log()
        loss = (kl * answer).sum(1).mean()
    elif method == "multi_label_soft_margin":
        loss = F.multilabel_soft_margin_loss(pred, answer)
    else:
        print("Error, please define loss function")
    return loss


def masked_unk_softmax(x, dim, mask_idx):
    x1 = F.softmax(x, dim=dim)
    x1[:, mask_idx] = 0
    x1_sum = torch.sum(x1, dim=1, keepdim=True)
    y = x1 / x1_sum
    return y


def compute_VQA_accuracy(expected, output):
    output = masked_unk_softmax(output, 1, 0)
    output = output.argmax(dim=1)  # argmax
    one_hots = expected.new_zeros(*expected.size())
    one_hots.scatter_(1, output.view(-1, 1), 1)
    scores = one_hots * expected
    accuracy = torch.sum(scores) / expected.size(0)
    return accuracy


def compute_VQAEvalAI_accuracy(
    answers_indices, output, answer_vocab, evalai_answer_processor
):
    output = masked_unk_softmax(output, 1, 0)
    output = output.argmax(dim=1).clone().tolist()
    expected = [answer_vocab.idx2word(ans_idx) for ans_idx in answers_indices]
    accuracy = []

    for idx, answer_id in enumerate(output):
        answer = answer_vocab.idx2word(answer_id)
        answer = evalai_answer_processor(answer)
        gt_answers = [evalai_answer_processor(x) for x in expected[idx]]
        gt_answers = list(enumerate(gt_answers))

        gt_acc = []
        for gt_answer in gt_answers:
            other_answers = [item for item in gt_answers if item != gt_answer]
            matching_answers = [item for item in other_answers if item[1] == answer]
            acc = min(1, float(len(matching_answers)) / 3)
            gt_acc.append(acc)
        avgGTAcc = float(sum(gt_acc)) / len(gt_acc)
        accuracy.append(avgGTAcc)
    accuracy = round(100 * float(sum(accuracy)) / len(accuracy), 2)
    return torch.Tensor(accuracy)


def get_optimizer(model, lr):
    return AdamW(model.parameters(), lr)


def get_scheduler(optimizer, warmup_steps, total_steps):
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
