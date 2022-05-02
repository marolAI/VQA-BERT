import numpy as np
import os
import pickle

from collections import defaultdict
from ..dataset.qa import Database
from ..utilities.answer_utils import EvalAIAnswerPreprocessor, compute_answers_scores
from ..utilities.config import *
from ..utilities.vocab import VocabDict


def filter_answers(answers_dset, evalai_answer_processor, num_ans):
    print("[FILTER ANSWERS] Top %s answers..." % num_ans)
    answers_dict = defaultdict(lambda: 0)
    for ans in answers_dset:
        gtruth = ans["multiple_choice_answer"]
        gtruth = evalai_answer_processor(gtruth)
        answers_dict[gtruth] += 1
    answers_list = sorted(answers_dict, key=answers_dict.get, reverse=True)
    answers_list = [t.strip() for t in answers_list if len(t.strip()) > 0]
    assert "<unk>" not in answers_list
    answers_list = ["<unk>"] + answers_list[: num_ans - 1]
    print("[FILTER ANSWERS] Top %s answers...DONE" % num_ans)
    return answers_list


def extract_answers(annotations, valid_answers_list, evalai_answer_processor):
    print("[EXTRACT ANSWERS] ...")
    answers = [None] * len(annotations)
    for i in range(len(annotations)):
        question_id = annotations[i]["question_id"]
        all_answers = [
            evalai_answer_processor(answer["answer"])
            for answer in annotations[i]["answers"]
        ]

        valid_answers = [None] * len(all_answers)
        for j, answer in enumerate(all_answers):
            if answer in valid_answers_list:
                valid_answers[j] = answer
            else:
                valid_answers[j] = "<unk>"

        answers[i] = dict(question_id=question_id, valid_answers=valid_answers)
    print("[EXTRACT ANSWERS] ...DONE")
    return answers


def compute_target(dataroot, outputs_dir, evalai_answer_processor, num_ans, split):
    print("[COMPUTE TARGET] %s set." % split)
    train2014 = Database(dataroot, "train2014")
    train_annotations = train2014.load("annotations")
    val2014 = Database(dataroot, "val2014")
    val_annotations = val2014.load("annotations")

    answers_dset = train_annotations + val_annotations
    vocab_entry = filter_answers(answers_dset, evalai_answer_processor, num_ans)

    answer_file_name = "answers_top%s_vocab.txt" % num_ans
    answer_file = (os.path.join(outputs_dir, answer_file_name),)
    with open(answer_file, "w") as f:
        f.writelines([w + "\n" for w in vocab_entry])

    answer_vocab = VocabDict(vocab_entry)
    valid_answers_list = answer_vocab.word_lists

    if split == "train":
        ans = extract_answers(
            train_annotations, valid_answers_list, evalai_answer_processor
        )
    elif split == "val":
        ans = extract_answers(
            val_annotations, valid_answers_list, evalai_answer_processor
        )

    targets = []
    for i in range(len(ans)):
        question_id = ans[i]["question_id"]
        valid_answers = ans[i]["valid_answers"]
        answers_indices = [answer_vocab.word2idx(ans) for ans in valid_answers]
        answers_scores = compute_answers_scores(
            answers_indices, answer_vocab.num_vocab, answer_vocab.UNK_idx
        )

        targets.append(
            {
                "question_id": question_id,
                "answers_indices": answers_indices,
                "answers_scores": answers_scores,
            }
        )
    target_file = os.path.join(outputs_dir, split + "_target.pkl")
    pickle.dump(targets, open(target_file, "wb"))
    print("[COMPUTE TARGET] %s set....DONE" % split)
    return


if __name__ == "__main__":
    evalai_answer_processor = EvalAIAnswerPreprocessor()
    compute_target(dataroot, outputs_dir, evalai_answer_processor, num_answers, train)
    compute_target(dataroot, outputs_dir, evalai_answer_processor, num_answers, val)
