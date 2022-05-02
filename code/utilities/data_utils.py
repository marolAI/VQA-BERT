import os
import pickle 

from ..dataset.qa import Database


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

def create_entry(question, answer):
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'question'    : question['question'],
        'answer'      : answer}
    return entry

def load_dataset(dataroot, split):
    """Load entries
    dataroot: root path of dataset
    split: 'train', 'val'
    """
    if split == 'train':
        train2014 = Database(dataroot, '%s2014' %split)
        train_questions = train2014.load('questions')
        questions = sorted(train_questions, key=lambda x: x['question_id'])
    elif split == 'val':
        val2014 = Database(dataroot, '%s2014' %split)
        val_questions = val2014.load('questions')
        questions = sorted(val_questions, key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'preprocessed', '%s_target.pkl' % split)
    answers = sorted(pickle.load(open(answer_path, 'rb')), key=lambda x: x['question_id'])

    assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        assert_eq(question['question_id'], answer['question_id'])
        entries.append(create_entry(question, answer))
    return entries