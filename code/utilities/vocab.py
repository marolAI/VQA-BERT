

class VocabDict:
    def __init__(self, vocab_entry):
        self.vocab_entry = vocab_entry
        self.word_list = self.load_str_list()
        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}
        self.num_vocab = len(self.word_list)
        self.UNK_idx = (self.word2idx_dict['<unk>'] if '<unk>' in self.word2idx_dict else None)

    def load_str_list(self):
        if isinstance(self.vocab_entry, list):
            lines = self.vocab_entry
        elif isinstance(self.vocab_entry, str):
            with open(self.vocab_entry) as f:
                lines = f.readlines()
            lines = [l.strip() for l in lines]
        return lines

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_idx is not None:
            return self.UNK_idx
        else:
            raise ValueError('word %s not in dictionary (while dictionary does not contain <unk>)' % w)