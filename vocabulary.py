import operator


class Vocabulary:
    def __init__(self, voc_size=20000, corpus_size=-1, max_sentence_len=30):
        self.voc_size = voc_size
        self.corpus_size = corpus_size
        self.max_sentence_len = max_sentence_len
        self.voc = {}  # token -> (tokID, collectionCount)
        self.sorted_voc = []  # list of (token, (tokID, collectionCount)) tuples

    def init(self, corpus_file):
        tmp_dict = {}
        with open(corpus_file, "r") as fin:
            lines = fin.readlines()
            for line in lines[:self.corpus_size]:
                tokens = line.strip('\n').split(' ')
                for tok in tokens:
                    tmp_dict[tok] = tmp_dict.get(tok, 0) + 1

        self.sorted_voc = sorted(
                tmp_dict.items(),
                key=operator.itemgetter(1),
                reverse=True)
        self.voc = dict([(item[0], (index, item[1]))
                         for index, item in enumerate(self.sorted_voc)])

    """
    Get a sentence (string) as input and return a list of tokens. Returns None
    if the sentence size is above the threshold.
    """
    def parse(self, sentence):
        tokens = sentence.split(' ')
        if len(tokens) > self.max_sentence_len-2:
            return None

        # Check if there are out-of-vocabulary tokens.
        tokens = [tok if tok in self.voc else "<unk>" for tok in tokens]

        # Add bos/eos symbols.
        tokens = ["<bos>"] + tokens + ["<eos>"]

        # Pad the sentence.
        tokens = tokens + ["<pad>"] * (self.max_sentence_len - len(tokens))

        return tokens

    def load_from_file(self, voc_file):
        with open(voc_file, "r") as fin:
            lines = fin.readlines()

            def cast(l):
                return (l[0], (int(l[1]), int(l[2])))
            self.voc = \
                dict([cast(line.strip("\n").split(' ')) for line in lines])

    def dump_to_file(self, out_file):
        with open(out_file, "w") as fout:
            for index, item in enumerate(self.sorted_voc[:self.voc_size]):
                fout.write("{0} {1} {2}\n".format(item[0], index, item[1]))
