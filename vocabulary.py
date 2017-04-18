import operator


class Vocabulary:
    def __init__(self, voc_size=20000, corpus_size=-1):
        self.voc_size = voc_size
        self.corpus_size = corpus_size
        self.sorted_voc = []

    def init(self, corpus_file):
        voc = {}
        with open(corpus_file, "r") as fin:
            lines = fin.readlines()
            for line in lines[:self.corpus_size]:
                tokens = line.strip('\n').split(' ')
                for tok in tokens:
                    voc[tok] = voc.get(tok, 0) + 1

        self.sorted_voc = sorted(
                voc.items(),
                key=operator.itemgetter(1),
                reverse=True)

    def load_from_file(self, voc_file):
        with open(voc_file, "r") as fin:
            lines = fin.readlines()
            self.sorted_voc = [tuple(line.rsplit(',', 1)) for line in lines]

    def dump_to_file(self, out_file):
        with open(out_file, "w") as fout:
            for item in self.sorted_voc[:self.voc_size]:
                fout.write("{0},{1}\n".format(item[0], item[1]))
