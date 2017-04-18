import operator
import sys

VOC_SIZE = 20000
CORPUS_SIZE = -1

def create_vocabulary(corpus, out_file):
    voc = {}
    with open(corpus, "r") as fin:
        lines = fin.readlines()
        for line in lines[:CORPUS_SIZE]:
            tokens = line.strip('\n').split(' ')
            for tok in tokens:
                voc[tok] = voc.get(tok, 0) + 1

    with open(out_file, "w") as fout:
        sorted_voc = sorted(
                voc.items(),
                key=operator.itemgetter(1),
                reverse=True)
        for item in sorted_voc[:VOC_SIZE]:
            fout.write("{0},{1}\n".format(item[0], item[1]))


if __name__ == "__main__":
    create_vocabulary(sys.argv[1], sys.argv[2])
