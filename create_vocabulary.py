import sys
from vocabulary import Vocabulary

VOC_SIZE = 20000
CORPUS_SIZE = -1

if __name__ == "__main__":
    voc = Vocabulary(VOC_SIZE, CORPUS_SIZE)
    voc.init(sys.argv[1])
    voc.dump_to_file(sys.argv[2])
