from vocabulary import Vocabulary
import sys
from lstm import LSTM_LM
from datasource import DataSource

if __name__ == "__main__":
    voc = Vocabulary()
    voc.load_from_file("data/vocabulary.train")
    
    pad_idx = voc.voc["<pad>"].idx
    data_source = DataSource("data/encoded.test", pad_idx)

    model = LSTM_LM(voc, is_training=False, exp_name=sys.argv[1])
    model.create_model()
    if len(sys.argv) < 4:
        print("python eval.py <exp name> <saved-model-name> <iters>")
        sys.exit(1)
    model.eval(data_source, sys.argv[2], MAX_ITERS=int(sys.argv[3]))
