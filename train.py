from vocabulary import Vocabulary
import sys
from lstm import LSTM_LM
from datasource import DataSource

if __name__ == "__main__":
    voc = Vocabulary()
    voc.load_from_file("data/vocabulary.train")

    pad_idx = voc.voc["<pad>"].idx
    data_source = DataSource("data/encoded.train", pad_idx)

    model = LSTM_LM(voc, is_training=True, exp_name=sys.argv[1])

    model.create_model()
    emb_path = None
    if len(sys.argv) >= 3:
        print("Loading embeddings from %s" % (sys.argv[2]))
        emb_path = sys.argv[2]
    if len(sys.argv) >= 4:
        model.train(data_source, pretrained_embeddings_path=emb_path, MAX_ITERS=int(sys.argv[3]))
    else:
        model.train(data_source, pretrained_embeddings_path=emb_path)
