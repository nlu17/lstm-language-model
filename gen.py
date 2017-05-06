from datasource import DataSource
from lstm import LSTM_LM
import sys
import tensorflow as tf
from vocabulary import Vocabulary

if __name__ == "__main__":
    voc = Vocabulary()
    voc.load_from_file("data/vocabulary.train")

#     pad_idx = voc.voc["<pad>"].idx
#     data_source = DataSource("data/encoded.train", pad_idx)

    model = LSTM_LM(voc, is_training=True, exp_name=sys.argv[1])

    model.create_model()
    with tf.Session() as sess:
        model.load_model(sess)
#     emb_path = None
#     if len(sys.argv) >= 3:
#         print("Loading embeddings from %s" % (sys.argv[2]))
#         emb_path = sys.argv[2]
#     model.train(data_source, emb_path)
