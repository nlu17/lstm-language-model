from vocabulary import Vocabulary
import sys
from lstm import LSTM_LM
from datasource import DataSource
from lstm_downsampling import LSTM_C
import tensorflow as tf

if __name__ == "__main__":
    voc = Vocabulary()
    voc.load_from_file("data/vocabulary.train")

    pad_idx = voc.voc["<pad>"].idx
    data_source_train = DataSource("data/encoded.train", pad_idx)
    data_source_test = DataSource("data/encoded.test", pad_idx)

    ### Creation
    if sys.argv[1] == 'c':
        model = LSTM_C(voc, is_training=True, exp_name=sys.argv[1])
    else:
        model = LSTM_LM(voc, is_training=True, exp_name=sys.argv[1])

    model.create_model()
        
    with tf.Session() as sess:
        ### Training
        if sys.argv[1] == 'a':
            model.train(sess, data_source_train)
        else:
            model.train(sess, data_source_train, 
                    pretrained_embeddings_path='data/wordembeddings-dim100.word2vec')

        ### Perplexity
        model.eval(sess, data_source_test)

        ### Generation
        if sys.argv[1] == 'c':
            CONTINUATION_FILE = "data/sentences.continuation"
            OUTPUT_GEN = "data/continuation.output"
            with open(CONTINUATION_FILE, "r") as fin, \
                open(OUTPUT_GEN, "w") as fout:
                lines = [line.strip("\n") for line in fin.readlines()]
                model.load_model(sess)

                for line in lines:
                    pred = model.generate(sess, line)
                    fout.write(pred + "\n")

