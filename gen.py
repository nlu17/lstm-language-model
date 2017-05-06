from datasource import DataSource
from lstm import LSTM_LM
import sys
import tensorflow as tf
from vocabulary import Vocabulary

CONTINUATION_FILE = "data/sentences.continuation"
OUTPUT_GEN = "data/continuation.output"

if __name__ == "__main__":
    voc = Vocabulary()
    voc.load_from_file("data/vocabulary.train")

    model = LSTM_LM(voc, is_training=True, exp_name=sys.argv[1])

    model.create_model()

    with tf.Session() as sess, open(CONTINUATION_FILE, "r") as fin, \
        open(OUTPUT_GEN, "w") as fout:
        lines = [line.strip("\n") for line in fin.readlines()]
        model.load_model(sess)

        for line in lines:
            pred = model.generate(sess, line)
            fout.write(pred + "\n")
