from load_embeddings import load_embedding
import tensorflow as tf
from vocabulary import Vocabulary

EMB_SIZE = 100
LSTM_HIDDEN = 512
BATCH_SIZE = 64
CLIP_NORM = 10


class LSTM_LM:
    def __init__(self, session, vocab):
        self.sess = session
        self.vocab = vocab
        self.embeddings = tf.Variable(tf.zeros([vocab.voc_size, EMB_SIZE]))

    def init_inputs(self, path):
        load_embedding(self.sess, self.vocab, self.embeddings, path, EMB_SIZE)

if __name__ == "__main__":
    voc = Vocabulary()
    voc.load_from_file("data/vocabulary.train")
    with tf.Session() as sess:
        model = LSTM_LM(sess, voc)
        model.init_inputs("wordembeddings-dim100.word2vec")
        print(model.embeddings.get_shape())
        print(model.embeddings[0])
