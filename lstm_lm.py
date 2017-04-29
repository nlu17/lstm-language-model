from load_embeddings import load_embedding
import tensorflow as tf
from vocabulary import Vocabulary

EMB_SIZE = 100
SEQ_LEN = 30
LSTM_HIDDEN = 512
BATCH_SIZE = 64
CLIP_NORM = 10


class LSTM_LM:
    def __init__(self, session, vocab, is_training, keep_prob=1):
        self.sess = session
        self.vocab = vocab
        self.is_training = is_training
        self.keep_prob = keep_prob

        self.embeddings = tf.Variable(tf.zeros([vocab.voc_size, EMB_SIZE]))
        self.input_size = EMB_SIZE

    def init_inputs(self, path):
        load_embedding(self.sess, self.vocab, self.embeddings, path, EMB_SIZE)

    def get_model(self):
        input_data = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LEN])
        targets = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LEN])

        embedding = tf.get_variable(
                "embedding", [self.vocab.voc_size, EMB_SIZE], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, input_data)
        if self.is_training and self.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self.keep_prob)

        cell = tf.contrib.rnn.BasicLSTMCell(LSTM_HIDDEN, state_is_tuple=True)
        if self.is_training and self.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=self.keep_prob)

        initial_state = cell.zero_state(BATCH_SIZE, tf.float32)

        print("inputs", inputs)
        outputs = []
        state = initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(SEQ_LEN):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        final_state = state
        print("final_state", final_state)
        print("outputs", outputs)

        output = tf.reshape(
            tf.concat(axis=1, values=outputs),
            [-1, LSTM_HIDDEN])
        softmax_w = tf.get_variable(
            "softmax_w", [LSTM_HIDDEN, self.vocab.voc_size], dtype=tf.float32)
        softmax_b = tf.get_variable(
            "softmax_b", [self.vocab.voc_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        print("logits", logits)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(targets, [-1])],
            [tf.ones([BATCH_SIZE * SEQ_LEN],dtype=tf.float32)])
        print("loss", loss)
        cost = tf.reduce_sum(loss) / BATCH_SIZE
        print("cost", cost)
        final_state = state

        if not self.is_training:
            return


if __name__ == "__main__":
    voc = Vocabulary()
    voc.load_from_file("data/vocabulary.train")
    with tf.Session() as sess:
        model = LSTM_LM(sess, voc, is_training=True)
        model.init_inputs("wordembeddings-dim100.word2vec")
        print(model.embeddings.get_shape())
        idx = model.vocab.voc["something"].idx
        print(model.sess.run(model.embeddings[idx]))

        model.get_model()
