from load_embeddings import load_embedding
import tensorflow as tf
from vocabulary import Vocabulary

EMB_SIZE = 100
SEQ_LEN = 30
LSTM_HIDDEN = 512
BATCH_SIZE = 64
CLIP_NORM = 10  # TODO: clip the norms

LEARNING_RATE = 0.001
DISPLAY_STEP = 100
MAX_ITERS = 1000


class LSTM_LM:
    def __init__(self, session, vocab, is_training, keep_prob=1):
        self.sess = session
        self.vocab = vocab
        self.is_training = is_training
        self.keep_prob = keep_prob

        self.embeddings = tf.Variable(tf.zeros([vocab.voc_size, EMB_SIZE]))

        self.softmax_w = tf.get_variable(
            "softmax_w",
            [LSTM_HIDDEN, self.vocab.voc_size],
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32)
        self.softmax_b = tf.get_variable(
            "softmax_b",
            [self.vocab.voc_size],
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32)

    def init_inputs(self, path):
        # TODO: needs rethinking; rather load the embedding matrix to be used
        # with embedding_lookup
        load_embedding(self.sess, self.vocab, self.embeddings, path, EMB_SIZE)

    def get_model(self):
        self.init_weights = tf.global_variables_initializer()

        self.input_data = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LEN])
        self.targets = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LEN])

        # TODO: make it so we can initialize the embeddings with a matrix of
        # pretrained embeddings.
        emb_initializer = tf.contrib.layers.xavier_initializer()
        embedding = tf.get_variable(
            "embedding",
            [self.vocab.voc_size, EMB_SIZE],
            initializer=emb_initializer,
            dtype=tf.float32)
        emb_inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        if self.is_training and self.keep_prob < 1:
            emb_inputs = tf.nn.dropout(emb_inputs, self.keep_prob)

        cell = tf.contrib.rnn.BasicLSTMCell(LSTM_HIDDEN, state_is_tuple=True)
        if self.is_training and self.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=self.keep_prob)

        initial_state = cell.zero_state(BATCH_SIZE, tf.float32)

        print("inputs", emb_inputs)
        outputs = []
        state = initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(SEQ_LEN):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(emb_inputs[:, time_step, :], state)
                outputs.append(cell_output)
        final_state = state
        print("final_state", final_state)
        print("outputs", outputs)

        output = tf.reshape(
            tf.concat(axis=1, values=outputs),
            [-1, LSTM_HIDDEN])
        logits = tf.matmul(output, self.softmax_w) + self.softmax_b
        print("logits", logits)
        # TODO: replace loss with sparse_softmax_cross_entropy_with_logits
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([BATCH_SIZE * SEQ_LEN], dtype=tf.float32)])
        print("loss", loss)
        self.cost = tf.reduce_sum(loss) / BATCH_SIZE
        print("cost", self.cost)
        final_state = state

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=LEARNING_RATE) .minimize(self.cost)
        correct_pred = tf.equal(
            tf.argmax(self.logits, 1),
            tf.argmax(self.targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def train(self):
        with tf.Session() as sess:
            sess.run(self.init_weights)
            step = 1
            # Keep training until reach max iterations
            while step * BATCH_SIZE < MAX_ITERS:
                # TODO: get next batch
                batch_x, batch_y = next_train_batch(
                        self.batch_size)
                # Run optimization op (backprop)
                sess.run(
                        self.optimizer,
                        feed_dict={
                            self.input_data: batch_x,
                            self.targets: batch_y,
                            self.keep_prob: self.dropout})
                if step % DISPLAY_STEP == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run(
                            [self.cost, self.accuracy],
                            feed_dict={
                                    self.input_data: batch_x,
                                    self.targets: batch_y,
                                    self.keep_prob: 1})
                    print("Iter " + str(step*BATCH_SIZE) +
                          ", Minibatch Loss = {:.6f}".format(loss) +
                          ", Training Accuracy = {:.5f}".format(acc))
                step += 1


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
