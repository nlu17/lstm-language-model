from load_embeddings import load_embedding
import tensorflow as tf

EMB_SIZE = 100
SEQ_LEN = 30
LSTM_HIDDEN = 512
BATCH_SIZE = 64
CLIP_NORM = 10

LEARNING_RATE = 0.001
DISPLAY_STEP = 10
SAVE_STEP = 1000
# MAX_ITERS = 2000000
MAX_ITERS = 2000

MAX_GEN_LENGTH = 20


class LSTM_LM:
    def __init__(self, vocab, is_training, exp_name):
        self.vocab = vocab
        self.is_training = is_training
        self.exp_name = exp_name

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

    def create_model(self):
        self.input_data = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LEN])
        self.targets = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LEN])

        self.embeddings = tf.get_variable(
            "embeddings",
            [self.vocab.voc_size, EMB_SIZE],
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32)

        emb_inputs = tf.nn.embedding_lookup(self.embeddings, self.input_data)

        self.cell = tf.contrib.rnn.BasicLSTMCell(LSTM_HIDDEN, state_is_tuple=True)

        initial_state = self.cell.zero_state(BATCH_SIZE, tf.float32)

        print("inputs", emb_inputs)
        outputs = []
        final_state = None
        with tf.variable_scope("RNN"):
            state = initial_state
            for time_step in range(SEQ_LEN):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.cell(emb_inputs[:, time_step, :], state)
                outputs.append(cell_output)
            final_state = state
        print("final_state", final_state)
        print("outputs", outputs)

        output = tf.reshape(
            tf.concat(axis=1, values=outputs),
            [-1, LSTM_HIDDEN])
        logits = tf.matmul(output, self.softmax_w) + self.softmax_b
        print("logits", logits)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=[tf.reshape(self.targets, [-1])],
            logits=[logits])

        print("loss", loss)
        self.cost = tf.reduce_sum(loss) / BATCH_SIZE
        print("cost", self.cost)
        final_state = state

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, tvars),
            CLIP_NORM)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=LEARNING_RATE).apply_gradients(zip(grads, tvars))
        correct_pred = tf.equal(
            tf.cast(tf.argmax(logits, 1), dtype=tf.int32),
            tf.reshape(self.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.init_weights = tf.global_variables_initializer()

    """
    Saves model in a file
    """
    def save_model(self, sess, filename="final"):
        filename = "model/" + self.exp_name + "_" + filename + ".ckpt"
        saver = tf.train.Saver()
        saver.save(sess, filename)

    """
    Loads the model from a file
    """
    def load_model(self, sess, filename="final"):
        filename = "model/" + self.exp_name + "_" + filename + ".ckpt"
        saver = tf.train.Saver()
        saver.restore(sess, filename)

    def train(self, data_source, pretrained_embeddings_path=None):
        with tf.Session() as sess:
            sess.run(self.init_weights)
            if pretrained_embeddings_path is not None:
                load_embedding(
                    sess,
                    self.vocab,
                    self.embeddings,
                    pretrained_embeddings_path,
                    EMB_SIZE)
                print("EMB", self.embeddings)

            step = 1
            # Keep training until reach max iterations
            while step * BATCH_SIZE < MAX_ITERS:
                # Get next batch.
                batch_inputs, batch_targets = \
                    data_source.next_train_batch(BATCH_SIZE)
                # Run optimization op (backprop)
                sess.run(
                        self.optimizer,
                        feed_dict={
                            self.input_data: batch_inputs,
                            self.targets: batch_targets})
                if step % DISPLAY_STEP == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run(
                            [self.cost, self.accuracy],
                            feed_dict={
                                    self.input_data: batch_inputs,
                                    self.targets: batch_targets})
                    print("Iter " + str(step*BATCH_SIZE) +
                          ", Minibatch Loss = {:.6f}".format(loss) +
                          ", Training Accuracy = {:.5f}".format(acc))
                if step % SAVE_STEP == 0:
                    # save model
                    print("Saving model iter %d" % step)
                    self.save_model(sess, str(step))

                step += 1
            print("Saving final model")
            self.save_model(sess, "final")

            # Test the model

        def create_generation_setup(self):


        """
        Performs conditional generation of sentences based on the trained
        language model.
        """
        def generate(self, init_seq, max_length=MAX_GEN_LENGTH):
            # TODO: take into account that max_length doesn't count <bos> too
            # init_sez doesn't start with <bos>
            pass
