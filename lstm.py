from load_embeddings import load_embedding
import numpy as np
import tensorflow as tf

EMB_SIZE = 100
SEQ_LEN = 30
LSTM_HIDDEN = 512
BATCH_SIZE = 64
CLIP_NORM = 10

LEARNING_RATE = 0.001
DISPLAY_STEP = 10
SAVE_STEP = 10000
MAGIC = 2000000
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

    """
    Creates computation graph
    """
    def create_model(self):
        self.input_data = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LEN])
        self.targets = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LEN])

        self.embeddings = tf.get_variable(
            "embeddings",
            [self.vocab.voc_size, EMB_SIZE],
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32)

        self.emb_inputs = tf.nn.embedding_lookup(
            self.embeddings,
            self.input_data)

        self.cell = tf.contrib.rnn.BasicLSTMCell(
            LSTM_HIDDEN,
            state_is_tuple=True)

        # Define nodes for prediction.
        self.curr_word_emb = tf.placeholder(tf.float32, [None, EMB_SIZE])
        self.prev_state = \
            (tf.placeholder(tf.float32, [None, LSTM_HIDDEN]),
             tf.placeholder(tf.float32, [None, LSTM_HIDDEN]))
        self.run_cell = self.cell(self.curr_word_emb, self.prev_state)

        self.initial_state = self.cell.zero_state(BATCH_SIZE, tf.float32)

        outputs = []
        with tf.variable_scope("RNN"):
            state = self.initial_state
            for time_step in range(SEQ_LEN):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.cell(
                    self.emb_inputs[:, time_step, :],
                    state)
                outputs.append(cell_output)

        output = tf.reshape(
            tf.concat(axis=1, values=outputs),                          # (B, S*H)
            [-1, LSTM_HIDDEN])                                          # (B*S, H)
        logits = tf.matmul(output, self.softmax_w) + self.softmax_b     # (B*S, V)

        """
        <Perplexity computation>
        """
        flat_targets = tf.reshape(self.targets, [-1])
        nums = tf.range(flat_targets.shape[0])
        print("n shape ", nums.shape, " targets shape ", flat_targets.shape)
        idx = tf.transpose(
                tf.concat([tf.reshape(nums, [1, -1]), tf.reshape(flat_targets, [1, -1])],
                axis=0))
        print("idx shape", idx.shape)
        log_word_probs = tf.nn.log_softmax(logits, name="word_probs")
        log_probs = tf.reshape(
                tf.gather_nd(log_word_probs, idx),                       # (B*S)
                [-1, SEQ_LEN])                                           # (B, S)
        print("log probs shape", log_probs.shape)
        zeros = tf.zeros_like(log_probs)
        ones = tf.ones_like(log_probs)
        pads = tf.fill(log_probs.shape, self.vocab.voc["pad"], name="pads")
        mask = tf.equal(self.input_data, pads)
        sum_perplexity = tf.reduce_sum(
                tf.where(mask, zeros, log_probs),
                axis=1,
                name = "sum_perplexity"
                )
        count_perplexity = tf.reduce_sum(
                tf.where(mask, zeros, ones),
                axis=1,
                name = "count_perplexity"
                )
        avg_perplexity = tf.div(sum_perplexity, count_perplexity, name="avg_perplexity")
        two = tf.fill(avg_perplexity.shape, 2.)
        self.perplexity = tf.pow(two, avg_perplexity, name="perplexity")
        print("perplexity shape", self.perplexity.shape)
        """
        </Perplexity computation>
        """

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=[flat_targets],
            logits=[logits])

        print("loss", loss)
        self.cost = tf.reduce_sum(loss) / BATCH_SIZE
        print("cost", self.cost)

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
        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(sess, filename)

    """
    Loads the model from a file
    """
    def load_model(self, sess, filename="final"):
        filename = "model/" + self.exp_name + "_" + filename + ".ckpt"
        print("loading from", filename)
        saver = tf.train.Saver()
        saver.restore(sess, filename)

    """
    Train function, takes pretrained embeddings as optional argument
    """
    def train(self, data_source, MAX_ITERS = MAGIC, pretrained_embeddings_path=None):
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
            while step * BATCH_SIZE <= MAX_ITERS:
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

    """
    Performs conditional generation of sentences based on the trained
    language model.
    """
    def generate(self, sess, sentence, max_length=MAX_GEN_LENGTH):
        # TODO: take into account that max_length doesn't count <bos> too
        # init_seq doesn't start with <bos>

        tokens = sentence.split(' ')
        tokens = [tok if tok in self.vocab.voc else "<unk>" for tok in tokens]
        tokens = ["<bos>"] + tokens
        init_seq = self.vocab.get_tok_ids(tokens)

        step = 0
        state = (np.zeros([1, LSTM_HIDDEN]), np.zeros([1, LSTM_HIDDEN]))
        w_value = sess.run(self.softmax_w)
        b_value = sess.run(self.softmax_b)
        embeddings = sess.run(self.embeddings)

        while step < len(init_seq) - 1:
            curr_emb = embeddings[init_seq[step]].reshape([1, EMB_SIZE])
            _, state = sess.run(
                self.run_cell,
                feed_dict={
                    self.curr_word_emb: curr_emb,
                    self.prev_state: state})
            step += 1

        curr_emb = embeddings[init_seq[len(init_seq)-1]].reshape([1, EMB_SIZE])

        while step < MAX_GEN_LENGTH:
            cell_output, state = sess.run(
                self.run_cell,
                feed_dict={
                    self.curr_word_emb: curr_emb,
                    self.prev_state: state})
            logits = np.dot(cell_output, w_value) + b_value
            next_word_id = np.argmax(logits)
            next_word = self.vocab.sorted_voc[next_word_id][0]
            if next_word == "eos":
                sentence += " <eos>"
                break

            curr_emb = embeddings[next_word_id].reshape([1, EMB_SIZE])

            sentence += " " + next_word
            step += 1

        return sentence

    """
    Evaluates sentence perplexity for each sentence from data_source
    """
    def eval(self, data_source, model_name="final", MAX_ITERS=BATCH_SIZE):
        with tf.Session() as sess:
            self.load_model(sess, model_name)
            step = 0                                            # last few values will be repeated
            idx = 0
            while step * BATCH_SIZE <= MAX_ITERS:
                # Get next batch.
                batch_inputs, batch_targets = \
                    data_source.next_train_batch(BATCH_SIZE)
                # Run perplexity
                perplexity = sess.run(
                        self.perplexity,
                        feed_dict={
                            self.input_data: batch_inputs,
                            self.targets: batch_targets})
                for p in perplexity:
                    print(idx, p)
                    idx += 1
                step += 1
