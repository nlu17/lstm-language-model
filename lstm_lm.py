from load_embeddings import load_embedding
import numpy as np
import tensorflow as tf
from vocabulary import Vocabulary

EMB_SIZE = 100
SEQ_LEN = 30
LSTM_HIDDEN = 512
BATCH_SIZE = 64
CLIP_NORM = 10  # TODO: clip the norms

LEARNING_RATE = 0.001
DISPLAY_STEP = 10
MAX_ITERS = 5000


class LSTM_LM:
    def __init__(self, vocab, data_source, is_training, dropout=1):
        self.vocab = vocab
        self.data_source = data_source
        self.is_training = is_training
        self.dropout = dropout

        #self.embeddings = tf.Variable(tf.zeros([vocab.voc_size, EMB_SIZE]))

        # Dropout (keep probability).
        self.keep_prob = tf.placeholder(tf.float32)

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

    def create_model(self, path=None):
        self.input_data = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LEN])
        self.targets = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LEN])

        # TODO: make it so we can initialize the embeddings with a matrix of
        # pretrained embeddings.
        embeddings = tf.Variable(tf.zeros([vocab.voc_size, EMB_SIZE]))
        if path is None:
            emb_initializer = tf.contrib.layers.xavier_initializer()
            embeddings = tf.get_variable(
                "embeddings",
                [self.vocab.voc_size, EMB_SIZE],
                initializer=emb_initializer,
                dtype=tf.float32)
        else:
            with tf.Session() as sess:
                load_embedding(self.sess, self.vocab, embeddings, path, EMB_SIZE)

        emb_inputs = tf.nn.embedding_lookup(embeddings, self.input_data)
        if self.is_training and self.dropout < 1:
            emb_inputs = tf.nn.dropout(emb_inputs, self.keep_prob)

        cell = tf.contrib.rnn.BasicLSTMCell(LSTM_HIDDEN, state_is_tuple=True)
        # TODO: get rid of dropout, saurav
        if self.is_training and self.dropout < 1:
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

    def train(self):
        with tf.Session() as sess:
            sess.run(self.init_weights)
            # TODO: run the assign op
            step = 1
            # Keep training until reach max iterations
            while step * BATCH_SIZE < MAX_ITERS:
                # TODO: get next batch
                batch_inputs, batch_targets = \
                    self.data_source.next_train_batch(BATCH_SIZE)
                # Run optimization op (backprop)
                sess.run(
                        self.optimizer,
                        feed_dict={
                            self.input_data: batch_inputs,
                            self.targets: batch_targets,
                            self.keep_prob: self.dropout})
                if step % DISPLAY_STEP == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run(
                            [self.cost, self.accuracy],
                            feed_dict={
                                    self.input_data: batch_inputs,
                                    self.targets: batch_targets,
                                    self.keep_prob: 1})
                    print("Iter " + str(step*BATCH_SIZE) +
                          ", Minibatch Loss = {:.6f}".format(loss) +
                          ", Training Accuracy = {:.5f}".format(acc))
                step += 1


class DataSource:
    def __init__(self, data_file, pad_idx):
        self.start = 0
        self.dataset = {}
        with open(data_file, "r") as f:
            lines = f.readlines()[:100] # TODO: remove the :100 part
            lines = [line.strip("\n").split(" ") for line in lines]
            lines = [np.array([int(x) for x in line]) for line in lines]
            targets = [np.append(line[1:], pad_idx) for line in lines]

            self.dataset["input"] = lines
            self.dataset["target"] = targets

        self.size = len(self.dataset["input"])

    def next_train_batch(self, batch_size):
        batch_inputs = []
        batch_targets = []
        end = self.start + batch_size
        batch_inputs = self.dataset["input"][self.start:end]
        batch_targets = self.dataset["target"][self.start:end]

        self.start = end

        if (len(batch_inputs) < batch_size):
            rest = batch_size - len(batch_inputs)
            batch_inputs += self.dataset["input"][:rest]
            batch_targets += self.dataset["target"][:rest]
            self.start = rest

        return batch_inputs, batch_targets


if __name__ == "__main__":
    voc = Vocabulary()
    voc.load_from_file("data/sentences.train")

    pad_idx = voc.voc["<pad>"].idx
    data_source = DataSource("data/encoded.train", pad_idx)

    model = LSTM_LM(voc, data_source, is_training=True)
    print(model.embeddings.get_shape())
    idx = model.vocab.voc["something"].idx

    model.create_model()
    #model.create_model("wordembeddings-dim100.word2vec")
    model.train()
