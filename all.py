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

    experiment_name = sys.argv[1]

    is_training = True
    if len(sys.argv) >= 3 and sys.argv[2] == 'load':
        is_training = False

    if experiment_name == 'c':
        model = LSTM_C(voc, is_training=is_training, exp_name=experiment_name)
    else:
        model = LSTM_LM(voc, is_training=is_training, exp_name=experiment_name)

    model.create_model()

    with tf.Session() as sess:
        ### Training

        if is_training:
            if experiment_name == 'a':
                model.train(sess, data_source_train)
            else:
                model.train(sess, data_source_train,pretrained_embeddings_path='data/wordembeddings-dim100.word2vec')
            print('Training over. Evaluating perplexity.')
        else:
            model.load_model(sess)
            print('Loaded the model')


        
        
        perplexities = model.eval(sess, data_source_test, MAX_NUM_SENTENCES = 100)

        print('Number of perplexities: ', len(perplexities))

        perp_file_name = 'outputs/group09.perplexity' + experiment_name.upper()
        with open(perp_file_name, "w") as fout:
            for perp in perplexities:
                fout.write(str(perp) + '\n')

            print('Wrote the perplexities in ', perp_file_name)

        ### Generation
        if experiment_name == 'c':
            CONTINUATION_FILE = "data/sentences.continuation"
            OUTPUT_GEN = "data/continuation.output"
            with open(CONTINUATION_FILE, "r") as fin, \
                open(OUTPUT_GEN, "w") as fout:
                lines = [line.strip("\n") for line in fin.readlines()][:100]
                #model.load_model(sess)
                print('Read all the lines.')
                for line in lines:
                    pred = model.generate(sess, line)
                    fout.write(pred + "\n")

