#!/bin/bash

DATA_DIR="data"

function fail {
  echo >&2 $1
  exit 1
}

# Checking for data directory
if ! [[ -d "${DATA_DIR}" ]]; then
  fail "Data directory not present, download and extract data"
fi

# Making vocabulary.train
if ! [[ -f  "${DATA_DIR}/vocabulary.train" ]]; then
  echo "Creating vocabulary.train"
  python3 create_vocabulary.py $DATA_DIR/sentences.train $DATA_DIR/vocabulary.train || fail "Unable to create vocabulary"
  echo "Created vocabulary.train"
else
  echo "Vocabulary found"
fi

# Encoding dataset
if ! [[ -f  "${DATA_DIR}/encoded.train" ]]; then
  echo "Creating encoded.train"
  python3 encode_dataset.py data/vocabulary.train data/sentences.train data/encoded.train || fail "Unable to create encoded dataset"
  echo "Created encoded.train"
else
  echo "Encoded found"
fi

if ! [[ -f  "${DATA_DIR}/encoded.test" ]]; then
  echo "Creating encoded.test"
  python3 encode_dataset.py data/vocabulary.train data/sentences.test data/encoded.test || fail "Unable to create encoded dataset"
  echo "Created encoded.test"
else
  echo "Encoded found"
fi

# Train the model
STARTTIME=$(date +%s)
if [[ $# == 0 ]]; then
  echo "Running 1.1 Experiment A"
  # source activate tensorflow      # tensorflow conda environment
  export CUDA_VISIBLE_DEVICES=""
  python3 train.py a || fail "Unable to run lstm"
elif [[ $# == 1 ]]; then
  if [[ $1 == "w2vec" || $1 == "word2vec" ]]; then
    echo "Running 1.1 Experiment B"
    if ! [[ -f  "${DATA_DIR}/wordembeddings-dim100.word2vec" ]]; then
      echo "Word2vec embedding file not present, aborting" 
      exit 2
    fi
    python3 train.py b $DATA_DIR/wordembeddings-dim100.word2vec || fail "Unable to run lstm"
  else
    # probably an overkill
    echo "Running with embedding file " $1
    python3 train.py b $1 || fail "Unable to run lstm"
  fi
elif [[ $# == 2 && $2 == "c" ]]; then
  echo "Running 1.1. Experiment C"
  python3 train.py c $DATA_DIR/wordembeddings-dim100.word2vec || fail "Unable to run lstm"
else
  # For other experiments
  echo "Much arguments, such wow"
fi
ENDTIME=$(date +%s)
echo "It takes $((($ENDTIME - $STARTTIME) / 60)) minutes to train..."

# Test the model

# Generate sentences
