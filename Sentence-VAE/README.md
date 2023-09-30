# Sentence Variational Autoencoder

PyTorch re-implementation of [_Generating Sentences from a Continuous Space_](https://arxiv.org/abs/1511.06349) by Bowman et al. 2015.
Note: This implementation does not support LSTM's at the moment, but RNN's and GRU's._
## Results
### Performance
Training was stopped after 4 epochs. The true ELBO was optimized for approximately 1 epoch (as can bee see in the graph above). Results are averaged over entire split.

| Split       | NLL   | KL    |
|:------------|:------:|:-----:|
| Train       | 99.821 | 7.944 |
| Validation  | 103.220 | 7.346 |
| Test        | 103.967 | 7.269 |

## Setting up
```
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xf simple-examples.tgz
mkdir data
mv simple-examples/data/ptb.train.txt data
mv simple-examples/data/ptb.test.txt data
mv simple-examples/data/ptb.valid.txt data
rm -rf simple-examples*
mkdir models
pip install -r Sentence-VAE/requirements.txt
```

## Training
To run the training, please download the Penn Tree Bank data first (download from [Tomas Mikolov's webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)). The code expects to find at least `ptb.train.txt` and `ptb.valid.txt` in the specified data directory. The data can also be donwloaded with the `dowloaddata.sh` script.

Then training can be executed with the following command:
```
python3 train.py
```

The following arguments are available:

`--data_dir`  The path to the directory where PTB data is stored, and auxiliary data files will be stored.  
`--create_data` If provided, new auxiliary data files will be created form the source data.  
`--max_sequence_length` Specifies the cut off of long sentences.  
`--min_occ` If a word occurs less than "min_occ" times in the corpus, it will be replaced by the <unk> token.  
`--test` If provided, performance will also be measured on the test set.

`-ep`, `--epochs`  
`-bs`, `--batch_size`  
`-lr`, `--learning_rate`

`-eb`, `--embedding_size`  
`-rnn`, `--rnn_type` Either 'rnn' or 'gru'.  
`-hs`, `--hidden_size`  
`-nl`, `--num_layers`  
`-bi`, `--bidirectional`  
`-ls`, `--latent_size`  
`-wd`, `--word_dropout` Word dropout applied to the input of the Decoder, which means words will be replaced by `<unk>` with a probability of `word_dropout`.  
`-ed`, `--embedding_dropout` Word embedding dropout applied to the input of the Decoder.

`-af`, `--anneal_function` Either 'logistic' or 'linear'.  
`-k`, `--k` Steepness of the logistic annealing function.  
`-x0`, `--x0` For 'logistic', this is the mid-point (i.e. when the weight is 0.5); for 'linear' this is the denominator.

`-v`, `--print_every`  
`-tb`, `--tensorboard_logging` If provided, training progress is monitored with tensorboard.  
`-log`, `--logdir` Directory of log files for tensorboard.  
`-bin`,`--save_model_path` Directory where to store model checkpoints.

## Inference
For obtaining samples and interpolating between senteces, inference.py can be used.
```
python3 inference.py -c $CHECKPOINT -n $NUM_SAMPLES
```