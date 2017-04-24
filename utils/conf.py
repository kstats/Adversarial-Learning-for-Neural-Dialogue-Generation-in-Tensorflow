__author__ = 'MORAN01'
import os
class disc_config(object):
    batch_size = 5
    lr = 0.1
    lr_decay = 0.6
    vocabulary_size = 25000
    embed_dim = 128
    hidden_neural_size = 200
    hidden_layer_num = 3
    train_dir = 'data/training30k.txt.query.pkl'
    # train_dir = 'data/subj0.pkl'
    max_len = 40
    valid_num = 100
    checkpoint_num = 1000
    init_scale = 0.1
    class_num = 2
    keep_prob = 0.5
    num_epoch = 60
    max_decay_epoch = 30
    max_grad_norm = 5
    out_dir = os.path.abspath(os.path.join(os.path.curdir,"runs"))
    checkpoint_every = 10

class gen_config(object):
    beam_size = 5
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 1
    size = 256
    num_layers = 2
    vocab_size = 25000
    data_dir = "data/"
    train_dir = "data/"
    max_train_data_size = 0
    steps_per_checkpoint = 200
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (50, 50)]


