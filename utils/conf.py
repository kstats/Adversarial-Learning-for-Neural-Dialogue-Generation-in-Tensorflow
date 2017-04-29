__author__ = 'MORAN01'
import os
class disc_config(object):
    batch_size          = 128
    lr                  = 0.1
    lr_decay            = 0.6
    vocabulary_size     = 25003
    keep_prob           = 0.6  
    hidden_neural_size  = 512
    
    embed_dim           = 128
    hidden_layer_num    = 1
    train_dir           = 'data/training30k.txt.query.pkl'
    train_data_file     = "./data/training200k.txt"
    max_len             = 40
    valid_num           = 100
    checkpoint_num      = 1000
    init_scale          = 0.1
    class_num           = 2
    num_epoch           = 60
    max_decay_epoch     = 30
    max_grad_norm       = 5
    out_dir             = os.path.abspath(os.path.join(os.path.curdir,"runs"))
    checkpoint_every    = 10

class gen_config(object):
    batch_size                  = 1 
    learning_rate               = 0.5
    learning_rate_decay_factor  = 0.99
    vocab_size                  = 25003
    keep_prob                   = 0.8
    size                        = 512
    num_layers                  = 1
    beam_size                   = 5
    max_gradient_norm           = 5.0
    vocab_path                  = './data/movie_25000'
    data_dir                    = "data/"
    train_ratio                 = 0.9
    train_dir                   = "data/"
    train_data_file             = "training500k.txt"
    # train_data_file             = "training200k.txt"
    max_train_data_size         = 0
    steps_per_checkpoint        = 20000
    buckets                     = [(5, 10), (10, 15), (20, 25), (40, 50), (50, 50)]
