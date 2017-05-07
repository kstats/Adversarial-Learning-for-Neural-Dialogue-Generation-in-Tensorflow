import os
class disc_config(object):
    
    hidden_layer_num            = 1
    hidden_neural_size          = 512
    lr                          = 0.02
    lr_decay                    = 0.6
    max_grad_norm               = 5    
    keep_prob                   = 1.  
    batch_size                  = 5
    embed_dim                   = 128
    class_num                   = 2
    init_scale                  = 0.1    
    
    vocabulary_size             = 25003    

    train_dir                   = 'data/'
    train_data_file             = "training500k.txt"
    max_len                     = 50
    out_dir                     = os.path.abspath(os.path.join(os.path.curdir,"runs"))

    checkpoint_num              = 1000 
    checkpoint_every            = 10
    num_epoch                   = 60
    max_decay_epoch             = 30

    iters                       = 1
   # valid_num                   = 100

    #after how many global steps should we pickle?
    plot_every                  = 1

class gen_config(object):
    
    num_layers                  = 1
    size                        = 512
    learning_rate               = 0.1
    learning_rate_decay_factor  = 0.99
    max_gradient_norm           = 5.0
    keep_prob                   = 1.
    batch_size                  = 5
    beam_size                   = 5

    vocab_path                  = './data/movie_25000'
    vocab_size                  = 25003
    
    data_dir                    = "data/"
    train_data_file             = "training500k.txt"
    train_ratio                 = 0.9    
    max_train_data_size         = 0
    train_dir                   = "data/"

    steps_per_checkpoint        = 8000
    steps_per_sample            = 150
    
    iters                       = 1
    force_iters                 = 0
    buckets                     = [(5, 10), (10, 15), (20, 25), (40, 50), (50, 50)]
