"""
description: this file helps to load raw file and gennerate batch x,y
author:luchi
date:22/11/2016
"""
import numpy as np
import cPickle as pkl
import data_utils as du

#file path
# dataset_path='data/training30k.txt.query.pkl'
dataset_path='data/subj0.pkl'

def set_dataset_path(path):
    dataset_path=path


def load_data(max_len, n_words=25000, valid_portion=0.1, sort_by_len = True, debug=False):
 
    #TODO change this to be dynamic
    dataset                     = du.create_dataset("data/training200k.txt") 
    tarin_dataset, test_set     = du.split_dataset(dataset)
    
    #shuffle and generate train and valid dataset
    shuffled_tarin_dataset      = du.shuffle_dataset(tarin_dataset)
    train_set, valid_set        = du.split_dataset(shuffled_tarin_dataset, 1 - valid_portion)


    train_set = du.convert_to_format(du.dataset_padding(train_set, max_len))

    valid_set = du.convert_to_format(du.dataset_padding(valid_set, max_len))

    test_set = du.convert_to_format(du.dataset_padding(test_set, max_len))


   # import pdb; pdb.set_trace()
    

    #remove unknow words
    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

        '''valid_set_x = [valid_set[0], valid_set[1]]
    valid_set_y = [valid_set[2]]
    train_set_x = [train_set[0], train_set[1]]
    train_set_y = train_set[2]'''

    '''train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)'''



    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
    #TODO rewrite this?
    '''if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]


        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]'''

  
    return train_set,valid_set,test_set


#return batch dataset
def batch_iter(data,batch_size):

    #get dataset and label
    x,y,mask_x=data
    x=np.array(x)
    y=np.array(y)
    data_size=len(x)
    num_batches_per_epoch=int((data_size-1)/batch_size)
    for batch_index in range(num_batches_per_epoch):
        start_index=batch_index*batch_size
        end_index=min((batch_index+1)*batch_size,data_size)
        return_x = x[start_index:end_index]
        return_y = y[start_index:end_index]
        return_mask_x = mask_x[:,start_index:end_index]
        # if(len(return_x)<batch_size):
        #     print(len(return_x))
        #     print return_x
        #     print return_y
        #     print return_mask_x
        #     import sys
        #     sys.exit(0)
        yield (return_x,return_y,return_mask_x)


