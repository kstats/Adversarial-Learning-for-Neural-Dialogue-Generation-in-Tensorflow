# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import pickle
import random
import cPickle as pkl


# Special vocabulary symbols - we always put them at the start.
_PAD         = b"_PAD"
_GO          = b"_GO"
_EOS         = b"_EOS"
_UNK         = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID  = 0
UNK_ID  = 1
GO_ID   = 25001
EOS_ID  = 25002


# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE   = re.compile(br"\d")

# URLs for WMT data.
#_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
#_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"


#def maybe_download(directory, filename, url):
#  """Download filename from url unless it's already in directory."""
#  if not os.path.exists(directory):
#    print("Creating directory %s" % directory)
#    os.mkdir(directory)
#  filepath = os.path.join(directory, filename)
#  if not os.path.exists(filepath):
#    print("Downloading %s to %s" % (url, filepath))
#    filepath, _ = urllib.request.urlretrieve(url, filepath)
#    statinfo = os.stat(filepath)
#    print("Succesfully downloaded", filename, statinfo.st_size, "bytes")
#  return filepath


#def gunzip_file(gz_path, new_path):
#  """Unzips from gz_path into new_path."""
#  print("Unpacking %s to %s" % (gz_path, new_path))
#  with gzip.open(gz_path, "rb") as gz_file:
#    with open(new_path, "wb") as new_file:
#      for line in gz_file:
#        new_file.write(line)


#def get_wmt_enfr_train_set(directory):
#  """Download the WMT en-fr training corpus to directory unless it's there."""
#  train_path = os.path.join(directory, "giga-fren.release2.fixed")
#  if not (gfile.Exists(train_path +".fr") and gfile.Exists(train_path +".en")):
#    corpus_file = maybe_download(directory, "training-giga-fren.tar",
#                                 _WMT_ENFR_TRAIN_URL)
#    print("Extracting tar file %s" % corpus_file)
#    with tarfile.open(corpus_file, "r") as corpus_tar:
#      corpus_tar.extractall(directory)
#    gunzip_file(train_path + ".fr.gz", train_path + ".fr")
#    gunzip_file(train_path + ".en.gz", train_path + ".en")
#  return train_path


#def get_wmt_enfr_dev_set(directory):
#  """Download the WMT en-fr training corpus to directory unless it's there."""
#  dev_name = "newstest2013"
#  dev_path = os.path.join(directory, dev_name)
#  if not (gfile.Exists(dev_path + ".fr") and gfile.Exists(dev_path + ".en")):
#    dev_file = maybe_download(directory, "dev-v2.tgz", _WMT_ENFR_DEV_URL)
#    print("Extracting tgz file %s" % dev_file)
#    with tarfile.open(dev_file, "r:gz") as dev_tar:
#      fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
#      en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
#      fr_dev_file.name = dev_name + ".fr"  # Extract without "dev/" prefix.
#      en_dev_file.name = dev_name + ".en"
#      dev_tar.extract(fr_dev_file, directory)
#      dev_tar.extract(en_dev_file, directory)
#  return dev_path


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path_list, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path_list))
    vocab = {}
    for data_path in data_path_list:
        with gfile.GFile(data_path, mode="rb") as f:
          counter = 0
          for line in f:
            counter += 1
            if counter % 100000 == 0:
              print("  processing line %d" % counter)
            line = tf.compat.as_bytes(line)
            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
            for w in tokens:
              word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
              if word in vocab:
                vocab[word] += 1
              else:
                vocab[word] = 1

    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
      vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
      for w in vocab_list:
        vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    #vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocabulary, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


#def prepare_chitchat_data(data_dir, vocabulary, vocabulary_size, tokenizer=None):
#  """Get WMT data into data_dir, create vocabularies and tokenize data.
#
#  Args:
#    data_dir: directory in which the data sets will be stored.
#    en_vocabulary_size: size of the English vocabulary to create and use.
#    fr_vocabulary_size: size of the French vocabulary to create and use.
#    tokenizer: a function to use to tokenize each data sentence;
#      if None, basic_tokenizer will be used.
#
#  Returns:
#    A tuple of 6 elements:
#      (1) path to the token-ids for English training data-set,
#      (2) path to the token-ids for French training data-set,
#      (3) path to the token-ids for English development data-set,
#      (4) path to the token-ids for French development data-set,
#      (5) path to the English vocabulary file,
#      (6) path to the French vocabulary file.
#  """
#  train_path = os.path.join(data_dir, "training200k.txt")
#  dev_path = os.path.join(data_dir, "dev100k.txt")
#
#  # Create token ids for the training data.
#  answer_train_ids_path = train_path + (".ids%d.answer" % vocabulary_size)
#  query_train_ids_path = train_path + (".ids%d.query" % vocabulary_size)
#  data_to_token_ids(train_path + ".answer.decoded", answer_train_ids_path, vocabulary, tokenizer)
#  data_to_token_ids(train_path + ".query.decoded", query_train_ids_path, vocabulary, tokenizer)
#
#  # Create token ids for the development data.
#  answer_dev_ids_path = dev_path + (".ids%d.answer" % vocabulary_size)
#  query_dev_ids_path = dev_path + (".ids%d.query" % vocabulary_size)
#  data_to_token_ids(dev_path + ".answer.decoded", answer_dev_ids_path, vocabulary, tokenizer)
#  data_to_token_ids(dev_path + ".query.decoded", query_dev_ids_path, vocabulary, tokenizer)
#
#  return (query_train_ids_path, answer_train_ids_path, query_dev_ids_path, answer_dev_ids_path)

#def prepare_defined_data(data_path, vocabulary, vocabulary_size, tokenizer=None):
#  #vocab_path = os.path.join(data_dir, "vocab%d.all" %vocabulary_size)
#  #query_vocab_path = os.path.join(data_dir, "vocab%d.query" %query_vocabulary_size)
#
#  answer_fixed_ids_path = data_path + (".ids%d.answer" % vocabulary_size)
#  query_fixed_ids_path = data_path + (".ids%d.query" % vocabulary_size)
#
#  data_to_token_ids(data_path + ".answer", answer_fixed_ids_path, vocabulary, tokenizer)
#  data_to_token_ids(data_path + ".query", query_fixed_ids_path, vocabulary, tokenizer)
#  return (query_fixed_ids_path, answer_fixed_ids_path)

#def get_dummy_set(dummy_path, vocabulary, vocabulary_size, tokenizer=None):
#    dummy_ids_path = dummy_path + (".ids%d" % vocabulary_size)
#    data_to_token_ids(dummy_path, dummy_ids_path, vocabulary, tokenizer)
#    dummy_set = []
#    with gfile.GFile(dummy_ids_path, "r") as dummy_file:
#        line = dummy_file.readline()
#        counter = 0
#        while line:
#            counter += 1
#            dummy_set.append([int(x) for x in line.split()])
#            line = dummy_file.readline()
#    return dummy_set

#def split_into_files(data_path, fname):
#  with open(data_path+fname,'r') as f:
#    lines = np.asarray([x.strip().split('|') for x in f.readlines()])
#    with open(data_path+fname+'.query','w+') as f1:
#      f1.write('\n'.join(lines[:,0]))
#    with open(data_path+fname+'.answer','w+') as f2:
#      f2.write('\n'.join(lines[:,1]))

def translate(data_path):
  vocab, reverse = initialize_vocabulary("./data/movie_25000")
  sentences = []
  with open(data_path, 'r') as f:
    for line in f.readlines():
      #print(line)
      sent = ""
      for token in line.split(" "):
        sent += reverse[int(token) - 1] + " "
      sentences.append(sent)
      #print(sent)
  return sentences

def decode_sentence(sent, vocab, reverse):
    #TODO
   return ' '.join(map(lambda x: reverse[int(x)-1],sent))
   #return ' '.join(map(lambda x: reverse[int(x)], sent))

def decode_file(fname):   
   v, r = initialize_vocabulary("./data/movie_25000")
   with open(fname,'r') as f:
       lines = [map(int,x.strip().split(' ')) for x in f.readlines()]
   with open(fname+'.decoded','w+') as f:
       f.writelines([decode_sentence(x,v,r)+'\n' for x in lines])


#def create_disc_pretrain_data(fname, vocabulary_size):
#    
#    def fake_sentence(vocabulary_size):
#        rc = []
#        for i in range(12):
#            rc.append(np.random.randint(vocabulary_size))
#            if np.random.rand() < 0.1 and i>1:
#                break
#        return rc[:12-np.random.randint(6)]
#
#  with open(fname,'r') as f:
#    lines = [map(int,x.strip().split(' ')) for x in f.readlines()]
#  n = len(lines)
#  l = int(0.9*n)
#  tset = lines[:l] + [fake_sentence(vocabulary_size) for _ in range(l)]
#  vset = lines[l:] + [fake_sentence(vocabulary_size) for _ in range(n-l)]
#  tlabels = [1]*l + [0]*l
#  vlabels = [1]*(n-l) + [0]*(n-l)
#  with open(fname+'.pkl','w+') as f:
#    pickle.dump((tset,tlabels),f)
#    pickle.dump((vset,vlabels),f)
#  return tset, vset



''' ====================================================================
    ====================================================================
        Dataset functions: 
    ====================================================================
    ===================================================================='''
def create_dataset(fname, is_disc=True):

    with open(fname,'r') as f:
        dialogs = [x.strip().split('|') for x in f.readlines()]
        n_sent, n_context = np.shape(dialogs)
    
    dataset = {}
    dataset['context']      = np.array([map(int,x[0].split(" ")) for x in dialogs])
    dataset['response']     = np.array([map(int,x[1].split(" ")) for x in dialogs])
    if is_disc:
        dataset['label']    = np.array([1] * n_sent)
    dataset['len']          = n_sent
    dataset['is_disc']      = is_disc

    return dataset


def shuffle_dataset(dataset):

    shuffled_dataset = dataset.copy()

    n_samples = dataset['len']
    
    sidx = np.random.permutation(n_samples)

    shuffled_dataset['context']  = [dataset['context'][s] for s in sidx]
    shuffled_dataset['response'] = [dataset['response'][s] for s in sidx]
    if dataset['is_disc']:
        shuffled_dataset['label'] = [dataset['label'][s] for s in sidx]

    return shuffled_dataset


def split_dataset(dataset, ratio = 0.9):
    
    n_set = int (ratio * dataset['len'] )

    set1, set2 = {}, {}

    set1['context'],  set2['context']   = dataset['context'][:n_set],  dataset['context'][n_set:]
    set1['response'], set2['response']  = dataset['response'][:n_set], dataset['response'][n_set:]
    if dataset['is_disc']:
        set1['label'], set2['label'] = dataset['label'][:n_set], dataset['label'][n_set:]
    set1['is_disc'],  set2['is_disc'] = dataset['is_disc'], dataset['is_disc']
    set1['len'], set2['len'] = n_set, dataset['len'] - n_set

    return set1, set2

def gen_false_dataset(dataset, n_sent):
    
        context, response    = [], []

        rand1,rand2  = 0, 0
        while rand1 == rand2:
            rand1 = random.randint(0,n_sent-1)
            rand2 = random.randint(0,n_sent-1)

        context, response   = dataset['context'][rand1], dataset['response'][rand2]
        return context, response


def gen_dataset_w_false_ex(dataset):
    
    mixed_dataset = dataset.copy()
    n_sent = dataset['len']

    fcontext, fresponse    = [], []
    for _ in range(n_sent):
        fc, fr =  gen_false_dataset(dataset, n_sent) 
        fcontext.append(fc)
        fresponse.append(fr)
     
    mixed_dataset['context']  = np.append(mixed_dataset['context'], fcontext)
    mixed_dataset['response'] = np.append(mixed_dataset['response'], fresponse)
    if dataset['is_disc']:
        mixed_dataset['label']    = np.append(mixed_dataset['label'], [0]*n_sent)

    mixed_dataset['len'] = dataset['len'] + n_sent
      
    return mixed_dataset



''' ====================================================================
   ====================================================================
        Disc functions: 
    ====================================================================
    ===================================================================='''


def _padding(data, max_len):
    # Get lengths of each row of data
    lens        = np.array([len(i) for i in data])

    # Mask of valid places in each row 
    valid       = np.arange(np.append(lens,max_len).max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out         = np.empty(valid.shape, dtype = np.int32)
    out.fill(PAD_ID)
    out[valid]  = np.concatenate(data)
    out         = np.delete(out, np.s_[max_len:], axis=1)

    return out


def _gen_mask(data, max_len):
       # Get lengths of each row of data
    lens        = np.array([len(i) for i in data])

    # Mask of valid places in each row 
    valid       = np.arange(np.append(lens,max_len).max()) < lens[:,None]

    mask        = np.zeros(valid.shape, dtype = np.int32)
    mask[valid] = 1
    mask        = np.delete(mask, np.s_[max_len:], axis=1)

    return mask


def dataset_padding(dataset, max_len):

    dataset['c_mask']   = np.transpose(_gen_mask(dataset['context'], max_len))
    dataset['context']  = _padding(dataset['context'], max_len)
    
    dataset['r_mask']   = np.transpose(_gen_mask(dataset['response'], max_len))
    dataset['response'] = _padding(dataset['response'], max_len)

    return dataset

def convert_to_format(dataset):
    
    return ( np.stack([dataset['context'], dataset['response']], axis = 1), # batch * 2 * maxlen
                dataset['label'], 
                np.stack([dataset['c_mask'], dataset['r_mask']], axis = 2))   # mask is maxlen * batch * 2 [2 is ctx,response]



def disc_load_data(max_len, fname, valid_portion=0.1):
 
    dataset                     = create_dataset(fname)
    mixed_dataset               = gen_dataset_w_false_ex(dataset)
    #shuffle and generate train and valid dataset
    shuffled_mixed_dataset      = shuffle_dataset(mixed_dataset)
    
    train_dataset, test_set     = split_dataset(shuffled_mixed_dataset)
    train_set, valid_set        = split_dataset(train_dataset, 1 - valid_portion)


    train_set = convert_to_format(dataset_padding(train_set, max_len))

    valid_set = convert_to_format(dataset_padding(valid_set, max_len))

    test_set = convert_to_format(dataset_padding(test_set, max_len))
     
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
        yield (return_x,return_y,return_mask_x)


def src_to_disc(src_encoder, src_decoder, src_labels, max_len):

    dataset = {}
    dataset['context']  = src_encoder
    dataset['response'] = src_decoder
    dataset['label']    = src_labels

    train_inputs, train_labels, train_masks = convert_to_format(dataset_padding(dataset, max_len))

    return train_inputs, train_labels, train_masks    



''' ====================================================================
   ====================================================================
        Gen functions: 
    ====================================================================
    ===================================================================='''


def split_into_buckets(dataset, _buckets, max_size=None):
    
    data_set = [[] for _ in _buckets]
    for i in range(dataset['len']):
        if (max_size and i > max_size):
            break
        source_ids = dataset['context'][i]
        target_ids = dataset['response'][i]

        target_ids.append(EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets): 
            if len(source_ids) < source_size and len(target_ids) < target_size:
                data_set[bucket_id].append([source_ids, target_ids])
                break
    return data_set


def src_to_gen(source_encoder, source_decoder, _buckets, bucket_id, batch_size):
    
    encoder_size, decoder_size        = _buckets[bucket_id]
    encoder_inputs, decoder_inputs    = [], []
           
    for encoder_input in source_encoder:
    # Encoder inputs are padded and then reversed.
        encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

    for decoder_input in source_decoder:
        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = decoder_size - len(decoder_input) - 1 
        decoder_inputs.append([GO_ID] + decoder_input +  [PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                for batch_idx in xrange(batch_size)], dtype=np.int32))
    # at this point, batch_encoder_inputs are transposed, enc_size*batch_size

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                for batch_idx in xrange(batch_size)], dtype=np.int32))

    # Each line is a word across the entire batch. Some of them are just pads, some are real

    # Create target_weights to be 0 for targets that are padding.
        batch_weight = np.ones(batch_size, dtype=np.float32)
        for batch_idx in xrange(batch_size):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights


def transform_responses(responses, _buckets, bucket_id):
 #   import pdb; pdb.set_trace()
    decoder_inputs  = []
    dec_size        = _buckets[bucket_id][1]
    
    for res in responses:
        dec_gen = [GO_ID] + res[:dec_size]
        if len(dec_gen) < dec_size:
            dec_gen = dec_gen + [PAD_ID] * (dec_size - len(dec_gen))
        dec_gen = np.reshape(dec_gen, (-1, 1))
        decoder_inputs.append(dec_gen)
    decoder_inputs = np.transpose(np.asarray(decoder_inputs))
    decoder_inputs = np.squeeze(decoder_inputs)

    return decoder_inputs

def prepare_data(gen_config):
    
    train_path                  = os.path.join(gen_config.data_dir, gen_config.train_data_file)
    vocab, rev_vocab            = initialize_vocabulary(gen_config.vocab_path)
    dataset                     = create_dataset(train_path, is_disc = False)
    train_dataset, dev_dataset  = split_dataset(dataset, ratio = gen_config.train_ratio )

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)." % gen_config.max_train_data_size)
    train_set, dev_set = read_data(train_dataset, gen_config.buckets, gen_config.max_train_data_size), read_data(dev_dataset, gen_config.buckets)

    return vocab, rev_vocab, dev_set, train_set

def read_data(dataset, _buckets, max_size=None):
    
    data_set = [[] for _ in _buckets]
    for i in range(dataset['len']):
        if (max_size and i > max_size):
            break
        source_ids = dataset['context'][i]
        target_ids = dataset['response'][i]

        target_ids.append(EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets): 
            if len(source_ids) < source_size and len(target_ids) < target_size:
                data_set[bucket_id].append([source_ids, target_ids])
                break
    return data_set
