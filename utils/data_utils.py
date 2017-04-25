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
# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"


def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(directory):
    print("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    print("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Succesfully downloaded", filename, statinfo.st_size, "bytes")
  return filepath


def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "wb") as new_file:
      for line in gz_file:
        new_file.write(line)


def get_wmt_enfr_train_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  train_path = os.path.join(directory, "giga-fren.release2.fixed")
  if not (gfile.Exists(train_path +".fr") and gfile.Exists(train_path +".en")):
    corpus_file = maybe_download(directory, "training-giga-fren.tar",
                                 _WMT_ENFR_TRAIN_URL)
    print("Extracting tar file %s" % corpus_file)
    with tarfile.open(corpus_file, "r") as corpus_tar:
      corpus_tar.extractall(directory)
    gunzip_file(train_path + ".fr.gz", train_path + ".fr")
    gunzip_file(train_path + ".en.gz", train_path + ".en")
  return train_path


def get_wmt_enfr_dev_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  dev_name = "newstest2013"
  dev_path = os.path.join(directory, dev_name)
  if not (gfile.Exists(dev_path + ".fr") and gfile.Exists(dev_path + ".en")):
    dev_file = maybe_download(directory, "dev-v2.tgz", _WMT_ENFR_DEV_URL)
    print("Extracting tgz file %s" % dev_file)
    with tarfile.open(dev_file, "r:gz") as dev_tar:
      fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
      en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
      fr_dev_file.name = dev_name + ".fr"  # Extract without "dev/" prefix.
      en_dev_file.name = dev_name + ".en"
      dev_tar.extract(fr_dev_file, directory)
      dev_tar.extract(en_dev_file, directory)
  return dev_path


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


def prepare_chitchat_data(data_dir, vocabulary, vocabulary_size, tokenizer=None):
  """Get WMT data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """
  # Get wmt data to the specified directory.
  #train_path = get_wmt_enfr_train_set(data_dir)
  train_path = os.path.join(data_dir, "chitchat.train")
  #dev_path = get_wmt_enfr_dev_set(data_dir)
  dev_path = os.path.join(data_dir, "chitchat.dev")
  # fixed_path = os.path.join(data_dir, "chitchat.fixed")
  # weibo_path = os.path.join(data_dir, "chitchat.weibo")
  # qa_path = os.path.join(data_dir, "chitchat.qa")

  # voc_file_path = [train_path+".answer", fixed_path+".answer", weibo_path+".answer", qa_path+".answer",
  #                    train_path+".query", fixed_path+".query", weibo_path+".query", qa_path+".query"]
  #voc_query_path = [train_path+".query", fixed_path+".query", weibo_path+".query", qa_path+".query"]
  # Create vocabularies of the appropriate sizes.
  #vocab_path = os.path.join(data_dir, "vocab%d.all" % vocabulary_size)
  #query_vocab_path = os.path.join(data_dir, "vocab%d.query" % en_vocabulary_size)

  #create_vocabulary(vocab_path, voc_file_path, vocabulary_size)


  #create_vocabulary(query_vocab_path, voc_query_path, en_vocabulary_size)

  # Create token ids for the training data.
  answer_train_ids_path = train_path + (".ids%d.answer" % vocabulary_size)
  query_train_ids_path = train_path + (".ids%d.query" % vocabulary_size)
  data_to_token_ids(train_path + ".answer", answer_train_ids_path, vocabulary, tokenizer)
  data_to_token_ids(train_path + ".query", query_train_ids_path, vocabulary, tokenizer)

  # Create token ids for the development data.
  answer_dev_ids_path = dev_path + (".ids%d.answer" % vocabulary_size)
  query_dev_ids_path = dev_path + (".ids%d.query" % vocabulary_size)
  data_to_token_ids(dev_path + ".answer", answer_dev_ids_path, vocabulary, tokenizer)
  data_to_token_ids(dev_path + ".query", query_dev_ids_path, vocabulary, tokenizer)

  return (query_train_ids_path, answer_train_ids_path,
          query_dev_ids_path, answer_dev_ids_path)

def prepare_defined_data(data_path, vocabulary, vocabulary_size, tokenizer=None):
  #vocab_path = os.path.join(data_dir, "vocab%d.all" %vocabulary_size)
  #query_vocab_path = os.path.join(data_dir, "vocab%d.query" %query_vocabulary_size)

  answer_fixed_ids_path = data_path + (".ids%d.answer" % vocabulary_size)
  query_fixed_ids_path = data_path + (".ids%d.query" % vocabulary_size)

  data_to_token_ids(data_path + ".answer", answer_fixed_ids_path, vocabulary, tokenizer)
  data_to_token_ids(data_path + ".query", query_fixed_ids_path, vocabulary, tokenizer)
  return (query_fixed_ids_path, answer_fixed_ids_path)

def get_dummy_set(dummy_path, vocabulary, vocabulary_size, tokenizer=None):
    dummy_ids_path = dummy_path + (".ids%d" % vocabulary_size)
    data_to_token_ids(dummy_path, dummy_ids_path, vocabulary, tokenizer)
    dummy_set = []
    with gfile.GFile(dummy_ids_path, "r") as dummy_file:
        line = dummy_file.readline()
        counter = 0
        while line:
            counter += 1
            dummy_set.append([int(x) for x in line.split()])
            line = dummy_file.readline()
    return dummy_set

def split_into_files(data_path, fname):
  with open(data_path+fname,'r') as f:
    lines = np.asarray([x.strip().split('|') for x in f.readlines()])
    with open(data_path+fname+'.query','w+') as f1:
      f1.write('\n'.join(lines[:,0]))
    with open(data_path+fname+'.answer','w+') as f2:
      f2.write('\n'.join(lines[:,1]))

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
      #import pdb; pdb.set_trace()
  return sentences

def fake_sentence(vocabulary_size):
  rc = []
  for i in range(12):
    rc.append(np.random.randint(vocabulary_size))
    if np.random.rand() < 0.1 and i>1:
      break
  return rc[:12-np.random.randint(6)]

def fake_sentence_and_context(vocabulary_size, data):
  rc = [[],[]]
  rand1 = 0
  rand2 = 0
  while rand1 == rand2:
    rand1 = random.randint(0,vocabulary_size)
    rand2 = random.randint(0,vocabulary_size)
  rc[0] = data[0][rand1]
  rc[1] = data[1][rand2]
  return rc

def decode_sentence(sent,vocab, reverse):
  return ' '.join(map(lambda x: reverse[int(x)-1],sent))

def create_disc_pretrain_data(fname, vocabulary_size):
  with open(fname,'r') as f:
    lines = [map(int,x.strip().split(' ')) for x in f.readlines()]
  n = len(lines)
  l = int(0.9*n)
  tset = lines[:l] + [fake_sentence(vocabulary_size) for _ in range(l)]
  vset = lines[l:] + [fake_sentence(vocabulary_size) for _ in range(n-l)]
  tlabels = [1]*l + [0]*l
  vlabels = [1]*(n-l) + [0]*(n-l)
  with open(fname+'.pkl','w+') as f:
    pickle.dump((tset,tlabels),f)
    pickle.dump((vset,vlabels),f)
  return tset, vset


def create_disc_context_data(fname, vocabulary_size):
  with open(fname,'r') as f:
    parts = [x.strip().split('|') for x in f.readlines()]
    context = [map(int,x[0].split(" ")) for x in parts]
    response = [map(int,x[1].split(" ")) for x in parts]
    lines = [context, response]
  print(np.shape(lines))
  n = len(lines[0])
  l = int(0.9*n)
  tset = [lines[0][:l], lines[1][:l], [0] * l] #+ [[fake_sentence_and_context(vocabulary_size) for _ in range(l)]]
  fake = [fake_sentence_and_context(vocabulary_size, tset) for _ in range(l)]
  #import pdb; pdb.set_trace()
  for f in range(np.shape(fake)[0]):
    tset[0].append(fake[f][0])
    tset[1].append(fake[f][1])
  #tset[1] = tset[1] + fake[:][1]
  vset = [lines[0][l:], lines[1][l:], [0] * (n-l)]# + [[fake_sentence_and_context(vocabulary_size) for _ in range(n-l)]]
  fake = [fake_sentence_and_context(vocabulary_size, tset) for _ in range(n-l)]
  for f in range(np.shape(fake)[0]):
    vset[0].append(fake[f][0])
    vset[1].append(fake[f][1])
  print (np.shape(tset))
  print (np.shape(vset))
  tlabels = [1]*l + [0]*l
  vlabels = [1]*(n-l) + [0]*(n-l)
  tset[2] = tlabels
  vset[2] = vlabels
  #import pdb; pdb.set_trace()
  print (vlabels)
  with open(fname+'.pkl','w+') as f:
    pickle.dump((tset,tlabels),f)
    pickle.dump((vset,vlabels),f)
  return tset, vset








