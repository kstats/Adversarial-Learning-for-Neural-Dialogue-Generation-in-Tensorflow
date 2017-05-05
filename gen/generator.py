from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pickle
import heapq
import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import utils.data_utils as data_utils
import utils.conf as conf
import gen.gen_model as seq2seq_model
from tensorflow.python.platform import gfile

sys.path.append('../utils')

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = conf.gen_config.buckets


def create_model(session, gen_config):
    start_time  = time.time()        
    """Create generation model and initialize or load parameters in session."""
    model = seq2seq_model.Seq2SeqModel(
                gen_config.vocab_size, gen_config.vocab_size, _buckets,
                gen_config.size, gen_config.num_layers, gen_config.max_gradient_norm, gen_config.batch_size,
                gen_config.learning_rate, gen_config.learning_rate_decay_factor, keep_prob=gen_config.keep_prob)

    ckpt = tf.train.get_checkpoint_state(gen_config.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        if gen_config.learning_rate < model.learning_rate.eval():
          print('Re-setting learning rate to %f' % gen_config.learning_rate)
          session.run(model.learning_rate.assign(gen_config.learning_rate),[])
    else:
        print("Created Gen_RNN model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    
    end_time    = time.time()
    print("Time to create Gen_RNN model: %.2f" % (end_time - start_time))

    return model


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def train(gen_config):
    vocab, rev_vocab, dev_set, train_set = data_utils.prepare_data(gen_config)

    with tf.Session() as sess:
        # Create model.
        train_bucket_sizes  = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size    = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        print("Creating %d layers of %d units." % (gen_config.num_layers, gen_config.size))
        model       = create_model(sess, gen_config)
      

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        moving_average_loss = 0.0
        current_step    = 0
        previous_losses = []

        step_loss_summary = tf.Summary()
        writer            = tf.summary.FileWriter("../logs/", sess.graph)

        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id        = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = model.get_batch(
                train_set, bucket_id, 0)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, mode=model.SM_TRAIN)
            
            #Uncomment to debug  forward_only = False, projection=True mode  
            # q, outputs_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only = False, projection=True)


            step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint
            loss += step_loss / gen_config.steps_per_checkpoint
            moving_average_loss += step_loss
            current_step += 1

            if current_step % gen_config.steps_per_sample == 0:
              
              sample_context, sample_response, sample_labels, responses = gen_sample(sess, gen_config, model, vocab,
                                               batch_source_encoder, batch_source_decoder, mc_search=False)
              print("Step %d loss is %f, learning rate is %f" % (model.global_step.eval(), moving_average_loss / gen_config.steps_per_sample, model.learning_rate.eval()))
              moving_average_loss = 0.0
              print("Sampled generator:\n")
              for input, response, label in zip(sample_context, sample_response, sample_labels):
                print(str(label) + "\t" + str(input) + "\t" + str(response))
            
            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % gen_config.steps_per_checkpoint == 0:

                bucket_value = step_loss_summary.value.add()
                bucket_value.tag = "loss"
                bucket_value.simple_value = float(loss)
                writer.add_summary(step_loss_summary, current_step)

                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.6f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                step_tracker = model.global_step.eval()

                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(gen_config.train_dir, "chitchat.model")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                sys.stdout.flush()



def get_predicted_sentence(sess, input_token_ids, vocab, model,
                            beam_size, buckets, mc_search=True,debug=False):
    
    def model_step(enc_inp, dec_inp, dptr, target_weights, bucket_id):
        logits  = model.step(sess, enc_inp, dec_inp, target_weights, bucket_id, mode=model.SM_SAMPLE)
        prob          = softmax(logits[dptr][0])
        return prob

    def greedy_dec(output_logits):
        #output_logits is [max_len X batch X vocab_size] ->
        #transpose to [batch X max_len X vocab_size]
        selected_token_ids = []
        for logits in np.transpose(output_logits, (1,0,2)):
            selected_token_ids.append([int(np.argmax(logit, axis=0)) for logit in logits])
        
        #Remove Multiple EOS 
        for b_id, s_t_id in enumerate(selected_token_ids):
            eos_id = np.where(np.asarray(s_t_id) == data_utils.EOS_ID)      
            selected_token_ids[b_id] = s_t_id if len(eos_id[0])== 0 else s_t_id[:np.min(eos_id[0])+1] 

        import pdb; pdb.set_trace()
        return selected_token_ids

    # Which bucket does it belong to?
    bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(input_token_ids)])
    outputs   = []
    feed_data = {bucket_id: [(input_token_ids, outputs)]}

    # Get a 1-element batch to feed the sentence to the model.   None,bucket_id, True
    encoder_inputs, decoder_inputs, target_weights, _, _ = model.get_batch(feed_data, bucket_id, 0)
    if debug: print("\n[get_batch]\n", encoder_inputs, decoder_inputs, target_weights)

    ### Original greedy decoding
    if beam_size == 1 or (not mc_search): 
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, mode=model.SM_EVAL)
        # import pdb; pdb.set_trace()
        return [{"dec_inp": greedy_dec(output_logits), 'prob': 1}]

    # Get output logits for the setence. # initialize beams as (log_prob, empty_string, eos)
    beams, new_beams, results = [(1, {'eos': 0, 'dec_inp': decoder_inputs, 'prob': 1, 'prob_ts': 1, 'prob_t': 1})], [], []

    for dptr in range(len(decoder_inputs)-1):
      if dptr > 0:
        target_weights[dptr] = [1.]
        beams, new_beams = new_beams[:beam_size], []
      if debug: print("=====[beams]=====", beams)
      heapq.heapify(beams)  # since we will srot and remove something to keep N elements
      for prob, cand in beams:
        if cand['eos']:
          results += [(prob, cand)]
          continue
        
        all_prob_ts = model_step(encoder_inputs, cand['dec_inp'], dptr, target_weights, bucket_id)
        all_prob_t  = [0]*len(all_prob_ts)
        all_prob    = all_prob_ts

        # suppress copy-cat (respond the same as input)
        if dptr < len(input_token_ids):
          all_prob[input_token_ids[dptr]] = all_prob[input_token_ids[dptr]] * 0.01

        # beam search
        for c in np.argsort(all_prob)[::-1][:beam_size]:
          new_cand = {
            'eos'     : (c == data_utils.EOS_ID),
            'dec_inp' : [(np.array([c]) if i == (dptr+1) else k) for i, k in enumerate(cand['dec_inp'])],
            'prob_ts' : cand['prob_ts'] * all_prob_ts[c],
            'prob_t'  : cand['prob_t'] * all_prob_t[c],
            'prob'    : cand['prob'] * all_prob[c],
          }
          new_cand = (new_cand['prob'], new_cand) # for heapq can only sort according to list[0]

          if (len(new_beams) < beam_size):
            heapq.heappush(new_beams, new_cand)
          elif (new_cand[0] > new_beams[0][0]):
            heapq.heapreplace(new_beams, new_cand)

    results += new_beams  # flush last cands

    # post-process results
    res_cands = []
    for prob, cand in sorted(results, reverse=True):
      res_cands.append(cand)
    return res_cands

def get_sampled_sentence(sess, input_token_ids, vocab, model,
                           buckets, mc_search=True, debug=False):
    def model_step(enc_inp, dec_inp, dptr, target_weights, bucket_id):
        logits = model.step(sess, enc_inp, dec_inp, target_weights, bucket_id, mode=model.SM_SAMPLE)
        #TODO fix this to not just take the first item in the batch...
        prob = softmax(logits[dptr][0])
        return prob


    # Which bucket does it belong to?
    bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(input_token_ids)])
    outputs = []

    feed_data = {bucket_id: [(input_token_ids, outputs)]}
    #decoder inputs are just "go"
    encoder_inputs, decoder_inputs, target_weights, _, _ = model.get_batch(feed_data, bucket_id, 0)
    #Hacky way to get around both batching and sampling
    #Rencoder_inputs = np.array(encoder_inputs)[:,0]
    #decoder_inputs = np.array(decoder_inputs)[:,0]
    #target_weights = np.array(target_weights)[:,0]


    for n in range(len(encoder_inputs)): encoder_inputs[n] = np.array([encoder_inputs[n][0]])
    for n in range(len(decoder_inputs)): decoder_inputs[n] = np.array([decoder_inputs[n][0]])
    for n in range(len(target_weights)): target_weights[n] = np.array([target_weights[n][0]])

    decoder_len = buckets[bucket_id][1]
    #target_weights = np.zeros(decoder_len)

    # Get output logits for the setence. # initialize beams as (log_prob, empty_string, eos)
    beams, new_beams, results = [(1,
                                  {'eos': 0, 'dec_inp': decoder_inputs, 'prob': 1, 'prob_ts': 1, 'prob_t': 1})], [], []
    #TODO fix this to work with buckets
    # for dptr in range(decoder_len - 1):
    for dptr in range(decoder_len):
        # import pdb; pdb.set_trace()
        if dptr > 0:
            if not new_beams:
              break
            target_weights[dptr] = [1.]
            beams, new_beams = new_beams[:1], []
        #if debug: print("=====[beams]=====", beams)
        #heapq.heapify(beams)  # since we will srot and remove something to keep N elements
        for prob, cand in beams:
            if cand['eos']:
                results += [(prob, cand)]
                continue

            all_prob_ts = model_step(encoder_inputs, cand['dec_inp'], dptr, target_weights, bucket_id)
            all_prob_t = [0] * len(all_prob_ts)
            all_prob = all_prob_ts


            # suppress copy-cat (respond the same as input)
            # if dptr < len(encoder_inputs):
            #     all_prob[encoder_inputs[dptr]] = all_prob[encoder_inputs[dptr]] * 0.01

            # all_prob = softmax(all_prob)
            ca = np.where(np.random.multinomial(1, all_prob))[0][0]
            #TODO Change this to sample

            new_cand = {
                'eos': (ca == data_utils.EOS_ID),
                'dec_inp': [(np.array([ca]) if i == (dptr + 1) else k) for i, k in enumerate(cand['dec_inp'])],
                'prob_ts': cand['prob_ts'] * all_prob_ts[ca],
                'prob_t': cand['prob_t'] * all_prob_t[ca],
                'prob': cand['prob'] * all_prob[ca],
            }
            new_cand = (new_cand['prob'], new_cand)
            heapq.heappush(new_beams, new_cand)
            # beam search
            '''for c in np.argsort(all_prob)[::-1][:beam_size]:
                new_cand = {
                    'eos': (c == data_utils.EOS_ID),
                    'dec_inp': [(np.array([c]) if i == (dptr + 1) else k) for i, k in enumerate(cand['dec_inp'])],
                    'prob_ts': cand['prob_ts'] * all_prob_ts[c],
                    'prob_t': cand['prob_t'] * all_prob_t[c],
                    'prob': cand['prob'] * all_prob[c],
                }
                new_cand = (new_cand['prob'], new_cand)  # for heapq can only sort according to list[0]

                if (len(new_beams) < beam_size):
                    heapq.heappush(new_beams, new_cand)
                elif (new_cand[0] > new_beams[0][0]):
                    heapq.heapreplace(new_beams, new_cand)'''

    results += beams  # flush last cands

    # post-process results
    res_cands = []
    #for prob, cand in sorted(results, reverse=True):
    # import pdb; pdb.set_trace()
    temp = beams[0][1]['dec_inp']
    temp2 = [te[0] for te in temp]
    temp2.pop(0)
    return temp2



def gen_sample(sess ,gen_config, model, vocab, source_inputs, source_outputs, mc_search=True):
    
    sample_context = []
    sample_response = []
    sample_labels =[]
    rep = []

    for source_query, source_answer in zip(source_inputs, source_outputs):
        sample_context.append(source_query)
        sample_response.append(source_answer)
        sample_labels.append(1)
        responses = get_predicted_sentence(sess, source_query, vocab, model, gen_config.beam_size, _buckets, mc_search)

        for resp in responses:
            if gen_config.beam_size == 1 or (not mc_search):
                dec_inp = [dec for dec in resp['dec_inp']]
                rep.append(dec_inp)
                dec_inp = dec_inp[:]
            else:
                dec_inp = [dec.tolist()[0] for dec in resp['dec_inp'][:]]
                rep.append(dec_inp)
                dec_inp = dec_inp[1:]
            print("  (%s) -> %s" % (resp['prob'], dec_inp))
            sample_context.append(source_query)
            sample_response.append(dec_inp)
            sample_labels.append(0)

    return sample_context, sample_response, sample_labels, rep
    pass

def gen_guided_sample(sess, context, gold_standard, gen_config, model, vocab, num_samples=1):
    #import pdb; pdb.set_trace()
    sample_context = []
    sample_response = []
    sample_labels = []
    rep = []
    for con, gold in zip(context, gold_standard):
        sample_response.append(gold)
        sample_context.append(con)
        sample_labels.append(1)
        for i in range(num_samples):
            ret = get_sampled_sentence(sess, con, vocab, model, gen_config.buckets)
            sample_response.append([ret])
            sample_context.append(con)
            sample_labels.append(0)
            print ("Sampled response (of length %d): " % len(ret))

            print (ret)
            rep.append(ret)

    return sample_context, sample_response, sample_labels, rep