import os
#import pdb
import tensorflow as tf
import numpy as np
import time
import gen.generator as gens
import disc.discriminator as discs
import utils.data_utils as data_util
import utils.conf as conf
import pickle

gen_config  = conf.gen_config
disc_config = conf.disc_config
evl_config  = conf.disc_config



# pre train generator
def gen_pre_train():
    gens.train(gen_config)

# prepare data for discriminator and generator
def disc_train_data(sess, gen_model, vocab, source_inputs, source_outputs, gen_inputs, gen_outputs, mc_search=False, isDisc=True, temp=True):
    # sample_context, sample_response, sample_labels, responses = gens.gen_sample(sess, gen_config, gen_model, vocab,
    #                                            gen_inputs, gen_outputs, mc_search=mc_search)
    sample_context, sample_response, sample_labels, responses = gens.gen_guided_sample(sess, gen_inputs, gen_outputs, gen_config, gen_model, vocab)

    sample_response[1] = [sample_response[1]]
    #import pdb; pdb.set_trace()
    responses = [responses]
    print("disc_train_data, mc_search: ", mc_search)
    # import pdb; pdb.set_trace()
    rem_set = []
    for i in range(len(sample_labels)):
        if sample_labels[i] == 1:
            rem_set = [i] + rem_set

    for elem in rem_set:
        del sample_context[elem]
        del sample_response[elem]
        del sample_labels[elem]
        # del responses[elem]

    dataset = {}
    dataset['is_disc']  = True
    if temp:
        dataset['label']    = np.array([1] * (len(source_inputs)))
        for i in range(len(responses)):
            dataset['label'] = np.append(dataset['label'], 0)
    else:
        dataset['label'] = np.array([0])


    dataset['context'] = source_inputs if isDisc else []
    for context in sample_context:
        dataset['context'].append(context)
    
    dataset['response'] = source_outputs if isDisc else []
    if mc_search:
        for resp in sample_response: 
            dataset['response'].append(resp)
    else:
        responses = responses[0]
        for resp in sample_response:
            dataset['response'].append(resp[0])
    
    # dataset['len'] = (len(source_inputs)-1) + len(sample_context) if isDisc else len(sample_context)

   
    '''resp = []
    for input, response, label in zip(sample_context, sample_response, sample_labels):
       print(str(label) + "\t" + str(input) + "\t" + str(response))
       resp.append(response)

    sample_inputs = zip(sample_context, sample_response)
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
    sorted_index = len_argsort(sample_inputs)'''


    '''train_set_x = []
    train_set_y = []
    train_set_x = [sample_context[i] for i in sorted_index]
    train_set_y = [sample_labels[i] for i in sorted_index]
    train_set=(train_set_x,train_set_y)'''

    train_inputs, train_labels, train_masks = data_util.convert_to_format(data_util.dataset_padding(dataset, disc_config.max_len))

    return train_inputs, train_labels, train_masks, responses

# discriminator api
def disc_step(sess, disc_model, train_inputs, train_labels, train_masks, do_train=True):
    feed_dict={}

    feed_dict[disc_model.context] = train_inputs[:, 0, :]
    feed_dict[disc_model.response] = train_inputs[:, 1, :]
    feed_dict[disc_model.target] = train_labels

    feed_dict[disc_model.mask_c] = train_masks[:, :, 0]
    feed_dict[disc_model.mask_r] = train_masks[:, :, 1]

    disc_model.assign_new_batch_size(sess,len(train_inputs))
    if do_train:
        fetches = [disc_model.cost,disc_model.accuracy,disc_model.train_op,disc_model.summary]
        cost,accuracy,_,summary = sess.run(fetches,feed_dict)
        print("the train cost is: %f and the train accuracy is %f ."%(cost, accuracy))
        return accuracy
    else:        
        fetches = [disc_model.cost,disc_model.accuracy,tf.nn.softmax(disc_model.logits),disc_model.summary]
        cost,accuracy,logits,summary = sess.run(fetches,feed_dict)
        return logits


# pre train discriminator
def disc_pre_train():
    gen_config.batch_size = 1
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1 * disc_config.init_scale, 1 * disc_config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            disc_model = discs.create_model(sess, disc_config, is_training=True)
        gen_model = gens.create_model(sess, gen_config)
        #import pdb; pdb.set_trace()
        vocab, rev_vocab, dev_set, train_set = data_util.prepare_data(gen_config)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        checkpoint_dir = os.path.abspath(os.path.join(disc_config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "disc.model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        while True:
            gstep = disc_model.global_step.eval()
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            #import pdb;
            #pdb.set_trace()
            print("========lr=%f==============Update Discriminator step %d=======================" % (disc_model.lr.eval(),gstep))
            # 1.Sample (X,Y) from real data

            # import pdb; pdb.set_trace()

            _, _, _, source_inputs, source_outputs = gen_model.get_batch(train_set, bucket_id, 0)
            #1.5 get sample from data to generate from
            _, _, _, gen_inputs, gen_outputs = gen_model.get_batch(train_set, bucket_id, 0)
            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
            train_inputs, train_labels, train_masks, _ = disc_train_data(sess, gen_model, vocab,
                                                                         source_inputs, source_outputs, gen_inputs, gen_outputs, mc_search=False, isDisc=True)
            # 3.Update D using (X, Y ) as positive examples and(X, ^Y) as negative examples
            disc_step(sess, disc_model, train_inputs, train_labels, train_masks, isTrain = False)

            if gstep > 0 and gstep % 600 == 0:
                sess.run(disc_model.lr.assign(disc_model.lr.eval()*0.6),[])
            if gstep > 0 and gstep % 1000 == 0:
                path = disc_model.saver.save(sess,checkpoint_prefix,global_step=disc_model.global_step)
                print("Saved model chechpoint to{}\n".format(path))


def gen_pre_train2():
    gen_config.batch_size = 1
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1*disc_config.init_scale,1*disc_config.init_scale)
        with tf.variable_scope("model",reuse=None,initializer=initializer):
            disc_model = discs.create_model(sess, disc_config, is_training=True)
        gen_model = gens.create_model(sess, gen_config)
        vocab, rev_vocab, dev_set, train_set = data_util.prepare_data(gen_config)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        gstep=0
        rewards = []
        steps = []
        while True:
            gstep += 1
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

            update_gen_data = gen_model.get_batch(train_set, bucket_id, 0)      
            encoder, decoder, weights, source_inputs, source_outputs = update_gen_data

            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X) with Monte Carlo search
            train_inputs, train_labels, train_masks, responses = disc_train_data(sess,gen_model,vocab,
                                                        source_inputs,source_outputs, source_inputs, source_outputs, mc_search=False, isDisc = False, temp=False)
            # 3.Compute Reward r for (X, ^Y ) using D.---based on Monte Carlo search
            reward = disc_step(sess, disc_model, train_inputs, train_labels, train_masks,do_train = False)
            print("Step %d, here are the discriminator logits:" % gstep)
            print(reward)
            steps.append(gstep)
            rewards.append(reward)
            pickle.dump(steps, open("steps.p", "wb"))
            pickle.dump(rewards, open("rewards.p","wb"))
            # Change data back into generator format
            responses[0] = [data_util.GO_ID] + responses[0]
            dec_gen = responses[0][:gen_config.buckets[bucket_id][1]]
            if len(dec_gen)< gen_config.buckets[bucket_id][1]:
                dec_gen = dec_gen + [0]*(gen_config.buckets[bucket_id][1] - len(dec_gen))
            dec_gen = np.reshape(dec_gen, (-1,1))

            # Do a step of policy gradient on the generator

            gen_model.step(sess, encoder, dec_gen, weights, bucket_id, mode=gen_model.SM_POLICY_TRAIN, reward = reward[:,1])

# Adversarial Learning for Neural Dialogue Generation
def al_train():
    gen_config.batch_size = 1
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1*disc_config.init_scale,1*disc_config.init_scale)
        with tf.variable_scope("model",reuse=None,initializer=initializer):
            disc_model = discs.create_model(sess, disc_config, is_training=True)
        gen_model = gens.create_model(sess, gen_config)
        vocab, rev_vocab, dev_set, train_set = data_util.prepare_data(gen_config)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        rewards = []
        steps = []
        perplexity = []
        disc_loss = []
        gstep = 0
        while True:
            random_number_01 = np.random.random_sample()
            gstep += 1
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01])

            for i in range(disc_config.iters):
                #import pdb; pdb.set_trace()
                print("===========================Update Discriminator %d.%d=============================" % (gstep,i))
                # 1.Sample (X,Y) from real data
                _, _, _, source_inputs, source_outputs = gen_model.get_batch(train_set, bucket_id, 0)
                #1.5 sample x,y from data to generate samples from
                _, _, _, gen_inputs, gen_outputs = gen_model.get_batch(train_set, bucket_id, 0)
                # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
                train_inputs, train_labels, train_masks, _ = disc_train_data(sess,gen_model,vocab,
                                                            source_inputs,source_outputs,gen_inputs, gen_outputs, mc_search=False, isDisc=True, temp=True)
                # 3.Update D using (X, Y ) as positive examples and(X, ^Y) as negative examples
                print('Discriminator input/labels:')
                print(train_inputs)
                print(train_labels)
                disc_l = disc_step(sess, disc_model, train_inputs, train_labels, train_masks)
                disc_loss.append(disc_l)
                pickle.dump(disc_loss, open("disc_loss.p", "wb"))

            for i in range(gen_config.iters):
                print("===============================Update Generator %d.%d=============================" % (gstep,i))
                # 1.Sample (X,Y) from real data
                update_gen_data = gen_model.get_batch(train_set, bucket_id, 0)
                encoder, decoder, weights, source_inputs, source_outputs = update_gen_data

                # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X) with Monte Carlo search
                # train_inputs, train_labels, train_masks, responses = disc_train_data(sess,gen_model,vocab,
                #                                             source_inputs,source_outputs,source_inputs,source_outputs, mc_search=True, isDisc=False)
                train_inputs, train_labels, train_masks, responses = disc_train_data(sess,gen_model,vocab,
                                                            source_inputs,source_outputs,source_inputs,source_outputs, mc_search=False, isDisc=False, temp=False)
                # 3.Compute Reward r for (X, ^Y ) using D.---based on Monte Carlo search
                reward = disc_step(sess, disc_model, train_inputs, train_labels, train_masks,do_train = False)                
                print("Step %d, here are the discriminator inputs/labels on which we calculate reward:" % gstep)
                print(train_inputs)
                print(train_labels)
                print("Step %d, here are the discriminator logits:" % gstep)
                steps.append(gstep)
                pickle.dump(steps, open("steps.p", "wb"))                

                print(reward)
                rewards.append(reward)
                pickle.dump(rewards, open("rewards.p", "wb"))
                # 4.Update G on (X, ^Y ) using reward r
                dec_gen = responses[0][:gen_config.buckets[bucket_id][1]]
                if len(dec_gen)< gen_config.buckets[bucket_id][1]:
                    dec_gen = dec_gen + [0]*(gen_config.buckets[bucket_id][1] - len(dec_gen))
                dec_gen = np.reshape(dec_gen, (-1,1))
                gen_model.step(sess, encoder, dec_gen, weights, bucket_id, mode=gen_model.SM_POLICY_TRAIN, reward = reward[:,1])


            '''dec_gen = []
            for i in range(len(responses)):
                dec_gen.append(responses[i][:gen_config.buckets[bucket_id][1][0]])
                if len(dec_gen)< gen_config.buckets[bucket_id][1]:
                    dec_gen = dec_gen + [0]*(gen_config.buckets[bucket_id][1] - len(dec_gen))
            dec_gen = np.reshape(dec_gen, (-1,gen_config.batch_size,1))'''

            # 5.Teacher-Forcing: Update G on (X, Y )

            for i in range(gen_config.force_iters):
                print("===========================Force Generator %d.%d=============================" % (gstep,i))            
                _, loss, _ = gen_model.step(sess, encoder, decoder, weights, bucket_id, mode=gen_model.SM_TRAIN)
            
            # print("loss: ", loss)
            # perplexity.append(loss)
            # pickle.dump(perplexity, open("perplexity.p", "wb"))

        #add checkpoint
        checkpoint_dir = os.path.abspath(os.path.join(disc_config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "disc.model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        pass

def main(_):
    seed = int(time.time())
    np.random.seed(seed)  

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('train_type', type=str)
    parser.add_argument('--gen_file', type=str)
    parser.add_argument('--disc_file', type=str)
    parser.add_argument('--gen_prob', type=float)
    parser.add_argument('--disc_prob', type=float)

    args = parser.parse_args()

    if (args.gen_file):
        conf.gen_config.train_data_file = args.gen_file

    if(args.gen_prob):
        conf.gen_config.keep_prob = args.gen_prob

    if (args.disc_file):
        conf.disc_config.train_data_file = args.disc_file

    if(args.disc_prob):
        conf.disc_config.keep_prob = args.disc_prob
 
    if args.train_type == 'disc': 
        print ("Runinng Discriminator Pre-Train") 
        disc_pre_train()
    elif args.train_type == 'gen':
        print ("Runinng Generator Pre-Train")
        gen_pre_train()
    elif args.train_type == 'gen2':
        print ("Runinng Generator Pre-Train 2")
        gen_pre_train2()
    else:
        print ("Runinng Adversarial")        
        al_train()


if __name__ == "__main__":
    tf.app.run()
