import os
import tensorflow as tf
import numpy as np
import time
import gen.generator as gens
import disc.discriminator as discs
import utils.data_utils as data_utils
import utils.conf as conf
import pickle

gen_config  = conf.gen_config
disc_config = conf.disc_config
evl_config  = conf.disc_config

# prepare data for discriminator and generator
def disc_train_data(source_inputs, source_outputs, sample_context, sample_response, sample_labels, responses, mc_search=False, isDisc=True, temp=True):
    
    print("disc_train_data, mc_search: ", mc_search)
    rem_set = []
    for i in range(len(sample_labels)):
        if sample_labels[i] == 1:
            rem_set = [i] + rem_set

    for elem in rem_set:
        del sample_context[elem]
        del sample_response[elem]
        del sample_labels[elem]

    dataset = {}
    dataset['is_disc']  = True
    if temp:
        dataset['label']    = np.array([1] * (len(source_inputs)))
        for i in range(len(responses)):
            dataset['label'] = np.append(dataset['label'], 0)
    else:
        dataset['label'] = np.array([0] * len(responses))


    dataset['context'] = source_inputs if isDisc else []
    for context in sample_context:
        dataset['context'].append(context)

    dataset['response'] = source_outputs if isDisc else []
    if mc_search:
        for resp in sample_response:
            dataset['response'].append(resp)
    else:
        #responses = responses[0]
        for resp in sample_response:
            dataset['response'].append(resp[0])


    train_inputs, train_labels, train_masks = data_util.convert_to_format(data_util.dataset_padding(dataset, disc_config.max_len))

    return train_inputs, train_labels, train_masks, responses


# pre train generator
def gen_pre_train():
    gens.train(gen_config)

def gen_pre_train2():
    gen_config.batch_size = 1
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1*disc_config.init_scale,1*disc_config.init_scale)
        with tf.variable_scope("model",reuse=None,initializer=initializer):
            disc_model = discs.create_model(sess, disc_config, is_training=True)
        gen_model = gens.create_model(sess, gen_config)
        vocab, rev_vocab, dev_set, train_set = data_utils.prepare_data(gen_config)
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
            reward = disc_model.disc_step(sess,  train_inputs, train_labels, train_masks,do_train = False)
            print("Step %d, here are the discriminator logits:" % gstep)
            print(reward)
            steps.append(gstep)
            rewards.append(reward)
            pickle.dump(steps, open("steps.p", "wb"))
            pickle.dump(rewards, open("rewards.p","wb"))
            # Change data back into generator format
            responses[0] = [data_utils.GO_ID] + responses[0]
            dec_gen = responses[0][:gen_config.buckets[bucket_id][1]]
            if len(dec_gen)< gen_config.buckets[bucket_id][1]:
                dec_gen = dec_gen + [0]*(gen_config.buckets[bucket_id][1] - len(dec_gen))
            dec_gen = np.reshape(dec_gen, (-1,1))

            # Do a step of policy gradient on the generator

            gen_model.step(sess, encoder, dec_gen, weights, bucket_id, mode=gen_model.SM_POLICY_TRAIN, reward = reward[:,1])


# pre train discriminator
def disc_pre_train():
    gen_config.batch_size = 1
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1 * disc_config.init_scale, 1 * disc_config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            disc_model = discs.create_model(sess, disc_config, is_training=True)
        gen_model = gens.create_model(sess, gen_config)
        vocab, rev_vocab, dev_set, train_set = data_utils.prepare_data(gen_config)
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
            print("========lr=%f==============Update Discriminator step %d=======================" % (disc_model.lr.eval(),gstep))
            # 1.Sample (X,Y) from real data


            _, _, _, source_inputs, source_outputs = gen_model.get_batch(train_set, bucket_id, 0)
            #1.5 get sample from data to generate from
            _, _, _, gen_inputs, gen_outputs = gen_model.get_batch(train_set, bucket_id, 0)
            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
            train_inputs, train_labels, train_masks, _ = disc_train_data(sess, gen_model, vocab,
                                                                         source_inputs, source_outputs, gen_inputs, gen_outputs, mc_search=False, isDisc=True)
            # 3.Update D using (X, Y ) as positive examples and(X, ^Y) as negative examples
            disc_model.disc_step(sess, train_inputs, train_labels, train_masks, isTrain = False)

            if gstep > 0 and gstep % 600 == 0:
                sess.run(disc_model.lr.assign(disc_model.lr.eval()*0.6),[])
            if gstep > 0 and gstep % 1000 == 0:
                path = disc_model.saver.save(sess,checkpoint_prefix,global_step=disc_model.global_step)
                print("Saved model chechpoint to{}\n".format(path))



def sample_bucket_id(train_buckets_scale):
    return min([j for j in xrange(len(train_buckets_scale))
                                 if train_buckets_scale[j] > np.random.random_sample()])


# Adversarial Learning for Neural Dialogue Generation
def al_train():

    with tf.Session() as sess:
        
        initializer     = tf.random_uniform_initializer(-1 * disc_config.init_scale, 1 * disc_config.init_scale)
        
        #Create discriminator model
        with tf.variable_scope("model", reuse = None, initializer = initializer):
            disc_model  = discs.create_model(sess, disc_config, is_training = True)
        
        #Create generator model
        gen_model       = gens.create_model(sess, gen_config)
        
        #Get Vocabulary  
        vocab, _        = data_utils.initialize_vocabulary(gen_config.vocab_path)
        
        #Get training and development dataset from file
        train_path        = os.path.join(gen_config.data_dir, gen_config.train_data_file)
        dataset           = data_utils.create_dataset(train_path, is_disc = False)
        train_dataset, _  = data_utils.split_dataset(dataset, ratio = gen_config.train_ratio)

        #Split dataset into buckets
        print ("Reading development and training data (limit: %d)." % gen_config.max_train_data_size)        
        train_set         = data_utils.split_into_buckets(train_dataset, gen_config.buckets, gen_config.max_train_data_size) 

        #Data distribution according to buckets
        train_bucket_sizes  = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size    = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size 
                                                for i in xrange(len(train_bucket_sizes))]

        rewards             = []
        steps               = []
        perplexity          = []
        disc_loss           = []
        gstep               = 0

        while True:   

            gstep += 1
            
            for i in range(disc_config.iters):
                
                #Sample bucket id according to train_buckets_scale
                bucket_id = sample_bucket_id(train_buckets_scale)
                
                print("===========================Update Discriminator %d.%d=============================" % (gstep, i))
                # 1.Sample (X,Y) from real data
                source_inputs, source_outputs           = gen_model.get_batch(train_set, bucket_id)

                # 1.5 sample x,y from data to generate samples from
                gen_inputs, gen_outputs                 = gen_model.get_batch(train_set, bucket_id)
                
                encoder_inputs, decoder_inputs, _       = data_utils.src_to_gen(gen_inputs, gen_outputs, gen_config.buckets, bucket_id, gen_model.batch_size)
                
                # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
                #TODO Replace with Gil's function
                sample_responses                         = gens.gen_guided_sample(sess, encoder_inputs, decoder_inputs, gen_config, gen_model, vocab) 
                
                mixed_encoder  = list(np.concatenate((source_inputs, gen_inputs)))
                mixed_decoder  = list(np.concatenate((source_outputs, sample_responses)))
                mixed_labels   = list(np.concatenate(([1]*gen_model.batch_size, [0]*gen_model.batch_size)))

                train_inputs, train_labels, train_masks    = data_utils.src_to_disc(mixed_encoder, mixed_decoder, mixed_labels, disc_config.max_len)
               
                # 3.Update D using (X, Y ) as positive examples and(X, ^Y) as negative examples
                print('Discriminator input/labels:')
                print(train_inputs)
                print(train_labels)

                disc_l = disc_model.disc_step(sess, train_inputs, train_labels, train_masks)
                disc_loss.append(disc_l)
                
                pickle.dump(disc_loss, open("disc_loss.p", "wb"))

            for i in range(gen_config.iters):
                
                #Sample bucket id according to train_buckets_scale                
                bucket_id = sample_bucket_id(train_buckets_scale)

                print("===============================Update Generator %d.%d=============================" % (gstep, i))
                # 1.Sample (X,Y) from real data
                source_inputs, source_outputs               = gen_model.get_batch(train_set, bucket_id, 0)

                encoder, decoder, weights                   = data_utils.src_to_gen(source_inputs, source_outputs, gen_config.buckets, bucket_id, gen_model.batch_size) 

                # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X) with Monte Carlo search
                sample_response                             = gens.gen_guided_sample(sess, encoder, decoder, gen_config, gen_model, vocab)

                sample_labels                               = [0]*gen_model.batch_size

                train_inputs, train_labels, train_masks     = data_utils.src_to_disc(encoder, sample_response, sample_labels, disc_config.max_len)
                
                # 3.Compute Reward r for (X, ^Y ) using D.---based on Monte Carlo search
                reward                                      = disc_model.disc_step(sess,  train_inputs, train_labels, train_masks, do_train=False)
                
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
                decoder_inputs  = data_utils.transform_responses(sample_response) #TODO

                gen_model.step(sess, encoder, decoder_inputs, weights, bucket_id, mode=gen_model.SM_POLICY_TRAIN, reward=reward[:, 1])

         
            # 5.Teacher-Forcing: Update G on (X, Y )

            for i in range(gen_config.force_iters):
                print("===========================Force Generator %d.%d=============================" % (gstep, i))
                _, loss, _ = gen_model.step(sess, encoder, decoder, weights, bucket_id, mode = gen_model.SM_TRAIN)


        # add checkpoint
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
    parser.add_argument('--gen_batch', type=int)


    args = parser.parse_args()

    if (args.gen_batch):
        conf.gen_config.batch_size = args.gen_batch
    
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
