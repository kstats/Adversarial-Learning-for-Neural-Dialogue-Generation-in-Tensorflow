import os
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


def permute_seq(seq, perm):
    return [seq[perm[i]] for i in range(len(perm))]


# prepare data for discriminator and generator
def disc_train_data(sess, gen_model, vocab, source_inputs, source_outputs, gen_inputs, gen_outputs, bucket_id, mc_search=False, isDisc=True, temp=True):
    # sample_context2, sample_response2, sample_labels2, responses2 = gens.gen_sample(sess, gen_config, gen_model, vocab,
    #                                             gen_inputs, gen_outputs, mc_search=mc_search)
    sample_context, sample_response, sample_labels, responses = gens.gen_guided_sample(sess, gen_inputs, gen_outputs, gen_config, gen_model, vocab, bucket_id)
    # sample_responses_test  = gens.sample_from(sess, gen_inputs, bucket_id, gen_config, gen_model, vocab)

    #for n in range(len(sample_response)):
     #   if n % 2 == 1:
      #      sample_response[n] = [sample_response[n]]
    #responses = [responses]
    print("disc_train_data, mc_search: ", mc_search)
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
    perm = np.random.permutation(len(train_inputs))
    train_inputs = permute_seq(train_inputs,perm)
    train_labels = permute_seq(train_labels,perm)
    train_masks = np.transpose(permute_seq(np.transpose(train_labels,(1,0,2)),perm),(1,0,2))

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
        print("Its predictions on training data:")
        for i in prediction:
            print(i)
        return cost, accuracy        
    else:        
        fetches = [disc_model.cost,disc_model.accuracy,tf.nn.softmax(disc_model.logits),disc_model.summary]
        cost,accuracy,logits,summary = sess.run(fetches,feed_dict)
        return logits


def guided_disc_step(sess, disc_model, gen_model, train_inputs, train_labels, train_masks, bucket_id, do_train=True):
    blen = gen_model.buckets[bucket_id]
    
    def disc_to_gen_format(inp):
        # inp is a list of context/answer in disc format.
        ctx=inp[:,0,:]
        answer=inp[:,1,:]

        rc1 = np.flip(ctx[:,:blen[0]],1)
        rc2 = answer[:,:blen[1]-1]
        batch_size = len(ctx)
        go_vec = np.ones(batch_size).astype('int')*data_util.GO_ID
        rc2 = np.concatenate([np.transpose([go_vec]),rc2],axis=1)
        return rc1, rc2

    # halve everything (gen batch size idiocy), but only if we're training, then we have double the size of examples (pos+neg)...

    batch_size = len(train_inputs)
    if do_train:
        pos = train_inputs[0:batch_size/4,:,:]
        neg = train_inputs[batch_size/2:batch_size/2+batch_size/4,:,:]
        train_inputs = np.concatenate([pos,neg])    
        posl = train_labels[0:batch_size/4]
        negl = train_labels[batch_size/2:batch_size/2+batch_size/4]
        train_labels = np.concatenate([posl,negl])
        posm = train_masks[:,0:batch_size/4,:]
        negm = train_masks[:,batch_size/2:batch_size/2+batch_size/4,:]
        train_masks = np.concatenate([posm,negm],1)

        batch_size = batch_size / 2

    encoder_inputs, decoder_inputs = disc_to_gen_format(train_inputs)
    feed_dict = {gen_model.forward_only.name : False,
                      gen_model.do_projection.name: True,
                      gen_model.tf_bucket_id: bucket_id }
    for l in xrange(blen[0]):
        feed_dict[gen_model.encoder_inputs[l].name] = encoder_inputs.T[l]

    for l in xrange(blen[1]):
        feed_dict[gen_model.target_weights[l].name] = np.zeros(batch_size)
        feed_dict[gen_model.decoder_inputs[l].name] = decoder_inputs.T[l]

    # import pdb; pdb.set_trace()
    density = sess.run(gen_model.output_q[bucket_id],feed_dict)
 
    feed_dict = {disc_model.gen_density: density}
    feed_dict[disc_model.context] = train_inputs[:, 0, :]
    feed_dict[disc_model.response] = train_inputs[:, 1, :]
    feed_dict[disc_model.target] = train_labels

    feed_dict[disc_model.mask_c] = train_masks[:, :, 0]
    feed_dict[disc_model.mask_r] = train_masks[:, :, 1]

    disc_model.assign_new_batch_size(sess,len(train_inputs))
    if do_train:        
        fetches = [disc_model.cost,disc_model.accuracy,disc_model.train_op, disc_model.prediction]
        cost,accuracy,_ ,prediction= sess.run(fetches,feed_dict)
        print("the train cost is: %f and the train accuracy is %f ."%(cost, accuracy))
        print("Its predictions on training data:")
        for i in prediction:
            print(i)
        return cost, accuracy
    else:        
        fetches = [disc_model.cost,disc_model.accuracy,tf.nn.softmax(disc_model.logits)]
        cost,accuracy,probs = sess.run(fetches,feed_dict)
        return probs


# pre train discriminator
def disc_pre_train():    
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1 * disc_config.init_scale, 1 * disc_config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            disc_model = discs.create_model(sess, disc_config, is_training=True)
        # with tf.variable_scope("guided_disc", reuse=None, initializer=initializer):
        #     guided_disc_model = discs.create_guided_model(sess, disc_config, is_training=True)
        gen_model = gens.create_model(sess, gen_config)
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
            print("========lr=%f==============Update Discriminator step %d=======================" % (disc_model.lr.eval(),gstep))
            # 1.Sample (X,Y) from real data


            _, _, _, source_inputs, source_outputs = gen_model.get_batch(train_set, bucket_id, 0)
            #1.5 get sample from data to generate from
            _, _, _, gen_inputs, gen_outputs = gen_model.get_batch(train_set, bucket_id, 0)
            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
            train_inputs, train_labels, train_masks, _ = disc_train_data(sess, gen_model, vocab,
                                                                         source_inputs, source_outputs, gen_inputs, gen_outputs, bucket_id, mc_search=False, isDisc=True)
            # 3.Update D using (X, Y ) as positive examples and(X, ^Y) as negative examples
            # disc_l = guided_disc_step(sess, guided_disc_model, gen_model, train_inputs, train_labels, train_masks, bucket_id)
            disc_l, accuracy = disc_step(sess, disc_model, train_inputs, train_labels, train_masks)
            if gstep > 0 and gstep % 20 == 0:
                sess.run(disc_model.lr.assign(disc_model.lr.eval()*0.6),[])
            if gstep > 0 and gstep % 10 == 0:
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
                                                        source_inputs,source_outputs, source_inputs, source_outputs, bucket_id, mc_search=False, isDisc = False, temp=False)
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
    # gen_config.batch_size = 1
    with tf.Session() as sess:
        train_summary_dir = os.path.join(disc_config.out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        initializer = tf.random_uniform_initializer(-1 * disc_config.init_scale, 1 * disc_config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            disc_model = discs.create_model(sess, disc_config, is_training=True)
        # with tf.variable_scope("guided_disc", reuse=None, initializer=initializer):
        #     guided_disc_model = discs.create_guided_model(sess, disc_config, is_training=True)
        gen_model = gens.create_model(sess, gen_config)
        vocab, rev_vocab, dev_set, train_set = data_util.prepare_data(gen_config)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        disc_checkpoint_dir = os.path.abspath(os.path.join(disc_config.out_dir, "checkpoints"))
        disc_checkpoint_path = os.path.join(disc_checkpoint_dir, "disc.model")
        gen_checkpoint_path = os.path.join(gen_config.train_dir, "chitchat.model")

        rewards = []
        steps = []
        perplexity = []
        disc_loss = []
        gen_loss = []
        disc_steps = []
        gen_steps = []
        gstep = 0
        cumulative_step = 0
        while True:            
            gstep += 1
            steps.append(gstep)
            for i in range(disc_config.iters):
                bucket_id = min([j for j in xrange(len(train_buckets_scale))
                                 if train_buckets_scale[j] > np.random.random_sample()])
                print("===========================Update Discriminator %d.%d=============================" % (gstep, i))
                # 1.Sample (X,Y) from real data
                _, _, _, source_inputs, source_outputs = gen_model.get_batch(train_set, bucket_id, 0)
                # 1.5 sample x,y from data to generate samples from
                _, _, _, gen_inputs, gen_outputs = gen_model.get_batch(train_set, bucket_id, 0)
                # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
                train_inputs, train_labels, train_masks, _ = disc_train_data(sess, gen_model, vocab,
                                                                             source_inputs, source_outputs, gen_inputs,
                                                                             gen_outputs, bucket_id, mc_search=False, isDisc=True,
                                                                             temp=True)
                # 3.Update D using (X, Y ) as positive examples and(X, ^Y) as negative examples
                print('Discriminator input/labels:')
                print(train_inputs)
                print(train_labels)
                disc_steps.append((gstep-1) * disc_config.iters + i)
                disc_l, accuracy = disc_step(sess, disc_model, train_inputs, train_labels, train_masks)
                # import pdb; pdb.set_trace()
                # disc_l , accuracy = guided_disc_step(sess, guided_disc_model, gen_model, train_inputs, train_labels, train_masks, bucket_id)
                disc_loss.append((cumulative_step,disc_l))
                cumulative_step += 1                

            i = 0
            mean_prob = 0
            while i < gen_config.iters or mean_prob < 0.4:
                small_step = (gstep - 1) * gen_config.iters + i
                gen_steps.append(small_step)
                bucket_id = min([j for j in xrange(len(train_buckets_scale))
                                    if train_buckets_scale[j] > np.random.random_sample()])

                print("===============================Update Generator %d.%d=============================" % (gstep, i))
                # 1.Sample (X,Y) from real data
                update_gen_data = gen_model.get_batch(train_set, bucket_id, 0)
                encoder, decoder, weights, source_inputs, source_outputs = update_gen_data                
                # test = gens.sample_from(sess,np.asarray(encoder),bucket_id,gen_config,gen_model,vocab)
                # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X) with Monte Carlo search
                # train_inputs, train_labels, train_masks, responses = disc_train_data(sess,gen_model,vocab,
                #                                             source_inputs,source_outputs,source_inputs,source_outputs, mc_search=True, isDisc=False)
                train_inputs, train_labels, train_masks, responses = disc_train_data(sess, gen_model, vocab,
                                                                                     source_inputs, source_outputs,
                                                                                     source_inputs, source_outputs,bucket_id,
                                                                                     mc_search=False, isDisc=False,
                                                                                     temp=False)
                # 3.Compute Reward r for (X, ^Y ) using D.---based on Monte Carlo search
                reward = disc_step(sess, disc_model, train_inputs, train_labels, train_masks, do_train=False)
                #import pdb; pdb.set_trace()
                #rew = tf.summary.scalar("reward", tf.reduce_mean(reward[:,1]))
                #summ = sess.run(rew)
                #train_summary_writer.add_summary(summ, small_step)
                #rew2 = tf.summary.scalar("reward2", tf.reduce_mean(reward[:, 1]).eval())
                #sum2 = sess.run(rew2)
                #train_summary_writer.add_summary(sum2, small_step)
                print("Step %d, here are the discriminator inputs/labels on which we calculate reward:" % gstep)
                print(train_inputs)
                print(train_labels)
                print("Step %d, here are the discriminator logits:" % gstep)
                print(reward)
                mean_prob = np.mean(reward[:,1])
                print("Mean logits:")
                print(mean_prob)
                rewards.append(np.mean(reward[:,1]))
                gan_rewards = -np.log(reward[:,1]) + np.log(reward[:,0])
                print("GAN cost:")
                print(gan_rewards)
                gan_loss = np.mean(gan_rewards)
                print("GAN mean cost:")                
                print(gan_loss)
                gen_loss.append((cumulative_step, gan_loss))
                # 4.Update G on (X, ^Y ) using reward r
                # import pdb; pdb.set_trace()
                decoder_inputs = []
                for res in responses:
                    dec_gen = [data_util.GO_ID] + res[:gen_config.buckets[bucket_id][1]]
                    if len(dec_gen) < gen_config.buckets[bucket_id][1]:
                        dec_gen = dec_gen + [0] * (gen_config.buckets[bucket_id][1] - len(dec_gen))
                    dec_gen = np.reshape(dec_gen, (-1, 1))
                    decoder_inputs.append(dec_gen)
                # import pdb; pdb.set_trace()
                decoder_inputs = np.transpose(np.asarray(decoder_inputs))
                decoder_inputs = np.squeeze(decoder_inputs)
                gen_model.step(sess, encoder, decoder_inputs, weights, bucket_id, mode=gen_model.SM_POLICY_TRAIN,
                               # reward=reward[:,1])
                               reward=gan_rewards)

                i += 1
                cumulative_step += 1

            '''dec_gen = []
            for i in range(len(responses)):
                dec_gen.append(responses[i][:gen_config.buckets[bucket_id][1][0]])
                if len(dec_gen)< gen_config.buckets[bucket_id][1]:
                    dec_gen = dec_gen + [0]*(gen_config.buckets[bucket_id][1] - len(dec_gen))
            dec_gen = np.reshape(dec_gen, (-1,gen_config.batch_size,1))'''

            # 5.Teacher-Forcing: Update G on (X, Y )

            for i in range(gen_config.force_iters):
                print("===========================Force Generator %d.%d=============================" % (gstep, i))
                _, loss, _ = gen_model.step(sess, encoder, decoder, weights, bucket_id, mode=gen_model.SM_TRAIN)

                #print("loss: ", loss)
                perplexity.append(loss)
                # pickle.dump(perplexity, open("perplexity.p", "wb"))
            if gstep % disc_config.plot_every == 0:
                pickle.dump(disc_loss, open("disc_loss.p", "wb"))
                pickle.dump(gen_loss, open("gen_loss.p", "wb"))
                pickle.dump(disc_steps, open("disc_steps.p", "wb"))
                pickle.dump(steps, open("steps.p", "wb"))
                pickle.dump(rewards, open("rewards.p", "wb"))
                pickle.dump(perplexity, open("perplexity.p", "wb"))
                pickle.dump(gen_steps, open("gen_steps.p", "wb"))

            if gstep > 0 and gstep % 10 == 0:
                gen_model.saver.save(sess, gen_checkpoint_path, global_step=gen_model.global_step)
                guided_disc_model.saver.save(sess,disc_checkpoint_path,global_step=guided_disc_model.global_step)

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
