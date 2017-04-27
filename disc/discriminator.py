import tensorflow as tf
import numpy as np
import os
import time
import datetime
import disc_rnn_model as disc_rnn_model
import utils.data_utils as data_utils
import utils.conf as conf
import sys
sys.path.append('../utils')
import pdb


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)                
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def create_model(session, config, is_training):
    """Create translation model and initialize or load parameters in session."""
    model = disc_rnn_model.disc_rnn_model(config=config,is_training=is_training, isLstm=True)

    checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if is_training and ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)        
        # optimistic_restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created Disc_RNN model with fresh parameters.")
        if is_training:
            session.run(tf.global_variables_initializer())
    return model


def evaluate(model,session,data, batch_size,global_steps=None,summary_writer=None):

    correct_num=0
    total_num=len(data[0])
    for step, (x,y,mask_x) in enumerate(data_utils.batch_iter(data,batch_size=batch_size)):

        fetches = model.correct_num
        feed_dict={}
        

        feed_dict[model.context]=x[:,0,:]
        feed_dict[model.response] = x[:,1,:]
        feed_dict[model.target]=y

        feed_dict[model.mask_c]=mask_x[:,:,0]
        feed_dict[model.mask_r]=mask_x[:,:,1]
        model.assign_new_batch_size(session,len(x))

        count=session.run(fetches,feed_dict)
        correct_num+=count

    accuracy=float(correct_num)/total_num
    dev_summary = tf.summary.scalar('dev_accuracy',accuracy)
    dev_summary = session.run(dev_summary)
    if summary_writer:
        summary_writer.add_summary(dev_summary,global_steps)
        summary_writer.flush()
    return accuracy

def run_epoch(model,session,data,global_steps,valid_model,valid_data, batch_size, checkpoint_prefix, train_summary_writer, valid_summary_writer=None):
    for step, (x,y,mask_x) in enumerate(data_utils.batch_iter(data,batch_size=batch_size)):
        feed_dict={}

        feed_dict[model.context]=x[:,0,:]
        feed_dict[model.response] = x[:,1,:]
        feed_dict[model.target]=y

        feed_dict[model.mask_c]=mask_x[:,:,0]
        feed_dict[model.mask_r]=mask_x[:,:,1]
        model.assign_new_batch_size(session,len(x))
<<<<<<< HEAD
        fetches = [model.cost,model.accuracy,model.train_op,model.summary, model.prediction, model.logits,  model.grads]
        # state = session.run(model._initial_state)
        #for i , (c,h) in enumerate(model._initial_state):
         #   feed_dict[c]=state[i].c
          #  feed_dict[h]=state[i].h
        cost,accuracy,_,summary, pred, logits, grads  = session.run(fetches,feed_dict)
=======
        fetches = [model.cost,model.accuracy,model.train_op,model.summary, model.prediction, model.logits, model.lstm_w, model.grads]
        cost,accuracy,_,summary, pred, logits, w, grads  = session.run(fetches,feed_dict)
>>>>>>> bc3f9e1ef4f4d55ff36e6e98a87ddddb453f31ed
        # print (y)
        # print (pred)
        # print(logits)        
        # print(w)
        # print 'Printing grads!!\n\n\n'
        # print(grads)

        # import pdb; pdb.set_trace()

        #print (logits)
        #print (tf.argmax(logits,1))

        train_summary_writer.add_summary(summary,global_steps)
        train_summary_writer.flush()
        valid_accuracy=evaluate(valid_model,session,valid_data,batch_size,global_steps,valid_summary_writer)
        if(global_steps%10==0):
            print("the %i step, train cost is: %f and the train accuracy is %f and the valid accuracy is %f"%(model.global_step.eval(),cost,accuracy,valid_accuracy))
        if(global_steps%200==0):
            path = model.saver.save(session,checkpoint_prefix,global_step=model.global_step)
            print("Saved model chechpoint to{}\n".format(path))
        global_steps+=1

    return global_steps

def train_step(config_disc, config_evl):

    print("loading the disc train set")
    config = config_disc
    eval_config=config_evl
    eval_config.keep_prob=1.0

    train_data,valid_data,test_data=data_utils.disc_load_data(fname = config.train_data_file ,  debug = True, max_len = config.max_len)
    print("begin training")

    with tf.Graph().as_default(), tf.Session() as session:
        print("model training")
        initializer = tf.random_uniform_initializer(-1*config.init_scale,1*config.init_scale)
        with tf.variable_scope("model",reuse=None,initializer=initializer):
            model = create_model(session, config, is_training=True)
        with tf.variable_scope("model",reuse=True,initializer=initializer):
            valid_model = create_model(session, eval_config, is_training=False)
            test_model = create_model(session, eval_config, is_training=False)

        #add summary
        train_summary_dir = os.path.join(config.out_dir,"summaries","train")
        train_summary_writer =  tf.summary.FileWriter(train_summary_dir,session.graph)

        # dev_summary_op = tf.merge_summary([valid_model.loss_summary,valid_model.accuracy])
        dev_summary_dir = os.path.join(eval_config.out_dir,"summaries","dev")
        dev_summary_writer =  tf.summary.FileWriter(dev_summary_dir,session.graph)

        #add checkpoint
        checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "disc.model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        global_steps=1
        begin_time=int(time.time())

        for i in range(config.num_epoch):
            print("the %d epoch training..."%(i+1))
            lr_decay = config.lr_decay ** max(i-config.max_decay_epoch,0.0)
            model.assign_new_lr(session,config.lr*lr_decay)
            global_steps=run_epoch(model,session,train_data,global_steps,valid_model,
                                   valid_data, config_disc.batch_size, checkpoint_prefix, train_summary_writer,dev_summary_writer)

            if i% config.checkpoint_every==0:
                path = model.saver.save(session,checkpoint_prefix,global_step=model.global_step)
                print("Saved model chechpoint to{}\n".format(path))

        print("the train is finished")
        end_time=int(time.time())
        print("training takes %d seconds already\n"%(end_time-begin_time))
        test_accuracy=evaluate(test_model,session,test_data, config_disc.batch_size)
        print("the test data accuracy is %f"%test_accuracy)
        print("program end!")



def main(_):
    train_step()


if __name__ == "__main__":
    tf.app.run()
