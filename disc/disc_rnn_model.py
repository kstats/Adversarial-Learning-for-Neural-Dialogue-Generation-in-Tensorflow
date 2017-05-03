import tensorflow as tf
import numpy as np

class disc_rnn_model(object):

    def __init__(self, config, scope_name="disc_rnn", is_training=True, isLstm=False):
        self.scope_name = scope_name
        with tf.variable_scope(self.scope_name):
            self.keep_prob=config.keep_prob
            self.batch_size=tf.Variable(0,dtype=tf.int32,trainable=False)

            max_len=config.max_len
            self.input_data=tf.placeholder(tf.int32,[None,max_len])
            self.target = tf.placeholder(tf.int64,[None])
            self.mask_c = tf.placeholder(tf.float32,[max_len,None])
            self.mask_r = tf.placeholder(tf.float32,[max_len,None])
            self.context = tf.placeholder(tf.int32, [None,max_len])
            self.response = tf.placeholder(tf.int32, [None,max_len])


            class_num=config.class_num
            hidden_neural_size=config.hidden_neural_size
            vocabulary_size=config.vocabulary_size
            embed_dim=config.embed_dim
            hidden_layer_num=config.hidden_layer_num
            self.new_batch_size = tf.placeholder(tf.int32,shape=[],name="new_batch_size")
            self._batch_size_update = tf.assign(self.batch_size,self.new_batch_size)

            #build LSTM network
            lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(hidden_neural_size,forget_bias=0.0,state_is_tuple=True)
            if self.keep_prob<1:
                lstm_cell =  tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell,output_keep_prob=self.keep_prob
                )

            cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([lstm_cell]*hidden_layer_num,state_is_tuple=True)

            #builds second LSTM network
            lstm_cell2 = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(hidden_neural_size, forget_bias=0.0, state_is_tuple=True)
            if self.keep_prob < 1:
                lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell2, output_keep_prob=self.keep_prob
                )

            cell2 = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([lstm_cell2] * hidden_layer_num, state_is_tuple=True)

            self._initial_state = cell.zero_state(self.batch_size,dtype=tf.float32)

            #embedding layer
            #TODO should we make this only one embedding lookup?
            with tf.device("/cpu:0"),tf.name_scope("embedding_layer_context"):
                embedding = tf.get_variable("embedding",[vocabulary_size,embed_dim],dtype=tf.float32)
                context_inputs = self.context_inputs=tf.nn.embedding_lookup(embedding,self.context) #[batch_size, max_len, embed_dim]
                response_inputs = self.response_inputs=tf.nn.embedding_lookup(embedding,self.response) #[batch_size, max_len, embed_dim]

            #TODO how should I handle both dropouts?
            if self.keep_prob<1:
                context_inputs = tf.nn.dropout(context_inputs,self.keep_prob)
                response_inputs = tf.nn.dropout(response_inputs,self.keep_prob)


            def extract_axis_1(data, ind):
                """
                Get specified elements along the first axis of tensor.
                :param data: Tensorflow tensor that will be subsetted.
                :param ind: Indices to take (one for each element along axis 0 of data).
                :return: Subsetted tensor.
                """

                batch_range = tf.range(tf.shape(data)[0])
                indices = tf.stack([batch_range, ind], axis=1)
                res = tf.gather_nd(data, indices)

                return res

            self.mask_c_len = tf.count_nonzero(self.mask_c, 0, dtype=tf.int32)
            self.mask_r_len = tf.count_nonzero(self.mask_r, 0, dtype=tf.int32)
            with tf.variable_scope("LSTM_layer_context"):
                self.out_put1_test, state = tf.nn.dynamic_rnn(cell, context_inputs, sequence_length = self.mask_c_len, \
                initial_state = self._initial_state)
                self.out_put1 = self.out_put1_test[:,-1,:]
                self.output1 = extract_axis_1(self.out_put1_test,self.mask_c_len-1)
                out_put1 = self.output1

            with tf.variable_scope("LSTM_layer_response"):
                self.out_put2_test, state = tf.nn.dynamic_rnn(cell2, response_inputs, sequence_length = self.mask_r_len, initial_state = self._initial_state)
                self.out_put2 = self.out_put2_test[:,-1,:]
                self.output2 = extract_axis_1(self.out_put2_test,self.mask_r_len-1)
                out_put2 = self.output2

            #if not isLstm:
            with tf.variable_scope("Combine_LSTM"):
                cat_input = tf.concat([out_put1, out_put2],1)
                self.lstm_w = tf.get_variable("lstm_w", [cat_input.get_shape()[1], hidden_neural_size], dtype=tf.float32, initializer=tf.random_normal_initializer())
                lstm_b = tf.get_variable("lstm_b", [hidden_neural_size], dtype=tf.float32, initializer=tf.random_normal_initializer())
                out_put = tf.nn.relu(tf.matmul(cat_input,self.lstm_w)+lstm_b)  #tf.layers.dense(inputs=input, units=1024, activation=tf.nn.relu)
                #dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
            '''else:
                with tf.variable_scope("Combine_LSTM"):
                    lstm_cell3 = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(hidden_neural_size, forget_bias=0.0,
                                                                            state_is_tuple=True)
                    if self.keep_prob < 1:
                        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                            lstm_cell3, output_keep_prob=self.keep_prob
                        )
                    cell3 = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([lstm_cell3] * hidden_layer_num,
                                                                      state_is_tuple=True)
                    self._initial_state3 = cell3.zero_state(self.batch_size, dtype=tf.float32)

                    #self.comb_inputs = tf.Variable(tf.zeros([None, 2, hidden_neural_size]), name="combined_input")
                    #self.comb_inputs[:,0,:] = out_put1
                    #self.comb_inputs[:,1,:] = out_put2
                    self.comb_inputs = tf.transpose(tf.stack([out_put1, out_put2]), [1,0,2])
                    #self.comb_inputs.transpose([1,0,2])
                    self.size = tf.ones([self.batch_size], dtype=tf.int32) + 1
                    self.out_put3_test, state = tf.nn.dynamic_rnn(cell3, self.comb_inputs,
                                                                 sequence_length=self.size,
                                                                 initial_state=self._initial_state3)
                    out_put = extract_axis_1(self.out_put3_test, tf.ones([self.batch_size], dtype=tf.int32))
                    #dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
'''
            with tf.name_scope("Softmax_layer_and_output"):
                softmax_w = tf.get_variable("softmax_w",[hidden_neural_size,class_num],dtype=tf.float32, initializer=tf.random_normal_initializer())
                softmax_b = tf.get_variable("softmax_b",[class_num],dtype=tf.float32, initializer=tf.random_normal_initializer())
                self.logits = tf.matmul(out_put,softmax_w)+softmax_b

            with tf.name_scope("loss"):
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits+1e-10,labels=self.target)
                self.cost = tf.reduce_mean(self.loss)

            with tf.name_scope("accuracy"):
                self.prediction = tf.argmax(self.logits,1)
                correct_prediction = tf.equal(self.prediction,self.target)
                self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")

            #add summary
            loss_summary = tf.summary.scalar("loss",self.cost)
            accuracy_summary=tf.summary.scalar("accuracy_summary",self.accuracy)

            if not is_training:
                return

            self.global_step = tf.Variable(0,name="global_step",trainable=False)
            self.lr = tf.Variable(config.lr,trainable=False)

            tvars = tf.trainable_variables()
            self.tvars = tvars
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          config.max_grad_norm)

            self.grads = grads


            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, tvars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            self.grad_summaries_merged = tf.summary.merge(grad_summaries)

            self.summary =tf.summary.merge([loss_summary,accuracy_summary,self.grad_summaries_merged])



            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op=optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

            self.new_lr = tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
            self._lr_update = tf.assign(self.lr,self.new_lr)

            all_variables = [k for k in tf.global_variables() if self.scope_name in k.name]
            self.saver = tf.train.Saver(all_variables)

    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value})
    def assign_new_batch_size(self,session,batch_size_value):
        session.run(self._batch_size_update,feed_dict={self.new_batch_size:batch_size_value})



















