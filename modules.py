# -*- coding:utf-8 -*-
import tensorflow as tf

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {'PAD':0,'EOS':1}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "EOS"}
        self.n_words = 2  # Count PAD and EOS

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
        return self.word2index[word]
    
class LayerNormGRUCell(tf.contrib.rnn.RNNCell):
    def __init__(self,num_units,dropout_keep_prob=1.0,activation=None,reuse=None):
        self.num_units = num_units
        self.dropout_keep_prob = dropout_keep_prob
        self.activation = tf.nn.tanh if activation == None else activation
        self.reuse = reuse
    
    @property
    def state_size(self):
        return self.num_units
    
    @property
    def output_size(self):
        return self.num_units
    
    def __call__(self,inputs,state):
        with tf.variable_scope("LayerNormGRUCell",reuse=self.reuse):
            cat = tf.concat([inputs,state],axis=1)
            #Layer normalization to speed up training before nonlinear transformation.
            #paper https://arxiv.org/pdf/1607.06450v1.pdf
            z = tf.contrib.layers.layer_norm(
                tf.layers.dense(cat,self.num_units),
                activation_fn=tf.nn.sigmoid
                )
            r = tf.contrib.layers.layer_norm(
                tf.layers.dense(cat,self.num_units),
                activation_fn=tf.nn.sigmoid
                )
            b_h = tf.get_variable("b_h",
                                  [self.num_units],
                                  tf.float32,
                                  tf.constant_initializer(0.0)
                                  )
            h_ = tf.layers.dense(inputs,self.num_units,use_bias=False) +\
                r*tf.layers.dense(state,self.num_units,use_bias=False) +\
                b_h
            h_ = tf.contrib.layers.layer_norm(h_,activation_fn=self.activation)
            next_state = (1-z) * state + z * h_ 
            #Dropout on rnn output to reduce overfitting.
            outputs = tf.nn.dropout(next_state,self.dropout_keep_prob)
            return outputs,next_state
        
class WordEncoder():
    def __init__(self,num_units,num_hiddens,dropout_keep_prob=1.0,activation=None,reuse=None,name="WordEncoder"):
        cells = []
        for i in range(num_hiddens):
            cell = LayerNormGRUCell(num_units,dropout_keep_prob,activation)
            cells.append(cell)
        self.cell_fw = tf.contrib.rnn.MultiRNNCell(cells)
        cells = []
        for i in range(num_hiddens):
            cell = LayerNormGRUCell(num_units,dropout_keep_prob,activation)
            cells.append(cell)
        self.cell_bw = tf.contrib.rnn.MultiRNNCell(cells)
        self.reuse = reuse
        self.name = name
        
    def __call__(self,batch_size,inputs,sequence_length=None):
        '''
        inputs: shape=(batch_size,max_sent_num,max_sent_length,embedding_dim)
        sequence_length: shape=(batch_size,max_sent_num)
        '''
        with tf.variable_scope(self.name,reuse=self.reuse):
            sh = inputs.get_shape().as_list()
            #(batch_size*max_sent_num,max_sent_length,embedding_dim)
            inputs = tf.reshape(inputs,[-1,sh[2],sh[3]]) 
            sequence_length = tf.reshape(sequence_length,[-1,]) #(batch_size*max_sent_num,)
            init_state_fw = self.cell_fw.zero_state(batch_size*sh[1],tf.float32)
            init_state_bw = self.cell_bw.zero_state(batch_size*sh[1],tf.float32)
            outputs,_ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw,
                                                        self.cell_bw,
                                                        inputs,
                                                        sequence_length,
                                                        init_state_fw,
                                                        init_state_bw,
                                                        tf.float32
                                                        )
            
            #(batch_size*max_sent_num,max_sent_length,2*num_units)
            outputs = tf.concat(outputs,axis=2)
            return outputs,sequence_length
        
class WordAttention():
    def __init__(self,
                 num_units,
                 score_mask_value=-1e30,
                 reuse=None,
                 name="WordAttention"
                 ):
        self.num_units = num_units
        self.score_mask_value = score_mask_value
        self.reuse = reuse
        self.name = name
        
    def __call__(self,inputs,sequence_length,batch_size,dropout_keep_prob=1.0):
        '''
        inputs: shape=(batch_size*max_sent_num,max_sent_length,dim)
        sequence_length: shape=(batch_size*max_sent_num,)
        '''
        with tf.variable_scope(self.name,reuse=self.reuse):
            sh = inputs.get_shape().as_list()
            U = tf.contrib.layers.layer_norm(tf.layers.dense(inputs,self.num_units),
                                         activation_fn=tf.nn.tanh
                                         )
            scores = tf.squeeze(tf.layers.dense(U,1,use_bias=False)) #(batch_size*max_sent_num,max_sent_length)
            #The padding parts in the sentence will be determine false.
            mask1 = tf.sequence_mask(sequence_length,sh[1])
            padding = tf.ones_like(scores) * self.score_mask_value
            alignments = tf.nn.softmax(tf.where(mask1,scores,padding))
            #The sentences with a true length of 0 in the text will be determine false.
            mask2 = tf.tile(
                tf.expand_dims(tf.cast(tf.sign(sequence_length),tf.bool),1),
                [1,sh[1]]
                )
            alignments = tf.nn.dropout(
                tf.where(mask2,alignments,tf.zeros_like(alignments)),
                dropout_keep_prob
                )
            #contexts: shape=(batch_size*max_sent_num,dim)
            contexts = tf.squeeze(tf.matmul(tf.expand_dims(alignments,1),inputs))
            #outputs: shape=(batch_size,max_sent_num,dim)
            outputs = tf.reshape(contexts,[batch_size,-1,sh[2]])
            #output_sequence_length: shape=(batch_size,)
            output_sequence_length = tf.reduce_sum(
                tf.sign(tf.reshape(sequence_length,[batch_size,-1])),
                1
            )
            return outputs,output_sequence_length
    
class SentenceEncoder():
    def __init__(self,
                 num_units,
                 num_hiddens,
                 dropout_keep_prob=1.0,
                 activation=None,
                 reuse=None,
                 name="SentenceEncoder"
                 ):
        cells = []
        for i in range(num_hiddens):
            cell = LayerNormGRUCell(num_units,dropout_keep_prob,activation)
            cells.append(cell)
        self.cell_fw = tf.contrib.rnn.MultiRNNCell(cells)
        cells = []
        for i in range(num_hiddens):
            cell = LayerNormGRUCell(num_units,dropout_keep_prob,activation)
            cells.append(cell)
        self.cell_bw = tf.contrib.rnn.MultiRNNCell(cells)
        self.reuse = reuse
        self.name = name
        
    def __call__(self,batch_size,inputs,sequence_length=None):
        '''
        inputs: shape=(batch_size,max_sent_num,dim)
        sequence_length: shape=(batch_size,)
        '''
        with tf.variable_scope(self.name,reuse=self.reuse):
            init_state_fw = self.cell_fw.zero_state(batch_size,tf.float32)
            init_state_bw = self.cell_bw.zero_state(batch_size,tf.float32)            
            outputs,_ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw,
                                                        self.cell_bw,
                                                        inputs,
                                                        sequence_length,
                                                        init_state_fw,
                                                        init_state_bw,
                                                        tf.float32
                                                        )
            # output: shape=(batch_size,max_sent_num,2*dim)
            outputs = tf.concat(outputs,2) 
            return outputs,sequence_length

class SentenceAttention():
    def __init__(self,
                 num_units,
                 score_mask_value=-1e30,
                 reuse=None,
                 name="SentenceAttention"
                 ):
        self.num_units = num_units
        self.score_mask_value = score_mask_value
        self.reuse = reuse
        self.name = name
    
    def __call__(self,inputs,sequence_length,batch_size,dropout_keep_prob=1.0):
        '''
        inputs: shape=(batch_size,max_sent_num,dim)
        sequence_length: shape=(batch_size,)
        '''
        sh = inputs.get_shape().as_list()
        U = tf.contrib.layers.layer_norm(
            tf.layers.dense(inputs,self.num_units),
            activation_fn=tf.nn.tanh
        )
        scores = tf.squeeze(tf.layers.dense(U,1,use_bias=False)) # (batch_size,max_sent_num)
        mask = tf.sequence_mask(sequence_length,sh[1]) #(batch_size,max_sent_num)
        padding = tf.ones_like(scores) * self.score_mask_value
        alignments = tf.nn.dropout(
            tf.nn.softmax(tf.where(mask,scores,padding)),
            dropout_keep_prob
        )
        #(batch_size,dim)
        contexts = tf.squeeze(tf.matmul(tf.expand_dims(alignments,1),inputs))
        return contexts 
