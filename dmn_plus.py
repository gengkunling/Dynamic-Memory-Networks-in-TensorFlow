from __future__ import print_function
from __future__ import division

import sys
import time

import numpy as np
from copy import deepcopy

import tensorflow as tf
from attention_gru_cell import AttentionGRUCell
from answer_gru_cell import AnswerCell

from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

import contract_input
from tensorflow.python.ops import array_ops

class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 100
    embed_size = 80
    hidden_size = 80

    max_epochs = 200
    early_stopping = 20

    dropout = 0.5
    lr = 0.001 * 10
    l2 = 0.001

    cap_grads = False
    # gkl
    max_grad_val = 10
    #max_grad_val = 12
    noisy_grads = False

    word2vec_init = False
    embedding_init = np.sqrt(3) 

    # set to zero with strong supervision to only train gates
    strong_supervision = False
    beta = 1

    # NOTE not currently used hence non-sensical anneal_threshold
    anneal_threshold = 1000
    anneal_by = 1.5

    num_hops = 3
    num_attention_features = 4

    max_allowed_inputs = 130
    # num_train = 900

    floatX = np.float32

    babi_id = "1"
    babi_test_id = ""

    train_mode = True

def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

# from https://github.com/domluna/memn2n
def _position_encoding(sentence_size, embedding_size):
    """Position encoding described in section 4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)"""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

class DMN_PLUS(object):

    def load_data(self, debug=False):
        """Loads train/valid/test data and sentence encoding"""
        if self.config.train_mode:
            # self.train, self.valid, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size = babi_input.load_babi(self.config, split_sentences=True)
            self.train, self.valid, self.word_embedding, self.max_q_len, self.max_a_len, self.max_input_len, self.max_sen_len, \
                self.num_supporting_facts, self.vocab_size = contract_input.load_data(self.config, split_sentences=True)

        else:
            # self.test, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size = babi_input.load_babi(self.config, split_sentences=True)
            self.test, self.word_embedding, self.max_q_len, self.max_a_len, self.max_input_len, self.max_sen_len,\
                self.num_supporting_facts, self.vocab_size = contract_input.load_data(
                self.config, split_sentences=True)

        self.encoding = _position_encoding(self.max_sen_len, self.config.embed_size)

    def add_placeholders(self):
        """add data placeholder to graph"""
        self.question_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_q_len))
        self.input_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_input_len, self.max_sen_len))

        self.question_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))
        self.answer_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))
        self.input_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))

        self.answer_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_a_len))

        print('answer place holder size')
        print(self.answer_placeholder.get_shape())
        print('max a len')
        print(self.max_a_len)

        print('question place holder size')
        print(self.question_placeholder.get_shape())
        print('max q len')
        print(self.max_q_len)

        self.rel_label_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.num_supporting_facts))

        self.dropout_placeholder = tf.placeholder(tf.float32)

    def get_predictions(self, output):
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred
      
    def add_loss_op(self, output):
        """Calculate loss"""

        print('output size')
        print(output.get_shape())

        # output = output
        # embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")
        # optional strong supervision of attention with supporting facts
        gate_loss = 0
        if self.config.strong_supervision:
            for i, att in enumerate(self.attentions):
                labels = tf.gather(tf.transpose(self.rel_label_placeholder), 0)
                gate_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=att, labels=labels))

        # answers = tf.nn.embedding_lookup(embeddings, self.answer_placeholder)
        answers = self.answer_placeholder
        print(answers)
        #answers = tf.reshape(answers, [self.config.batch_size, -1])
        # loss = self.config.beta*tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=answers)) + gate_loss
        # loss = self.config.beta * tf.reduce_sum(tf.contrib.keras.backend.categorical_crossentropy(output=output, target=answers, from_logits=True)) + gate_loss

        # self defined loss funciton
        # sum the loss for each word
        loss = gate_loss
        for i in range(output.get_shape()[1]):
            s_output = tf.squeeze(output[:, i, :])
            s_answer = tf.squeeze(answers[:,i])
            loss += self.config.beta * tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s_output, labels=s_answer))


        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += self.config.l2*tf.nn.l2_loss(v)

        tf.summary.scalar('loss', loss)

        return loss
        
    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        gvs = opt.compute_gradients(loss)

        # optionally cap and noise gradients to regularize
        if self.config.cap_grads:
            gvs = [(tf.clip_by_norm(grad, self.config.max_grad_val), var) for grad, var in gvs]
        if self.config.noisy_grads:
            gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]

        train_op = opt.apply_gradients(gvs)
        return train_op
  

    def get_question_representation(self, embeddings):
        """Get question vectors via embedding and GRU"""
        questions = tf.nn.embedding_lookup(embeddings, self.question_placeholder)

        #debug
        print('questions')
        print(questions.get_shape())

        gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        _, q_vec = tf.nn.dynamic_rnn(gru_cell,
                questions,
                dtype=np.float32,
                sequence_length=self.question_len_placeholder
        )
        # the output of the question module is the final state

        print('q_vec size:')
        print(q_vec.get_shape())
        return q_vec

    def get_input_representation(self, embeddings):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)

        # use encoding to get sentence representation
        inputs = tf.reduce_sum(inputs * self.encoding, 2)

        forward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        backward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                forward_gru_cell,
                backward_gru_cell,
                inputs,
                dtype=np.float32,
                sequence_length=self.input_len_placeholder
        )

        # f<-> = f-> + f<-
        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)

        # apply dropout
        # q: is dropout necessary?
        fact_vecs = tf.nn.dropout(fact_vecs, self.dropout_placeholder)

        print('inputs size')
        print(inputs.get_shape())
        print('input vec')
        print(fact_vecs.get_shape())

        return fact_vecs

    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=reuse):

            features = [fact_vec*q_vec,
                        fact_vec*prev_memory,
                        tf.abs(fact_vec - q_vec),
                        tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(features, 1)

            attention = tf.contrib.layers.fully_connected(feature_vec,
                            self.config.embed_size, # no. of output neurons
                            activation_fn=tf.nn.tanh,
                            reuse=reuse, scope="fc1")
        
            attention = tf.contrib.layers.fully_connected(attention,
                            1, # no. of output neurons
                            activation_fn=None, # sigmoidal in the paper?
                            reuse=reuse, scope="fc2")
            
        return attention

    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""

        attentions = [tf.squeeze(
            self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
            for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)

        reuse = True if hop_index > 0 else False
        
        # concatenate fact vectors and attentions for input into attGRU
        gru_inputs = tf.concat([fact_vecs, attentions], 2)

        with tf.variable_scope('attention_gru', reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(self.config.hidden_size),
                    gru_inputs,
                    dtype=np.float32,
                    sequence_length=self.input_len_placeholder
            )

        return episode


    # GKL update the answer module for output with multiple words
    def add_answer_module(self, last_mem, q_vec):
        """reccurent answer module"""


        # replicate the q_vec  with no. of time-stamps
        q_vec = [q_vec for _ in range(self.max_a_len)]
        q_vec = tf.stack(q_vec, 1)
        # q_vec = tf.expand_dims(q_vec, 0)

        print('q_vec stack size')
        print(q_vec.get_shape())

        print('last_mem size')
        print(last_mem.get_shape)



        # initial_state = tf.concat([last_mem, tf.zeros_like(last_mem)], 1)
        append_zeros = tf.zeros([self.config.batch_size, self.vocab_size])
        # initial_state = tf.concat([last_mem, append_zeros], 1)
        initial_state = (last_mem, append_zeros)

        #print('init state size')
        #print(initial_state.get_shape())

        answer_cell = AnswerCell(num_units=self.config.hidden_size, vocab_size=self.vocab_size)
        output, _ = tf.nn.dynamic_rnn(answer_cell,
                                     q_vec,
                                     dtype=np.float32,
                                     sequence_length=self.answer_len_placeholder,
                                     initial_state=initial_state
                                     )

        return output




    def inference(self):
        """Performs inference on the DMN model"""

        # set up embedding
        embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")
         
        # input fusion module
        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get question representation')
            q_vec = self.get_question_representation(embeddings)
         

        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get input representation')
            fact_vecs = self.get_input_representation(embeddings)

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> build episodic memory')

            # generate n_hops episodes
            prev_memory = q_vec

            for i in range(self.config.num_hops):
                # get a new episode
                print('==> generating episode', i)
                episode = self.generate_episode(prev_memory, q_vec, fact_vecs, i)

                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec], 1),
                            self.config.hidden_size,
                            activation=tf.nn.relu)

            last_mem = prev_memory

        # pass memory module output through linear answer module
        #debug: the size of the tensors are not match
        print("last_mem size:")
        print(last_mem.get_shape())
        print('q_vec size:')
        print(q_vec.get_shape())
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            output = self.add_answer_module(last_mem, q_vec)

        return output


    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
        config = self.config
        dp = config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) // config.batch_size
        total_loss = []
        accuracy = 0
        
        # shuffle data
        p = np.random.permutation(len(data[0]))
        qp, ip, ql, al, il, im, a, r = data
        qp, ip, ql, al, il, im, a, r = qp[p], ip[p], ql[p], al[p], il[p], im[p], a[p], r[p]

        for step in range(total_steps):
            index = range(step*config.batch_size,(step+1)*config.batch_size)
            feed = {self.question_placeholder: qp[index],
                  self.input_placeholder: ip[index],
                  self.question_len_placeholder: ql[index],
                  self.answer_len_placeholder: al[index],
                  self.input_len_placeholder: il[index],
                  self.answer_placeholder: a[index],
                  self.rel_label_placeholder: r[index],
                  self.dropout_placeholder: dp}
            loss, pred, summary, _ = session.run(
              [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch*total_steps + step)

            # answers = a[step*config.batch_size:(step+1)*config.batch_size]
            answers = a[index]

            accuracy += np.sum(pred == answers)/float(len(answers))


            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()


        if verbose:
            sys.stdout.write('\r')
        return np.mean(total_loss), accuracy/float(total_steps)


    def __init__(self, config):
        self.config = config
        self.variables_to_save = {}
        self.load_data(debug=False)
        self.add_placeholders()
        self.output = self.inference()
        self.pred = self.get_predictions(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_op(self.calculate_loss)
        self.merged = tf.summary.merge_all()

