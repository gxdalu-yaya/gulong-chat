#coding=utf-8
import tensorflow as tf
import logging

class Seq2seq:
    def __init__(self, max_sequence_len, batch_size, embedding_size, hidden_size, vocab_size, pretrained_embedding_mat=None):
        self.max_sequence_len = max_sequence_len
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.encoder_inputs = tf.placeholder(tf.int32, [batch_size, max_sequence_len], name="encoder_inputs")
        self.encoder_inputs_actual_lengths = tf.placeholder(tf.int32, [batch_size], name="encoder_inputs_actual_lengths")
        self.decoder_outputs = tf.placeholder(tf.int32, [batch_size, max_sequence_len], name="decoder_outputs")
        self.decoder_outputs_actual_lengths = tf.placeholder(tf.int32, [batch_size], name="decoder_outputs_actual_lengths")
        
        #self.embedding_mat = tf.Variable(pretrained_embedding_mat, name="vocab_W")
        self.embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="vocab_W")

        self.encoder_inputs_embeded = tf.nn.embedding_lookup(self.embedding_mat, self.encoder_inputs)
        self.decoder_outputs_embeded = tf.nn.embedding_lookup(self.embedding_mat, self.decoder_outputs)

        self.encoder_inputs_time_major = tf.transpose(self.encoder_inputs_embeded, perm=[1, 0, 2])
        self.decoder_outputs_time_major = tf.transpose(self.decoder_outputs_embeded, perm=[1, 0, 2])

        logging.debug("encoder_inputs_time_major's shape: {}".format(self.encoder_inputs_time_major.get_shape().as_list()))

        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, self.encoder_inputs_time_major, 
                sequence_length=self.encoder_inputs_actual_lengths, 
                dtype=tf.float32, time_major=True)

        logging.debug("encoder_state's shape: {}".format(encoder_state.h.get_shape().as_list()))

        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        
        # TrainingHelper: A helper for use during training.  Only reads inputs.
        #                 Returned sample_ids are the argmax of the RNN output logits.
        helper = tf.contrib.seq2seq.TrainingHelper(
                self.decoder_outputs_time_major, self.decoder_outputs_actual_lengths, time_major=True)

        projection_layer = tf.layers.Dense(vocab_size, use_bias=False)

        decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, helper, encoder_state,
                output_layer=projection_layer)

        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.decoder_outputs, logits=logits)

        target_weights = tf.sequence_mask(
                        self.decoder_outputs_actual_lengths, max_sequence_len, dtype=logits.dtype)
        self.loss = tf.reduce_sum(crossent*target_weights) / tf.to_float(self.batch_size) 

        optimizer = tf.train.AdamOptimizer(name="AdamOptimizer")
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
        self.clip_grads, _ = tf.clip_by_global_norm(self.grads, 5)
        self.train_op = optimizer.apply_gradients(zip(self.clip_grads, self.vars), global_step=self.global_step)


