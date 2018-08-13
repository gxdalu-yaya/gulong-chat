#coding=utf-8
import tensorflow as tf
import logging

class Seq2seq:
    def __init__(self, max_sequence_len, embedding_size, hidden_size, vocab_size, sent_sos_id, sent_eos_id, pretrained_embedding_mat=None):
        self.max_sequence_len = max_sequence_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sent_sos_id = sent_sos_id
        self.sent_eos_id = sent_eos_id

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.encoder_inputs = tf.placeholder(tf.int32, [None, max_sequence_len], name="encoder_inputs")
        self.encoder_inputs_actual_lengths = tf.placeholder(tf.int32, [None], name="encoder_inputs_actual_lengths")
        self.decoder_outputs = tf.placeholder(tf.int32, [None, max_sequence_len], name="decoder_outputs")
        self.decoder_outputs_actual_lengths = tf.placeholder(tf.int32, [None], name="decoder_outputs_actual_lengths")
        self.batch_size = tf.placeholder(tf.int32,name="batch_size")
        
        #self.embedding_mat = tf.Variable(pretrained_embedding_mat, name="vocab_W")
        self.embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="vocab_W")

        #batch_size = self.decoder_outputs.get_shape().as_list()[0]
        self.decoder_outputs_no_eos = tf.strided_slice(self.decoder_outputs, [0,0], [self.batch_size, -1], [1,1])
        self.decoder_inputs = tf.concat([tf.fill([self.batch_size, 1], sent_sos_id), self.decoder_outputs_no_eos], 1)
        self.encoder_inputs_embeded = tf.nn.embedding_lookup(self.embedding_mat, self.encoder_inputs)
        self.decoder_inputs_embeded = tf.nn.embedding_lookup(self.embedding_mat, self.decoder_inputs)

        self.encoder_inputs_time_major = tf.transpose(self.encoder_inputs_embeded, perm=[1, 0, 2])
        self.decoder_inputs_time_major = tf.transpose(self.decoder_inputs_embeded, perm=[1, 0, 2])

        self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        self.projection_layer = tf.layers.Dense(vocab_size, use_bias=False)

        logging.debug("encoder_inputs_time_major's shape: {}".format(self.encoder_inputs_time_major.get_shape().as_list()))

        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, self.encoder_inputs_time_major, 
                sequence_length=self.encoder_inputs_actual_lengths, 
                dtype=tf.float32, time_major=True)

        logging.debug("encoder_state's shape: {}".format(self.encoder_state.h.get_shape().as_list()))

    def decoding_layer_train(self):

        
        # TrainingHelper: A helper for use during training.  Only reads inputs.
        #                 Returned sample_ids are the argmax of the RNN output logits.
        helper = tf.contrib.seq2seq.TrainingHelper(
                self.decoder_inputs_time_major, self.decoder_outputs_actual_lengths, time_major=True)


        decoder = tf.contrib.seq2seq.BasicDecoder(
                self.decoder_cell, helper, self.encoder_state,
                output_layer=self.projection_layer)

        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, swap_memory=True)

        sample_id = outputs.sample_id
        logits = outputs.rnn_output

        #logging.debug("decoder_inputs_onehot_timemajor's shape: {}".format(self.decoder_inputs_onehot_timemajor.get_shape().as_list()))
        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(outputs.rnn_output))
        self.decoder_targets_true_length = tf.transpose(self.decoder_outputs, perm=[1, 0])[:decoder_max_steps]
        logging.debug("rnn_outpus's shape: {}".format(logits.get_shape().as_list()))

        #target_weights = tf.sequence_mask(
        #                self.decoder_inputs_actual_lengths, max_sequence_len, dtype=logits.dtype)
        self.mask = tf.to_float(tf.not_equal(self.decoder_targets_true_length, 0))
        self.loss = tf.contrib.seq2seq.sequence_loss(
                            outputs.rnn_output, self.decoder_targets_true_length, weights=self.mask)

        optimizer = tf.train.AdamOptimizer(name="AdamOptimizer")
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
        self.clip_grads, _ = tf.clip_by_global_norm(self.grads, 5)
        self.train_op = optimizer.apply_gradients(zip(self.clip_grads, self.vars), global_step=self.global_step)
        return self.train_op, self.loss

    def decoding_layer_inference(self):
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.embedding_mat,
                tf.fill([batch_size], self.sent_sos_id), self.sent_eos_id)

        decoder = tf.contrib.seq2seq.BasicDecoder(
                self.decoder_cell, helper, self.encoder_state,
                output_layer = self.projection_layer)
                
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=self.max_sequence_len)
        inference_id = outputs.sample_id
