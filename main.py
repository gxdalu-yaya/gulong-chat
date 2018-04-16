#coding=utf-8

import sys
import os
import logging
import numpy as np
import tensorflow as tf

import data_helpers
from model import Seq2seq

logging.basicConfig(
        level = logging.DEBUG,
        handlers = [
            logging.FileHandler("./log/log.txt"),
            logging.StreamHandler()
        ]
    )

tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("traindata_file", "./data/gulong_chat.train", "source data file")
tf.flags.DEFINE_string("testdata_file", "./data/gulong_chat.test", "source data file")

tf.flags.DEFINE_integer("embedding_dim", 90, "Dimensionality of word embedding")
tf.flags.DEFINE_integer("max_sequence_len", 60, "句子最大长度，问句和答案都一样")
tf.flags.DEFINE_integer("batch_size", 128, "Batch size")
tf.flags.DEFINE_integer("hidden_size", 64, "Hidden size")
tf.flags.DEFINE_integer("vocab_size", 23078, "Vocab size")

tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 3, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

train_data = data_helpers.load_data(open(FLAGS.traindata_file, "r").readlines()) 
test_data = data_helpers.load_data(open(FLAGS.testdata_file, "r").readlines())

# Training
logging.info("logging test")
 
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        model = Seq2seq(
            max_sequence_len=FLAGS.max_sequence_len,
            batch_size=FLAGS.batch_size,
            embedding_size=FLAGS.embedding_dim,
            hidden_size=FLAGS.hidden_size,
            vocab_size=FLAGS.vocab_size
        )
        
        saver = tf.train.Saver(tf.global_variables() ,max_to_keep=FLAGS.num_checkpoints)
        sess.run(tf.global_variables_initializer())
        
        train_batches = data_helpers.batch_iter(train_data, FLAGS.batch_size, FLAGS.num_epochs)
        for batch in train_batches:
            encoder_inputs, encoder_inputs_actual_lengths, decoder_outputs, decoder_outputs_actual_lengths = zip(*batch)
            print(np.array(decoder_outputs).shape)
            feed_dict = {
                model.encoder_inputs: encoder_inputs,
                model.encoder_inputs_actual_lengths: encoder_inputs_actual_lengths,
                model.decoder_outputs: decoder_outputs,
                model.decoder_outputs_actual_lengths: decoder_outputs_actual_lengths,
            }
            _, step, loss = sess.run([model.train_op, model.global_step, model.loss], feed_dict=feed_dict)
            logging.info("step: {}, loss: {}".format(step, loss))
            if step % FLAGS.evaluate_every == 0:
                logging.info("\nEvalution:")
                test_data = np.array(test_data)
                test_encoder_inputs, test_encoder_inputs_actual_lengths, test_decoder_outputs, test_decoder_outputs_actual_lengths = zip(*test_data)
                print(np.array(test_encoder_inputs).shape)
                print(np.array(test_decoder_outputs).shape)

                test_feed_dict = {
                    model.encoder_inputs: test_encoder_inputs,
                    model.encoder_inputs_actual_lengths: test_encoder_inputs_actual_lengths,
                    model.decoder_outputs: test_decoder_outputs,
                    model.decoder_outputs_actual_lengths: test_decoder_outputs_actual_lengths,
                }
                test_step, test_loss = sess.run([model.global_step, model.loss], feed_dict=test_feed_dict)
                logging.info("step: {}, test_loss: {}".format(test_step, test_loss))
            if step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, "./model", global_step=step)
                logging.info("Saved model checkpoint to {}\n".format(path))
