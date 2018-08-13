#coding=utf-8
import sys
import numpy as np
import jieba

MAX_LEN = 60
VOCAB_SIZE = 23078 


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def load_data(chat_datas, word_index):
    encoder_inputs = list()
    encoder_inputs_actual_lengths = list()
    decoder_outputs = list()
    decoder_outputs_onehot = list()
    decoder_outputs_actual_lengths = list()

    sent_end_id = word_index["</s>"]
    sent_start_id = word_index["<s>"]
    sent_pad_id = word_index["<pad>"]
    sent_unk_id = word_index["<unk>"]

    for line in chat_datas:
        datas = line.strip().split("\t")
        if len(datas) != 2:
            continue
        chat1 = datas[0]
        chat2 = datas[1]
        chat1_seged = jieba.cut(chat1)
        chat2_seged = jieba.cut(chat2)

        chat1_list = list()
        chat2_list = list()

        for word in chat1_seged:
            chat1_list.append(word_index.get(word, sent_unk_id))
        if len(chat1_list) < MAX_LEN:
            chat1_list.append(sent_end_id)
            temp_len = len(chat1_list)
            if temp_len < MAX_LEN:
                chat1_list.extend([sent_pad_id]*(MAX_LEN-temp_len))
        else:
            chat1_list = chat1_list[:MAX_LEN]
            chat1_list[-1] = sent_end_id 

        encoder_inputs.append(chat1_list)
        encoder_inputs_actual_lengths.append(chat1_list.index(sent_end_id))

        for word in chat2_seged:
            chat2_list.append(word_index.get(word, sent_unk_id))
        if len(chat2_list) < MAX_LEN:
            chat2_list.append(sent_end_id)
            temp_len = len(chat2_list)
            if temp_len < MAX_LEN:
                chat2_list.extend([sent_pad_id]*(MAX_LEN-temp_len))
        else:
            chat2_list = chat2_list[:MAX_LEN]
            chat2_list[-1] = sent_end_id
        decoder_outputs.append(chat2_list)
        decoder_outputs_actual_lengths.append(chat2_list.index(sent_end_id))
    return list(zip(encoder_inputs, encoder_inputs_actual_lengths, decoder_outputs, decoder_outputs_actual_lengths))

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size)+1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1)*batch_size, data_size)
            yield shuffled_data[start_index:end_index]
