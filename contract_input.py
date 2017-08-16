from __future__ import division
from __future__ import print_function

import sys

import os as os
import numpy as np

# can be sentence or word
input_mask_mode = "sentence"

# adapted from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
def init(fname):
    
    print("==> Loading test from %s" % fname)
    tasks = []
    task = None
    for i, line in enumerate(open(fname)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": "", "S": ""} 
            counter = 0
            id_map = {}
            
        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        # if not a question
        if line.find('?') == -1:
            task["C"] += line
            id_map[id] = counter
            counter += 1
            
        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            task["S"] = []
            #for num in tmp[2].split():
            #   task["S"].append(id_map[int(num.strip())])
            task["S"] = [0]
            tasks.append(task.copy())

    return tasks


def get_raw_data():
    train_file = "data/contract_data/contracts_train.txt"
    test_file = "data/contract_data/contracts_test.txt"

    train_raw = init(train_file)
    test_raw = init(test_file)

    return train_raw, test_raw

            
def load_glove(dim):
    word2vec = {}
    
    print("==> loading glove")
    with open(("./data/glove/glove.6B/glove.6B." + str(dim) + "d.txt")) as f:
        for line in f:    
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])
            
    print("==> glove is loaded")
    
    return word2vec


def create_vector(word, word2vec, word_vector_size, silent=True):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0,1.0,(word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print("utils.py::create_vector => %s is missing" % word)
    return vector

def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=True):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab: 
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")

def process_input(data_raw, floatX, word2vec, vocab, ivocab, embed_size, split_sentences=False):
    questions = []
    inputs = []
    answers = []
    input_masks = []
    relevant_labels = []
    for x in data_raw:
        if split_sentences:
            inp = x["C"].lower().split(' . ') 
            inp = [w for w in inp if len(w) > 0]
            inp = [i.split() for i in inp]
        else:
            inp = x["C"].lower().split(' ') 
            inp = [w for w in inp if len(w) > 0]

        q = x["Q"].lower().split(' ')
        q = [w for w in q if len(w) > 0]


        a = x["A"].lower().split(' ')
        a = [w for w in a if len(w) > 0]

        if split_sentences:
            inp_vector = [[process_word(word = w, 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab, 
                                        word_vector_size = embed_size, 
                                        to_return = "index") for w in s] for s in inp]
        else:
            inp_vector = [process_word(word = w, 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab, 
                                        word_vector_size = embed_size, 
                                        to_return = "index") for w in inp]
                                    
        q_vector = [process_word(word = w, 
                                    word2vec = word2vec, 
                                    vocab = vocab, 
                                    ivocab = ivocab, 
                                    word_vector_size = embed_size, 
                                    to_return = "index") for w in q]


        a_vector = [process_word(word=w,
                                    word2vec=word2vec,
                                    vocab=vocab,
                                    ivocab=ivocab,
                                    word_vector_size=embed_size,
                                    to_return="index") for w in a]

        
        if split_sentences:
            inputs.append(inp_vector)
        else:
            inputs.append(np.vstack(inp_vector).astype(floatX))
        questions.append(np.vstack(q_vector).astype(floatX))
        answers.append(np.vstack(a_vector).astype(floatX))





        # NOTE: here we assume the answer is one word!
        # Update by Kunling, need to support multiple words

        if not split_sentences:
            if input_mask_mode == 'word':
                input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32)) 
            elif input_mask_mode == 'sentence': 
                input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32)) 
            else:
                raise Exception("invalid input_mask_mode")

        relevant_labels.append(x["S"])
    
    return inputs, questions, answers, input_masks, relevant_labels 

def get_lens(inputs, split_sentences=False):
    lens = np.zeros((len(inputs)), dtype=int)
    for i, t in enumerate(inputs):
        lens[i] = t.shape[0]
    return lens

def get_sentence_lens(inputs):
    lens = np.zeros((len(inputs)), dtype=int)
    sen_lens = []
    max_sen_lens = []
    for i, t in enumerate(inputs):
        sentence_lens = np.zeros((len(t)), dtype=int)
        for j, s in enumerate(t):
            sentence_lens[j] = len(s)
        lens[i] = len(t)
        sen_lens.append(sentence_lens)
        max_sen_lens.append(np.max(sentence_lens))
    return lens, sen_lens, max(max_sen_lens)
    

def pad_inputs(inputs, lens, max_len, mode="", sen_lens=None, max_sen_len=None):
    if mode == "mask":
        padded = [np.pad(inp, (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
        return np.vstack(padded)

    elif mode == "split_sentences":
        padded = np.zeros((len(inputs), max_len, max_sen_len))
        for i, inp in enumerate(inputs):
            padded_sentences = [np.pad(s, (0, max_sen_len - sen_lens[i][j]), 'constant', constant_values=0) for j, s in enumerate(inp)]
            # trim array according to max allowed inputs
            if len(padded_sentences) > max_len:
                padded_sentences = padded_sentences[(len(padded_sentences)-max_len):]
                lens[i] = max_len
            padded_sentences = np.vstack(padded_sentences)
            padded_sentences = np.pad(padded_sentences, ((0, max_len - lens[i]),(0,0)), 'constant', constant_values=0)
            padded[i] = padded_sentences
        return padded

    padded = [np.pad(np.squeeze(inp, axis=1), (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
    return np.vstack(padded)

def create_embedding(word2vec, ivocab, embed_size):
    embedding = np.zeros((len(ivocab), embed_size))
    for i in range(len(ivocab)):
        word = ivocab[i]
        embedding[i] = word2vec[word]
    return embedding

def load_data(config, split_sentences=False):
    vocab = {}
    ivocab = {}

    train_raw, test_raw = get_raw_data()

    if config.word2vec_init:
        assert config.embed_size == 100
        word2vec = load_glove(config.embed_size)
    else:
        word2vec = {}

    # set word at index zero to be end of sentence token so padding with zeros is consistent
    process_word(word = "<eos>", 
                word2vec = word2vec, 
                vocab = vocab, 
                ivocab = ivocab, 
                word_vector_size = config.embed_size, 
                to_return = "index")

    print('==> get train inputs')
    train_data = process_input(train_raw, config.floatX, word2vec, vocab, ivocab, config.embed_size, split_sentences)
    print('==> get test inputs')
    test_data = process_input(test_raw, config.floatX, word2vec, vocab, ivocab, config.embed_size, split_sentences)

    if config.word2vec_init:
        assert config.embed_size == 100
        word_embedding = create_embedding(word2vec, ivocab, config.embed_size)
    else:
        word_embedding = np.random.uniform(-config.embedding_init, config.embedding_init, (len(ivocab), config.embed_size))

    inputs, questions, answers, input_masks, rel_labels = train_data if config.train_mode else test_data

    if split_sentences:
        input_lens, sen_lens, max_sen_len = get_sentence_lens(inputs)
        max_mask_len = max_sen_len
    else:
        input_lens = get_lens(inputs)
        mask_lens = get_lens(input_masks)
        max_mask_len = np.max(mask_lens)

    q_lens = get_lens(questions)
    max_q_len = np.max(q_lens)

    max_input_len = min(np.max(input_lens), config.max_allowed_inputs)

    #pad out arrays to max
    if split_sentences:
        inputs = pad_inputs(inputs, input_lens, max_input_len, "split_sentences", sen_lens, max_sen_len)
        input_masks = np.zeros(len(inputs))
    else:
        inputs = pad_inputs(inputs, input_lens, max_input_len)
        input_masks = pad_inputs(input_masks, mask_lens, max_mask_len, "mask")

    questions = pad_inputs(questions, q_lens, max_q_len)


    a_lens = get_lens(answers)
    max_a_len = np.max(a_lens)
    answers = pad_inputs(answers, a_lens, max_a_len)



    rel_labels = np.zeros((len(rel_labels), len(rel_labels[0])))

    for i, tt in enumerate(rel_labels):
        rel_labels[i] = np.array(tt, dtype=int)

    if config.train_mode:
        num_train = int(len(questions) * 0.8)
        print('num train')
        print(num_train)
        train = questions[:num_train], inputs[:num_train], q_lens[:num_train], a_lens[:num_train], \
                input_lens[:num_train], input_masks[:num_train], answers[:num_train], rel_labels[:num_train]

        valid = questions[num_train:], inputs[num_train:], q_lens[num_train:], a_lens[:num_train], \
                input_lens[num_train:], input_masks[num_train:], answers[num_train:], rel_labels[num_train:]

        return train, valid, word_embedding, max_q_len, max_a_len, max_input_len, max_mask_len, rel_labels.shape[1], len(vocab)

    else:
        test = questions, inputs, q_lens, a_lens, input_lens, input_masks, answers, rel_labels
        return test, word_embedding, max_q_len, max_a_len, max_input_len, max_mask_len, rel_labels.shape[1], len(vocab)


    
