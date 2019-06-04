import os
import codecs
import random
import numpy as np
from utils.utils import word_droput

def read_sentences(file_name, data_dir, token_embedder):
    """
    Reads sentences from a training or testing file and stores them as a list of training examples
    :param file_name: The name of the file e.g. "train.tsv"
    :param data_dir: The directory for the data where the training/test files are
    :param token_embedder: See Token embedder
    :return: A list of BowmanExamples containing the necessary information for training
    """
    sentences = []
    with codecs.open(os.path.join(data_dir, file_name), 'r', encoding='latin1') as f:
        for i, line in enumerate(f):
            try:
                s = line.strip().lower().split(' ')
                if len(s) < token_embedder.max_sentence_length:
                    sentences.append(BowmanExample(s, token_embedder))
            except:
                print("failed read")

    return sentences

class BowmanExample():
    def __init__(self, sentence, token_embedder):
        self.sentence = sentence
        self.sentence_embed = token_embedder.embed_sentence(sentence)
        self.mask = self.create_sentence_mask(len(sentence), token_embedder.max_sentence_length)
        self.seq_length = len(sentence)

    def create_sentence_mask(self, sentence_length, max_sentence_length):
        mask = []
        for i in range(max_sentence_length):
            if i < sentence_length:
                mask.append(1.0)
            else:
                mask.append(0.0)
        return mask


def create_batch(bowman_examples_list, batch_size, token_embedder, dropout_perc, step):
    """
    Creates a batch from a list of training or testing examples - batches are different every step
    :param bowman_examples_list: A list of BowmanExaples
    :param batch_size: size of batch to generate
    :param token_embedder: See TokenEmbedder
    :return: Batch dictionary with useful fields for tensorflow sess
    """

    examples_length = len(bowman_examples_list)
    start_index = step * batch_size % examples_length
    end_index = start_index + batch_size
    if (end_index > examples_length):
        overflow = end_index - examples_length
        indexes = [i for i in range(start_index, examples_length)] + [i for i in range(overflow)]
    else:
        indexes = [i for i in range(start_index, end_index)]

    bowman_examples_batch = [bowman_examples_list[i] for i in indexes]

    batch_data = {}
    batch_data["sentence"] = [example.sentence for example in bowman_examples_batch]
    sentence_embed = [example.sentence_embed for example in bowman_examples_batch]
    batch_data["sentence_embed"] = word_droput(dropout_perc, sentence_embed, token_embedder.embed_word(token_embedder.unk_key), token_embedder.embed_word(token_embedder.eos_key))
    batch_data['target_ohv'] = [token_embedder.gen_indexes_from_sentence(t) for t in batch_data['sentence']]
    batch_data['seq_lengths'] = [example.seq_length for example in bowman_examples_batch]
    batch_data['mask'] = [example.mask for example in bowman_examples_batch]
    return batch_data

def prepare_interact_batch(token_embedder, batch_size, sentences, input_droput):
    sentences = [s.lower().split(' ') for s in sentences]
    while len(sentences) < batch_size:
        sentences.append(sentences[-1])

    bowman_examples = [BowmanExample(s, token_embedder) for s in sentences]
    batch_data = create_batch(bowman_examples, batch_size, token_embedder, input_droput, 0)

    return batch_data


