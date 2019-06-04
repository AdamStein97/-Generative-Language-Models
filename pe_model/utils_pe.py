import codecs
import os
import random

def find_insert_delete_words(src_words, trg_words, free_set):
    """
    Given a source sentence and a target sentence finds the set of words to insert and delete to move from the source to the target
    :param src_words: A sentence in the form of list of words
    :param trg_words: A sentence in the form of list of words
    :param free_set: A set of words that are not included as insert words or delete words
    :return insert_words: A list of words which have been inserted
    :return delete_words: A list of words which have been deleted

    E.g.
    Input src_words = ["I", "really", "like", "pizza"]
    Input trg _words = ["pizza", "is", "really", "great"]
    Input free_set = set(["I", "is"])
    Output insert_words = ["great"]
    Output delete_words = ["like"]
    """
    src_set, trg_set = set(src_words), set(trg_words)
    insert_words = list(sorted(trg_set - src_set - free_set))
    delete_words = list(sorted(src_set - trg_set - free_set))

    return insert_words, delete_words

class EditExample():
    """
    An object created to help manage training examples used for batches in an easy format
    """
    def __init__(self, source, target, free_set, token_embedder):
        self.source_sentence = source
        self.source_sentence_embedded = token_embedder.embed_sentence(source)
        self.target_sentence = target
        self.target_sentence_embedded = token_embedder.embed_sentence(target)
        self.insert_words, self.delete_words = find_insert_delete_words(source, target, free_set)
        self.insert_words_embedded = token_embedder.embed_sentence(self.insert_words)[:5]
        self.delete_words_embedded = token_embedder.embed_sentence(self.delete_words)[:5]
        self.mask = self.create_sentence_mask(len(target), token_embedder.max_sentence_length)
        self.seq_length = len(target)

    def create_sentence_mask(self, sentence_length, max_sentence_length):
        mask = []
        for i in range(max_sentence_length):
            if i < sentence_length:
                mask.append(1.0)
            else:
                mask.append(0.0)
        return mask



def examples_from_file(file_name, data_dir, token_embedder, ignore_free_set):
    """
    Returns a list of EditExample objects by reading through a training/testing set that can be placed into batches
    :param file_name: The name of the file e.g. "train.tsv"
    :param data_dir: The directory to reach the data directory where the train/test sets are stored
    :param token_embedder: See Token embedder class for more details
    :param ignore_free_set: Boolean 0: don't include insert/delete words that are in free set 1: do include
    :return: A list of all preprocessed EditExamples from a train/test data file
    """
    examples = []
    if not ignore_free_set:
        with codecs.open(os.path.join(data_dir, 'free.txt'), 'r', encoding='utf-8') as f:
            free = [line.strip().lower() for line in f]
            free_set = set(free)
    else:
        free_set = set()

    with codecs.open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
        for line in f:
            src, trg = line.strip().lower().split('\t')
            src_words = src.split(' ')
            trg_words = trg.split(' ')
            assert len(src_words) > 0
            assert len(trg_words) > 0
            if len(src_words) < token_embedder.max_sentence_length and len(trg_words) < token_embedder.max_sentence_length:
                examples.append(EditExample(src_words, trg_words, free_set, token_embedder))
    random.shuffle(examples)
    return examples

def create_batch(edit_examples_list, batch_size, token_embedder, step):
    """
    Creates a batch to be used for training/testing
    :param edit_examples_list: A list of edit example objects that have been created from a training or testing data file
    :param batch_size: The size of the batch of data to grab
    :param token_embedder: See Token Embedder class
    :param step: Used to calculate start of batch
    :return: a batch dictionary containing various useful fields which will be used as feed_dicts for the tensorflow sess
    """
    examples_length = len(edit_examples_list)
    start_index = step*batch_size % examples_length
    end_index = start_index + batch_size
    if (end_index > examples_length):
        overflow = end_index - examples_length
        indexes = [i for i in range(start_index, examples_length)] + [i for i in range(overflow)]
    else:
        indexes = [i for i in range(start_index, end_index)]

    edit_examples_batch = [edit_examples_list[i] for i in indexes]
    batch_data = {}
    batch_data['sources_plaintext'] = [edit_example.source_sentence for edit_example in edit_examples_batch]
    batch_data['sources'] = [edit_example.source_sentence_embedded for edit_example in edit_examples_batch]
    batch_data['targets_plaintext'] = [edit_example.target_sentence for edit_example in edit_examples_batch]
    batch_data['targets'] = [edit_example.target_sentence_embedded for edit_example in edit_examples_batch]
    batch_data['insert_words'] = [edit_example.insert_words_embedded for edit_example in edit_examples_batch]
    batch_data['delete_words'] = [edit_example.delete_words_embedded for edit_example in edit_examples_batch]
    batch_data['target_ohv'] = [token_embedder.gen_indexes_from_sentence(t) for t in batch_data['targets_plaintext']]
    batch_data['seq_lengths'] = [example.seq_length for example in edit_examples_batch]
    batch_data['mask'] = [example.mask for example in edit_examples_batch]
    return batch_data

def print_list_sentences(sentences_list):
    sentences_joined = [" ".join(sentence) for sentence in sentences_list]
    for s in sentences_joined:
        print(s)


