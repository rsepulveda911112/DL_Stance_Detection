from typing import List, Dict
import numpy as np
from gensim.models.wrappers import FastText

from tqdm import tqdm

from te.util.text_processing import vocab_map, tokenize, load_whole_glove
from utils.config import Config
from utils.log_helper import LogHelper
from utils.reader import JSONLineReader

dim_fasttext = Config.dim_fasttext

def prediction_2_label(prediction):
    return Config.label_dict[prediction]


def load_corpus_train(path):
    labels = []
    jsonlReader = JSONLineReader()
    values = jsonlReader.read(path)
    h = []
    b = []
    for value in tqdm(values):
        b.append(value['summary'])
        h.append(value['claim'])
        if 'label' in value:
            labels.append(Config.label_dict.index(value['label']))
    datas = {'h': h, 'b':b}
    return datas, labels


def ids_padding_for_multi_sentences_set(sents_list, bodies_size=None, bodies_sent_size=None):
    b_sizes = np.asarray([len(sents) for sents in sents_list])
    bodies_sent_sizes_ = [[len(sent) for sent in sents]
                          for sents in sents_list]
    if bodies_size is None:
        bodies_size = max(b_sizes)
    else:
        b_sizes = np.asarray([size if size < bodies_size else bodies_size for size in b_sizes])
    if bodies_sent_size is None:
        bodies_sent_size = max(map(max, bodies_sent_sizes_))

    def padded_text_ids(_list, sent_sizes_, num_doc, max_num_sent, max_num_words):
        doc_np = np.zeros(
            [num_doc, max_num_sent, max_num_words], dtype=np.int32)
        doc_sent_sizes = np.zeros([num_doc, max_num_sent], dtype=np.int32)
        for i, doc in enumerate(_list):
            for j, sent in enumerate(doc):
                if j >= max_num_sent:
                    break
                doc_sent_sizes[i, j] = sent_sizes_[i][j] if sent_sizes_[i][j] < max_num_words else max_num_words
                for k, word in enumerate(sent):
                    if k >= max_num_words:
                        break
                    doc_np[i, j, k] = word
        return doc_np, doc_sent_sizes

    b_np, b_sent_sizes = padded_text_ids(sents_list, bodies_sent_sizes_, len(sents_list), bodies_size,
                                         bodies_sent_size)
    return b_np, b_sizes, b_sent_sizes


def ids_padding_for_single_sentence_set_given_size(sent_list, max_sent_size=None):
    sent_sizes_ = np.asarray([len(sent) for sent in sent_list])
    if max_sent_size is None:
        max_sent_size = sent_sizes_.max()

    def padded_text_ids(_list, sent_sizes_, num_doc, max_num_words):
        doc_np = np.zeros([num_doc, max_num_words], dtype=np.int32)
        doc_sent_sizes = np.zeros([num_doc], dtype=np.int32)
        for i, doc in enumerate(_list):
            doc_sent_sizes[i] = sent_sizes_[i] if sent_sizes_[
                                                      i] < max_num_words else max_num_words
            for k, word in enumerate(doc):
                if k >= max_num_words:
                    break
                doc_np[i, k] = word
        return doc_np, doc_sent_sizes

    b_np, b_sent_sizes = padded_text_ids(
        sent_list, sent_sizes_, len(sent_list), max_sent_size)
    return b_np, b_sent_sizes


def single_sentence_set_2_ids_given_vocab(texts, vocab_dict):
    logger = LogHelper.get_logger("single_sentence_set_2_ids_given_vocab")
    doc_ids = []
    out_of_vocab_counts = 0
    for sent in tqdm(texts):
        tokens = tokenize(sent)
        word_ids = []
        for token in tokens:
            if token.lower() in vocab_dict:
                word_ids.append(vocab_dict[token.lower()])
            else:
                out_of_vocab_counts += 1
                word_ids.append(vocab_dict['UNK'])
        doc_ids.append(word_ids)
    logger.debug("{} times out of vocab".format(str(out_of_vocab_counts)))
    return doc_ids


def multi_sentence_set_2_ids_given_vocab(texts, vocab_dict):
    logger = LogHelper.get_logger("multi_sentence_set_2_ids_given_vocab")
    doc_ids = []
    out_of_vocab_counts = 0
    for sents in tqdm(texts):
        sent_ids = []
        for sent in sents:
            tokens = tokenize(sent)
            word_ids = []
            for token in tokens:
                if token.lower() in vocab_dict:
                    word_ids.append(vocab_dict[token.lower()])
                else:
                    word_ids.append(vocab_dict['UNK'])
            sent_ids.append(word_ids)
        doc_ids.append(sent_ids)
    logger.debug("{} times out of vocab".format(str(out_of_vocab_counts)))
    return doc_ids


def single_sentence_set_2_fasttext_embedded(sents: List[str], fasttext_model):
    logger = LogHelper.get_logger("single_sentence_set_2_fasttext_embedded")
    if type(fasttext_model) == str:
        fasttext_model = FastText.load_fasttext_format(fasttext_model)
    fasttext_embeddings = []
    for sent in tqdm(sents):
        tokens = tokenize(sent)
        sent_embeddings = []
        for token in tokens:
            try:
                sent_embeddings.append(fasttext_model[token.lower()])
            except KeyError:
                sent_embeddings.append(np.ones([dim_fasttext], np.float32))
        fasttext_embeddings.append(sent_embeddings)
    return fasttext_embeddings, fasttext_model


def multi_sentence_set_2_fasttext_embedded(texts: List[List[str]], fasttext_model):
    logger = LogHelper.get_logger("multi_sentence_set_2_fasttext_embedded")
    fasttext_embeddings = []
    for sents in tqdm(texts):
        text_embeddings = []
        for sent in sents:
            tokens = tokenize(sent)
            sent_embeddings = []
            for token in tokens:
                try:
                    sent_embeddings.append(fasttext_model[token.lower()])
                except KeyError:
                    sent_embeddings.append(np.ones([dim_fasttext], np.float32))
            text_embeddings.append(sent_embeddings)
        fasttext_embeddings.append(text_embeddings)
    return fasttext_embeddings, fasttext_model


def fasttext_padding_for_single_sentence_set_given_size(fasttext_embeddings, max_sent_size=None):
    logger = LogHelper.get_logger("fasttext_padding_for_single_sentence_set_given_size")
    sent_sizes_ = np.asarray([len(sent) for sent in fasttext_embeddings])
    if max_sent_size is None:
        max_sent_size = sent_sizes_.max()

    def padded_text_ids(_list, num_doc, max_num_words):
        doc_np = np.zeros([num_doc, max_num_words, dim_fasttext], dtype=np.float32)
        for i, doc in enumerate(_list):
            for k, word in enumerate(doc):
                if k >= max_num_words:
                    break
                doc_np[i, k] = word
        return doc_np

    ft_np = padded_text_ids(fasttext_embeddings, len(fasttext_embeddings), max_sent_size)
    return ft_np


def fasttext_padding_for_multi_sentences_set(fasttext_embeddings, max_bodies_size=None, max_bodies_sent_size=None):
    logger = LogHelper.get_logger("fasttext_padding_for_multi_sentences_set")
    b_sizes = np.asarray([len(sents) for sents in fasttext_embeddings])
    bodies_sent_sizes_ = [[len(sent) for sent in sents] for sents in fasttext_embeddings]
    if max_bodies_size is None:
        max_bodies_size = max(b_sizes)
    if max_bodies_sent_size is None:
        max_bodies_sent_size = max(map(max, bodies_sent_sizes_))

    def padded_text_ids(_list, num_doc, max_num_sent, max_num_words):
        doc_np = np.zeros([num_doc, max_num_sent, max_num_words, dim_fasttext], dtype=np.float32)
        for i, doc in enumerate(_list):
            for j, sent in enumerate(doc):
                if j >= max_num_sent:
                    break
                for k, word in enumerate(sent):
                    if k >= max_num_words:
                        break
                    doc_np[i, j, k] = word
        return doc_np

    ft_np = padded_text_ids(fasttext_embeddings, len(fasttext_embeddings), max_bodies_size, max_bodies_sent_size)
    return ft_np


def embed_data_set_with_glove_and_fasttext(data_set_path: str,fasttext_model, glove_path: str = None,
                                           vocab_dict: Dict[str, int] = None, glove_embeddings=None,
                                           predicted: bool = True, threshold_b_sent_num=None,
                                           threshold_b_sent_size=50, threshold_h_sent_size=50, is_snopes=False):
    assert vocab_dict is not None and glove_embeddings is not None or glove_path is not None, "Either vocab_dict and glove_embeddings, or glove_path should be not None"
    if vocab_dict is None or glove_embeddings is None:
        vocab, glove_embeddings = load_whole_glove(glove_path)
        vocab_dict = vocab_map(vocab)
    logger = LogHelper.get_logger("embed_data_set_given_vocab")
    datas, labels = load_corpus_train(data_set_path)
    heads_ft_embeddings, fasttext_model = single_sentence_set_2_fasttext_embedded(datas['h'], fasttext_model)
    logger.debug("Finished sentence to FastText embeddings for claims")
    heads_ids = single_sentence_set_2_ids_given_vocab(datas['h'], vocab_dict)
    logger.debug("Finished sentence to IDs for claims")
    bodies_ft_embeddings, fasttext_model = multi_sentence_set_2_fasttext_embedded(datas['b'], fasttext_model)
    logger.debug("Finished sentence to FastText embeddings for evidences")
    bodies_ids = multi_sentence_set_2_ids_given_vocab(datas['b'], vocab_dict)
    logger.debug("Finished sentence to IDs for evidences")
    h_ft_np = fasttext_padding_for_single_sentence_set_given_size(heads_ft_embeddings, threshold_h_sent_size)
    logger.debug("Finished padding FastText embeddings for claims. Shape of h_ft_np: {}".format(str(h_ft_np.shape)))
    b_ft_np = fasttext_padding_for_multi_sentences_set(bodies_ft_embeddings, threshold_b_sent_num,
                                                       threshold_b_sent_size)
    logger.debug("Finished padding FastText embeddings for evidences. Shape of b_ft_np: {}".format(str(b_ft_np.shape)))
    h_np, h_sent_sizes = ids_padding_for_single_sentence_set_given_size(
        heads_ids, threshold_h_sent_size)
    logger.debug("Finished padding claims")
    b_np, b_sizes, b_sent_sizes = ids_padding_for_multi_sentences_set(
        bodies_ids, threshold_b_sent_num, threshold_b_sent_size)
    logger.debug("Finished padding evidences")

    processed_data_set = {'data': {
        'h_np': h_np,
        'b_np': b_np,
        'h_ft_np': h_ft_np,
        'b_ft_np': b_ft_np,
        'h_sent_sizes': h_sent_sizes,
        'b_sent_sizes': b_sent_sizes,
        'b_sizes': b_sizes
    }
    }
    # , 'id': datas['id']
    # if labels is not None and len(labels) == len(processed_data_set['id']):
    if labels is not None:
        processed_data_set['label'] = labels
    return processed_data_set, fasttext_model, vocab_dict, glove_embeddings, threshold_b_sent_num, threshold_b_sent_size
