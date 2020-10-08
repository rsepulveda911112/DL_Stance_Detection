import os
import argparse
import pickle
import numpy as np
import pandas as pd


from te.util.data_reader import embed_data_set_with_glove_and_fasttext, prediction_2_label
from te.util.estimator_definitions import get_estimator
from te.util.text_processing import load_whole_glove, vocab_map
from utils.config import Config
from utils.log_helper import LogHelper
from utils.reader import JSONLineReader
from tqdm import tqdm
from utils.Utils import scorePredict


def save_model(_clf, save_folder, filename, logger):
    """
    Dumps a given classifier to the specific folder with the given name
    """
    _path = os.path.join(save_folder, filename)
    logger.debug("save model to " + _path)
    with open(_path, 'wb') as handle:
        pickle.dump(_clf, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(save_folder, filename):
    """
    Loads and returns a classifier at the given folder with the given name
    """
    _path = os.path.join(save_folder, filename)
    if not os.path.isfile(_path):
        return None
    with open(_path, 'rb') as handle:
        print(handle)
        return pickle.load(handle)


# def generate_submission(_predictions, _ids, test_set_path, submission_path):
#     """
#     Generate submission file for shared task: http://fever.ai/task.html
#     :param _ids:
#     :param _predictions:
#     :param test_set_path:
#     :param submission_path:
#     :return:
#     """
#
#     _predictions_with_id = list(zip(_ids, _predictions))
#     jlr = JSONLineReader()
#     json_lines = jlr.read(test_set_path)
#     os.makedirs(os.path.dirname(os.path.abspath(submission_path)), exist_ok=True)
#     with open(submission_path, 'w') as f:
#         for line in tqdm(json_lines):
#             for i, evidence in enumerate(line['predicted_evidence']):
#                 line['predicted_evidence'][i][0] = ""
#                 # line['predicted_evidence'][i][0] = normalize(evidence[0])
#             _id = line['id']
#             _pred_label = prediction_2_label(2)
#             for _pid, _plabel in _predictions_with_id:
#                 if _pid == _id:
#                     _pred_label = prediction_2_label(_plabel)
#                     break
#             obj = {"id": _id,"predicted_label": _pred_label,"label": line['label'],"setences1": line['sentences1'],"sentence2": line['sentence2']}
#             f.write(json.dumps(obj))
#             f.write('\n')


def generate_submission_test(_predictions, test_set_path, submission_path):
    """
    :param _ids:
    :param _predictions:
    :param test_set_path:
    :param submission_path:
    :return:
    """

    data_temp = {'id': [],'is_Correct': [], 'label': [],'predicted_label': [], 'claim': [],'summary': [] }
    df_temp = pd.DataFrame(data_temp)
    jlr = JSONLineReader()
    json_lines = jlr.read(test_set_path)
    df_test = pd.DataFrame(json_lines, columns=['Id_Article', 'claim', 'summary',
                                                'label'])
    i = 0
    y_predict = []
    for line in tqdm(json_lines):
        _pred_label = ''
        _pred_label = prediction_2_label(_predictions[i])
        y_predict.append(_pred_label)
        value_is_Correct = False
        if _pred_label == line['label']:
            value_is_Correct = True
        df_temp = df_temp.append({"predicted_label": _pred_label,"label": line['label'],
                                    "claim": line['claim'],"summary": line['summary'],
                                  "is_Correct": value_is_Correct},
                                    ignore_index=True)
        i += 1
    labels = Config.label_dict
    scorePredict(y_predict, df_test.label, labels,submission_path)
    writer = pd.ExcelWriter(os.path.join(submission_path,"prediction.xlsx"), engine='xlsxwriter')
    df_temp.to_excel(writer, sheet_name='Sheet1')
    writer.save()


def main(config=None, estimator=None):
    mode = Config.mode
    LogHelper.setup()
    logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0] + "_" + mode)
    logger.info("model: " + mode + ", config: " + str(config))
    logger.info("scorer type: " + Config.estimator_name)
    logger.info("random seed: " + str(Config.seed))
    logger.info("ESIM arguments: " + str(Config.esim_hyper_param))
    fasttext_model = Config.fasttext_path
    if mode == 'train':
        # training mode
        training_set, fasttext_model, vocab, embeddings, _, _ = embed_data_set_with_glove_and_fasttext(
            Config.training_set_file, fasttext_model, glove_path=Config.glove_path,
            threshold_b_sent_num=Config.max_sentences, threshold_b_sent_size=Config.max_sentence_size,
            threshold_h_sent_size=Config.max_claim_size)
        h_sent_sizes = training_set['data']['h_sent_sizes']
        h_sizes = np.ones(len(h_sent_sizes), np.int32)


        training_set['data']['h_sent_sizes'] = np.expand_dims(h_sent_sizes, 1)
        training_set['data']['h_sizes'] = h_sizes
        training_set['data']['h_np'] = np.expand_dims(training_set['data']['h_np'], 1)
        training_set['data']['h_ft_np'] = np.expand_dims(training_set['data']['h_ft_np'], 1)

        X_dict = {
            'X_train': training_set['data'],
            # 'X_valid': valid_set['data'],
            # 'y_valid': valid_set['label'],
            'X_valid': None,
            'y_valid': None,
            'embedding': embeddings
        }

        if estimator is None:
            estimator = get_estimator(Config.estimator_name, Config.ckpt_folder)
        estimator.fit(X_dict, training_set['label'])
        save_model(estimator, Config.model_folder, Config.pickle_name, logger)
    elif mode == 'test':

        restore_param_required = estimator is None
        if estimator is None:
            estimator = load_model(Config.model_folder, Config.pickle_name)
            if estimator is None:
                estimator = get_estimator(Config.estimator_name, Config.ckpt_folder)
        vocab, embeddings = load_whole_glove(Config.glove_path)
        vocab = vocab_map(vocab)
        test_set, _, _, _, _, _ = embed_data_set_with_glove_and_fasttext(Config.test_set_file, fasttext_model, vocab_dict=vocab,
                                                                         glove_embeddings=embeddings,
                                                                         threshold_b_sent_num=Config.max_sentences,
                                                                         threshold_b_sent_size=Config.max_sentence_size,
                                                                         threshold_h_sent_size=Config.max_claim_size)
        del fasttext_model
        h_sent_sizes = test_set['data']['h_sent_sizes']
        h_sizes = np.ones(len(h_sent_sizes), np.int32)
        test_set['data']['h_sent_sizes'] = np.expand_dims(h_sent_sizes, 1)
        test_set['data']['h_sizes'] = h_sizes
        test_set['data']['h_np'] = np.expand_dims(test_set['data']['h_np'], 1)
        test_set['data']['h_ft_np'] = np.expand_dims(test_set['data']['h_ft_np'], 1)
        x_dict = {
            'X_test': test_set['data'],
            'embedding': embeddings
        }
        predictions = estimator.predict(x_dict, restore_param_required)
        # generate_submission(predictions, test_set['id'], Config.test_set_file, Config.submission_file())
        generate_submission_test(predictions, Config.test_set_file, Config.relative_submission_folder)
        # if 'label' in test_set:
        #     print_metrics(test_set['label'], predictions, logger)
    else:
        logger.error("Invalid argument --mode: " + mode + " Argument --mode should be either 'train’ or ’test’")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='\'train\' or \'test\'', default='test')
    parser.add_argument('--dataset', help='\'fnc\' or \'emergent\'', default='emergent')
    args = parser.parse_args()

    Config.update_param(args.mode,args.dataset)
    main()
