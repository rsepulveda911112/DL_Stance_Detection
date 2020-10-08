import json
import os
import os.path as path


class Config:

    # @classmethod
    # def load_config(cls, conf_path):
    #     with open(conf_path) as f:
    #         conf = json.load(f)
    #         for k, v in conf.items():
    #             setattr(cls, k, v)
    #         cls.make_all_dirs()
    #
    # @classmethod
    # def save_config(cls, conf_path):
    #     obj = {}
    #     for k, v in cls.__dict__.items():
    #         if not isinstance(v, classmethod) and not k.startswith('__'):
    #             obj.update({k: v})
    #     with open(conf_path, 'w') as f:
    #         json.dump(obj, f, indent=4)
    #
    # @classmethod
    # def make_all_dirs(cls):
    #     os.makedirs(cls.model_folder, exist_ok=True)
    #     os.makedirs(cls.ckpt_folder, exist_ok=True)
    #     os.makedirs(cls.submission_folder, exist_ok=True)
    #
    # @classmethod
    # def set_item(cls,item_name,value):
    #     try:
    #         cls.__setattr__(item_name,value)
    #     except:
    #         return


    label_dict=''
    BASE_DIR = os.getcwd()
    SUBMISSION_FILE_NAME = "predictions.jsonl"
    model_name = "esim_0"
    glove_path = path.join(BASE_DIR, "data/glove/glove.6B.300d.txt.gz")
    fasttext_path = path.join(BASE_DIR, "data/fasttext/wiki.en.bin")

    model_folder = path.join(BASE_DIR, "model/%s" % model_name)
    ckpt_folder = path.join(model_folder, 'rte_checkpoints')

    dataset_folder = path.join(os.getcwd(), "data/datasets")
    training_set_file = path.join(dataset_folder, "Emergent/Text Rank Summarizer/train/Emergent_TextRank_Summarizer_5Sentences_train.json")
    test_set_file = path.join(dataset_folder, "Emergent/Text Rank Summarizer/test/Emergent_TextRank_Summarizer_5Sentences_test.json")
    relative_submission_folder="data/submission"
    submission_folder = path.join(BASE_DIR, relative_submission_folder)


    estimator_name = "esim"
    pickle_name = estimator_name + ".p"
    esim_hyper_param = {
        'num_neurons': [
            250,
            180,
            180,
            900,
            550
        ],
        'lr': 0.002,
        'dropout': 0,
        'batch_size': 64,
        'pos_weight': [0.408658712, 1.942468514, 1.540587559],
        'max_checks_no_progress': 10,
        'trainable': False,
        'lstm_layers': 1,
        'optimizer': 'adam',
        'num_epoch': 2,
        'activation': 'relu',
        'initializer': 'he'
    }
    dim_fasttext = 300
    max_sentences = 5
    max_sentence_size = 50
    max_claim_size = max_sentence_size
    seed = 55
    name = 'claim_verification_esim'

    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(submission_folder, exist_ok=True)

    @classmethod
    def update_param(cls, mode, dataset):
        setattr(cls, 'mode', mode)
        setattr(cls, 'dataset', dataset)
        if dataset == "fnc":
            # For Fake news challenge
            setattr(cls, 'label_dict', ['agree', 'disagree', 'discuss'])
        else:
            if dataset == "emergent":
                # For Emergent
                setattr(cls, 'label_dict', ['for', 'against', 'observing'])

