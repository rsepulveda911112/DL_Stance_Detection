import numpy as np



def  get_estimator(scorer_type, save_folder=None):
    if scorer_type == 'esim':
        # submitted model, glove + fasttext, with attention
        from os import path
        from te.deep_models.ESIM_for_ensemble import ESIM
        from utils.config import Config
        pos_weight = np.asarray(Config.esim_hyper_param['pos_weight'], np.float32)
        clf = ESIM(random_state=Config.seed, tensorboard_logdir="logdir/", learning_rate=Config.esim_hyper_param['lr'],
                   max_check_without_progress=Config.esim_hyper_param['max_checks_no_progress'],
                   activation=Config.esim_hyper_param['activation'],
                   initializer=Config.esim_hyper_param['initializer'],
                   num_epoch=Config.esim_hyper_param['num_epoch'],
                   lstm_layers=Config.esim_hyper_param['lstm_layers'],
                   optimizer=Config.esim_hyper_param['optimizer'],
                   trainable=Config.esim_hyper_param['trainable'],
                   batch_size=Config.esim_hyper_param['batch_size'],
                   dropout_rate=Config.esim_hyper_param['dropout'],
                   num_neurons=Config.esim_hyper_param['num_neurons'], pos_weight=pos_weight,
                   ckpt_path=path.join(save_folder, Config.name + '.ckpt'), name=Config.name)
    return clf
