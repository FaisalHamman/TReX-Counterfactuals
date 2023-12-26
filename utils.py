import numpy as np
import tensorflow as tf
import pandas as pd
import datalib

def load_dataset(data, path_to_data_dir='dataset/data'):
    if data == 'Adult':
        X_train = np.load(
            path_to_data_dir+'/adult_X_train_norm.npy',
            allow_pickle=False).astype('float32')
        y_train = np.load(
            path_to_data_dir+'/adult_y_train_norm.npy',
            allow_pickle=False).astype('float32')
        X_test = np.load(
            path_to_data_dir+'/adult_X_test_norm.npy',
            allow_pickle=False).astype('float32')
        y_test = np.load(
            path_to_data_dir+'/adult_y_test_norm.npy',
            allow_pickle=False).astype('float32')
        n_classes = 2


    elif data == 'GermanCredit':
        X_train = np.load(
            path_to_data_dir+'/german_X_train.npy',
            allow_pickle=False).astype('float32')
        y_train = np.load(
            path_to_data_dir+'/german_y_train.npy',
            allow_pickle=False).astype('float32')
        X_test = np.load(
            path_to_data_dir+'/german_X_test.npy',
            allow_pickle=False).astype('float32')
        y_test = np.load(
            path_to_data_dir+'/german_y_test.npy',
            allow_pickle=False).astype('float32')
        n_classes = 2



    elif data == 'Seizure':
        X_train = np.load(
            path_to_data_dir+'/seizure_X_train.npy',
            allow_pickle=False).astype('float32')
        y_train = np.load(
            path_to_data_dir+'/seizure_y_train.npy',
            allow_pickle=False).astype('float32')
        X_test = np.load(
            path_to_data_dir+'/seizure_X_test.npy',
            allow_pickle=False).astype('float32')
        y_test = np.load(
            path_to_data_dir+'/seizure_y_test.npy',
            allow_pickle=False).astype('float32')
        n_classes = 2


    elif data == 'TaiwaneseCredit':
        X_train = np.load(
            path_to_data_dir+'/X_train_taiwanese.npy',
            allow_pickle=False).astype('float32')
        y_train = np.load(
            path_to_data_dir+'/y_train_taiwanese.npy',
            allow_pickle=False).astype('float32')
        X_test = np.load(
            path_to_data_dir+'/X_test_taiwanese.npy',
            allow_pickle=False).astype('float32')
        y_test = np.load(
            path_to_data_dir+'/y_test_taiwanese.npy',
            allow_pickle=False).astype('float32')
        n_classes = 2

    elif data == 'Warafin':
        X_train = np.load(
            path_to_data_dir+'/X_train_warafin.npy',
            allow_pickle=False).astype('float32')
        y_train = np.load(
            path_to_data_dir+'/y_train_warafin.npy',
            allow_pickle=False).astype('float32')
        X_test = np.load(
            path_to_data_dir+'/X_test_warafin.npy',
            allow_pickle=False).astype('float32')
        y_test = np.load(
            path_to_data_dir+'/y_test_warafin.npy',
            allow_pickle=False).astype('float32')
        n_classes = 3


    elif data == 'HELOC':
        X_train = np.load(
            path_to_data_dir+'/X_train_heloc.npy',
            allow_pickle=False).astype('float32')
        y_train = np.load(
            path_to_data_dir+'/y_train_heloc.npy',
            allow_pickle=False).astype('float32')
        X_test = np.load(
            path_to_data_dir+'/X_test_heloc.npy',
            allow_pickle=False).astype('float32')
        y_test = np.load(
            path_to_data_dir+'/y_test_heloc.npy',
            allow_pickle=False).astype('float32')
        n_classes = 2


    elif data == 'CTG':
        ctg=pd.read_csv('/dataset/data/ctg_data.csv')
        ctg_data=datalib.CustomData("ctg", ctg.values[:,0:21], (ctg.values[:,21]>1).astype('int'), 
                             processors=["normalize"], split=datalib.splitters.Split(tr=4, te=1, seed=0))
        X_train = ctg_data.x_tr.astype('float32')
        y_train = ctg_data.y_tr.astype('float32')
        X_test = ctg_data.x_te.astype('float32')
        y_test = ctg_data.y_te.astype('float32')
        n_classes = 2


    return (X_train, y_train), (X_test, y_test), n_classes


def batch_flatten(x):
    return np.reshape(x, (x.shape[0], -1))


def convert_to_super_labels(preds, affinity_set):
    for subset in affinity_set:
        for l in subset:
            preds[preds == l] = subset[0]
    return preds

def validation(counterfactuals,
                 modelA_counterfual_preds,
                 modelB,
                 modelA_pred=None,
                 batch_size=32,
                 affinity_set=[[0], [1, 2]]):

    if modelA_pred is None:
        modelA_pred = 1 - modelA_counterfual_preds

    modelB_counterfactual_probits = modelB.predict(counterfactuals,
                                                   batch_size=batch_size)

    is_bianary = modelB_counterfactual_probits.shape[1] <= 2

    modelB_counterfactual_pred = np.argmax(modelB_counterfactual_probits,
                                           axis=-1)
    
    if is_bianary:
        validation = np.mean(
            modelA_pred != modelB_counterfactual_pred)
    else:
        modelA_super_labels = convert_to_super_labels(modelA_pred.copy(), affinity_set)
        modelB_counterfactual_super_labels = convert_to_super_labels(modelB_counterfactual_pred.copy(), affinity_set)

        validation = np.mean(
            modelA_super_labels != modelB_counterfactual_super_labels)


    return validation