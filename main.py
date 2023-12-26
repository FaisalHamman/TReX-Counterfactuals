
import os 
import argparse
from recourse_methods import *
from recourse_utils import *
from tqdm import tqdm 
import ast
from email import message
import tensorflow as tf 
import numpy as np
import time
import consistency
from consistency import IterativeSearch
from consistency import StableNeighborSearch, TreXSearch
from utils import load_dataset
from utils import validation
from importlib import reload 
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
from consistency.attack_utils import gaussian_volume
import pickle 
import time
from baselines import *


parser = argparse.ArgumentParser(description="Run counterfactual search on datasets")
parser.add_argument('--dataset', type=str, default='HELOC', choices= ['GermanCredit','TaiwaneseCredit', 'HELOC', 'Adult', 'CTG'], help='Dataset to use')
parser.add_argument('--tau', type=float, default=0.75, help='Tau value for robustness')
parser.add_argument('--sig', type=float, default=0.05, help='Sigma value for Gaussian volume')
parser.add_argument('--max_steps', type=int, default=20, help='Maximum steps for iterative search')
parser.add_argument('--changed_models_size', type=int, default=50, help='Number of changed models')
parser.add_argument('--norm', type=int, default=2, help='Norm type for evaluations (1 or 2)')
args = parser.parse_args()

dataset = args.dataset
tau = args.tau
sig = args.sig
max_steps = args.max_steps
changed_models_size = args.changed_models_size
norm = args.norm


message=f'{dataset} tau={tau} sigma={sig} max_steps={max_steps} norm= {norm}'

(X_train, y_train), (X_test, y_test), n_classes = load_dataset(dataset, path_to_data_dir='/dataset/data')


clf = LocalOutlierFactor(n_neighbors=1,novelty=True)
clf.fit(X_train)

#train baseline model 
baseline_model = dnn(X_train.shape[1:], n_classes=n_classes)
baseline_model = train_dnn(baseline_model, X_train, y_train, X_test, y_test,epochs=50, batch_size=256)



# Creates several changed models for experimentation
changed_models=[]
for i in range(changed_models_size):
    model_new = dnn(X_train.shape[1:], n_classes=n_classes)
    model_new = train_dnn(model_new, X_train, y_train, X_test, y_test, batch_size=256)
    changed_models.append(model_new)


name=dataset+'_results.txt'
sourceFile = open(name, 'a')
print('-' * 70 + '\n', file=sourceFile) 
print(message,file = sourceFile)
print(dataset+'\n', file = sourceFile)

##Save models
baseline_model.save('baseline.h5')
for idx, model in enumerate(changed_models):
    model.save(f'changed_model{idx}.h5')


#Load models
# baseline_model = tf.keras.models.load_model('baseline.h5')
# changed_models = [tf.keras.models.load_model(f'changed_model{idx}.h5') for idx in range(50)]


#find rejected test data points
prop=1  #proportion of rejected data points to find counterfactuals for
y_pred=np.argmax(baseline_model.predict(X_test), axis=1)
X_test_rejected=X_test[(y_test==0)&(y_pred==0)]
y_test_rejected=y_test[(y_test==0)&(y_pred==0)]
y_pred_rejected=y_pred[(y_test==0)&(y_pred==0)]
X_test_rejected= X_test_rejected[:int(prop*len(X_test_rejected))]
y_test_rejected= y_test_rejected[:int(prop*len(y_test_rejected))]
y_pred_rejected= y_pred_rejected[:int(prop*len(y_pred_rejected))]



print(f'L{norm} Experiment')


print("Closest Counterfactual")
L1_iter_search = IterativeSearch(baseline_model,
                                clamp=[X_train.min(), X_train.max()],
                                num_classes=2,
                                eps=0.3,
                                nb_iters=60,
                                eps_iter=0.02,
                                norm=norm,
                                sns_fn=None)
start_time = time.time()                      
l1_cf, pred_cf, is_valid = L1_iter_search(X_test_rejected)


Result_cc=evaluate_counterfactuals(method_name="CC",
                        counterfactuals= l1_cf,
                        factuals= X_test_rejected, 
                        baseline_model=baseline_model, 
                        changed_models=changed_models, 
                        norm= norm,
                        sourceFile= sourceFile, 
                        clf= clf 
                        )
    

print('TreX \n')
trex_fn = TreXSearch(baseline_model,
                tau=tau,
                clamp=[X_train.min(), X_train.max()],
                num_classes=2,
                K=1000,
                sigma=sig,
                sns_eps=1,
                sns_nb_iters=max_steps, 
                sns_eps_iter=0.01,
                n_interpolations=20,
                batch_size=1)

L1_iter_search_trex = IterativeSearch(baseline_model,
                                clamp=[X_train.min(), X_train.max()],
                                num_classes=2,
                                eps=0.3,
                                nb_iters=60,
                                eps_iter=0.02,
                                norm=norm,
                                sns_fn=trex_fn)

start_time = time.time()                                
l1_cf_trex, pred_cf_trex, is_valid_trex = L1_iter_search_trex(X_test_rejected)


Result_trex=evaluate_counterfactuals(method_name="TreX",
                        counterfactuals= l1_cf_trex,
                        factuals= X_test_rejected, 
                        baseline_model=baseline_model, 
                        changed_models=changed_models, 
                        norm= norm,
                        sourceFile= sourceFile, 
                        clf= clf 
                        )

print("SNS \n")
sns_fn = StableNeighborSearch(baseline_model,
                clamp=[X_train.min(), X_train.max()],
                num_classes=2,
                sns_eps=0.2,
                sns_nb_iters=100,  
                sns_eps_iter=0.01,
                n_interpolations=20,
                batch_size=32) 

L1_iter_search_sns = IterativeSearch(baseline_model,
                                clamp=[X_train.min(), X_train.max()],
                                num_classes=2,
                                eps=0.3,
                                nb_iters=60,
                                eps_iter=0.02,
                                norm=norm,
                                sns_fn=sns_fn)

start_time = time.time()
l1_cf_sns, pred_cf_sns, is_valid_sns = L1_iter_search_sns(X_test_rejected)


Result_sns=evaluate_counterfactuals(method_name="SNS",
                        counterfactuals= l1_cf_sns,
                        factuals= X_test_rejected, 
                        baseline_model=baseline_model, 
                        changed_models=changed_models, 
                        norm= norm,
                        sourceFile= sourceFile, 
                        clf= clf 
                        )


print("Wacher \n")
wach_hype = {
        "feature_cost": "_optional_",
        "lr": 0.01,
        "lambda_": 0.01,
        "n_iter": 1000,
        "t_max_min": 0.5,
        "norm": norm,
        "clamp": True,
        "loss_type": "BCE",
        "y_target": [0, 1],
        "binary_cat_features": True,
    }

wach_cfx=baseline(method='wach', dataset_name=dataset , tf_model=baseline_model, factuals=X_test_rejected, hype=wach_hype )

Result_wach = evaluate_counterfactuals(method_name="Wach",
                        counterfactuals= wach_cfx,
                        factuals= X_test_rejected, 
                        baseline_model=baseline_model, 
                        changed_models=changed_models, 
                        norm= norm,
                        sourceFile= sourceFile, 
                        clf= clf 
                        )

print("Roar \n")
roar_hype = {
            "feature_cost": "_optional_",
            "lr": 0.01,
            "lambda_": 0.01,
            "delta_max": 0.001,
            "norm": norm,
            "t_max_min": 0.5,
            "loss_type": "BCE",
            "y_target": [0, 1],
            "binary_cat_features": False,
            "loss_threshold": 1e-3,
            "discretize": False,
            "sample": True,
            "lime_seed": 0,
            "seed": 0,
        }

roar_cfx=baseline(method='roar', dataset_name=dataset , tf_model=baseline_model, factuals=X_test_rejected, hype=roar_hype )

Result_roar=evaluate_counterfactuals(method_name="Roar",
                        counterfactuals= roar_cfx,
                        factuals= X_test_rejected, 
                        baseline_model=baseline_model, 
                        changed_models=changed_models, 
                        norm= norm,
                        sourceFile= sourceFile, 
                        clf= clf 
                        )


print("Cchvae \n")
cchvae_hype = {
        "data_name": "None",
        "n_search_samples": 100,
        "p_norm": norm,
        "step": 0.1,
        "max_iter": 1000,
        "clamp": True,
        "binary_cat_features": False,
        "vae_params": {
            "layers": [39, 512, 256, 8], #len(ml_model.feature_input_order)
            "train": True,
            "lambda_reg": 1e-6,
            "epochs": 5,
            "lr": 1e-3,
            "batch_size": 32,
        },
    }

cchvae_cfx=baseline(method='cchvae', dataset_name=dataset , tf_model=baseline_model, factuals=X_test_rejected, hype=cchvae_hype )

Result_cch= evaluate_counterfactuals(method_name="cchvae",
                        counterfactuals= cchvae_cfx,
                        factuals= X_test_rejected, 
                        baseline_model=baseline_model, 
                        changed_models=changed_models, 
                        norm= norm,
                        sourceFile= sourceFile, 
                        clf= clf 
                        )


print("Revise \n")
revise_hype = {
        "data_name": "None",
        "lambda": 0.5,
        "optimizer": "adam",
        "lr": 0.01,
        "max_iter": 1000,
        "target_class": [0, 1],
        "binary_cat_features": False,
        "vae_params": {
            "layers": [39,128,8],
            "train": True,
            "lambda_reg": 1e-6,
            "epochs": 50,
            "lr": 1e-3,
            "batch_size": 32,
        }
        }

revise_cfx=baseline(method='revise', dataset_name=dataset , tf_model=baseline_model, factuals=X_test_rejected, hype=revise_hype )

Result_revise = evaluate_counterfactuals(method_name="Revise",
                        counterfactuals= revise_cfx,
                        factuals= X_test_rejected, 
                        baseline_model=baseline_model, 
                        changed_models=changed_models, 
                        norm= norm,
                        sourceFile= sourceFile, 
                        clf= clf 
                        )


print("Tvae \n")
tvae_hype = {
        "data_name": "None",
        "lambda": 0.5,
        "tau": tau,
        "optimizer": "adam",
        "lr": 0.01,
        "max_iter": 1000,
        "target_class": [0, 1],
        "binary_cat_features": False,
        "vae_params": {
            "layers": [39,128,8],
            "train": True,
            "lambda_reg": 1e-6,
            "epochs": 50
            ,
            "lr": 0.001,
            "batch_size": 32,
        },
    }

tvae_cfx=baseline(method='tvae', dataset_name=dataset , tf_model=baseline_model, factuals=l1_cf, hype=tvae_hype )

Result_tvae = evaluate_counterfactuals(method_name="Tvae",
                        counterfactuals= tvae_cfx,
                        factuals= X_test_rejected, 
                        baseline_model=baseline_model, 
                        changed_models=changed_models, 
                        norm= norm,
                        sourceFile= sourceFile, 
                        clf= clf 
                        )


print("Nearest Neighbor Counterfactuals \n")
y_pred1=np.argmax(baseline_model.predict(X_train), axis=1)
NN=find_k_neighbors(X_train[y_pred1==1], X_test_rejected, 1)
NN_cf= X_train[y_pred1==1][NN[:,0]]

Result_NN = evaluate_counterfactuals(method_name="NN",
                        counterfactuals= NN_cf,
                        factuals= X_test_rejected, 
                        baseline_model=baseline_model, 
                        changed_models=changed_models, 
                        norm= norm,
                        sourceFile= sourceFile, 
                        clf= clf 
                        )


print("TreX Robust Nearest Neighbor Counterfactual \n")
kk=X_train[y_pred1==1].shape[0]-1
NN=find_k_neighbors(X_train[y_pred1==1], X_test_rejected, kk)

RNN_cf=[]
for idx in range(NN.shape[0]):
    for idy in range(NN.shape[1]):
        obj= gaussian_volume(baseline_model, np.expand_dims(X_train[y_pred1==1][NN[idx,idy]], axis=0), y=0, K=1000, sigma=sig, num_class=2)
        if obj> tau:
           RNN_cf.append(X_train[y_pred1==1][NN[idx,idy]])
           break
    else:
        RNN_cf.append(X_train[y_pred1==1][NN[idx,0]])

RNN_cf=np.array(RNN_cf)

Result_TrexNN = evaluate_counterfactuals(method_name="TrexNN",
                        counterfactuals= RNN_cf,
                        factuals= X_test_rejected, 
                        baseline_model=baseline_model, 
                        changed_models=changed_models, 
                        norm= norm,
                        sourceFile= sourceFile, 
                        clf= clf 
                        )


# Return Results in Table

result_dfs = []
locals=locals().copy().items()
# Iterate through local namespace objects
for key, value in locals:
    if key.startswith('Result_') and isinstance(value, pd.DataFrame):
        result_dfs.append(value)


Result_pd = pd.concat(result_dfs, ignore_index=True)

print(Result_pd)
print(Result_pd,file=sourceFile)

sourceFile.close()