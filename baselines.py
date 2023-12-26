import sys
import os
import warnings
from carla import MLModelCatalog, Benchmark
from carla.data.catalog import OnlineCatalog
from carla.recourse_methods import GrowingSpheres, Revise, Wachter, Roar, CCHVAE, Tvae
from carla.models.negative_instances import predict_negative_instances
from carla.data.catalog import CsvCatalog, MyDataset
import carla.evaluation.catalog as evaluation_catalog
from carla.recourse_methods.catalog.tvae.model import gaussian_volume
import torch
import numpy as np
import torch
from torch import nn
import pandas as pd
import tensorflow as tf
from utils import validation
from sklearn.neighbors import NearestNeighbors


def find_k_neighbors(dataset, datapoint, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(dataset)
    distances, indices = nbrs.kneighbors(datapoint)
    return indices


def dnn(input_shape, n_classes=2):
    x = tf.keras.Input(input_shape)
    y = tf.keras.layers.Dense(128)(x)
    y = tf.keras.layers.Activation('relu')(y) 
    y = tf.keras.layers.Dense(128)(y)
    y = tf.keras.layers.Activation('relu')(y) 
    y = tf.keras.layers.Dense(n_classes)(y)
    y = tf.keras.layers.Activation('softmax')(y)
    return tf.keras.models.Model(x, y)

def train_dnn(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    model.evaluate(X_test, y_test, batch_size=batch_size)
    return model


class AnnModel(nn.Module):
    def __init__(self, input_layer, hidden_layers, num_of_classes):
        """
        Defines the structure of the neural network

        Parameters
        ----------
        input_layer: int > 0
            Dimension of the input / number of features
        hidden_layers: list
            List where each element is the number of neurons in the ith hidden layer
        num_of_classes: int > 0
            Dimension of the output / number of classes.
        """
        super().__init__()

        self.input_neurons = input_layer

        # Layer
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(input_layer, hidden_layers[0]))
        # hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        # output layer
        self.layers.append(nn.Linear(hidden_layers[-1], num_of_classes))

        # Activation
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass through the network

        Parameters
        ----------
        x: tabular data
            input

        Returns
        -------
        prediction
        """
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = self.relu(x)
            else:
                x = self.softmax(x)

        return x

    def predict(self, data):
        """
        predict method for CFE-Models which need this method.

        Parameters
        ----------
        data: Union(torch, list)

        Returns
        -------
        np.array with prediction

        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
        else:
            input = torch.squeeze(data)

        return self.forward(input).detach().numpy()



def convert_tf_to_torch(tf_model):
    layer_sizes = [layer.units for layer in tf_model.layers if isinstance(layer, tf.keras.layers.Dense)]
    input_layer = tf_model.input_shape[1]
    hidden_layers = layer_sizes[:-1]
    num_of_classes = layer_sizes[-1]

    # Create the PyTorch model
    torch_model = AnnModel(input_layer, hidden_layers, num_of_classes)

    # Copy weights from the TensorFlow model to the PyTorch model
    for torch_layer, tf_layer in zip(torch_model.layers, [layer for layer in tf_model.layers if isinstance(layer, tf.keras.layers.Dense)]):
        torch_layer.weight.data = torch.tensor(tf_layer.get_weights()[0].T)
        torch_layer.bias.data = torch.tensor(tf_layer.get_weights()[1])

    return torch_model


def baseline(method, dataset_name , tf_model, factuals, hype ):

    dataset = MyDataset(dataset_name, '/dataset/data')

    torch_model= convert_tf_to_torch(tf_model)
    
    ml_model = MLModelCatalog( dataset,  model_type="ann", load_online=False, backend="pytorch", load_torch_model=True, torch_model= torch_model)



    col_num = factuals.shape[1]
    feature_names = ['f{}'.format(i + 1) for i in range(col_num)]
    factuals = pd.DataFrame(factuals, columns=feature_names)



    if method == 'wach':
    
        # wach_hype = {
        #     "feature_cost": "_optional_",
        #     "lr": 0.01,
        #     "lambda_": 0.01,
        #     "n_iter": 1000,
        #     "t_max_min": 0.5,
        #     "norm": 1,
        #     "clamp": True,
        #     "loss_type": "BCE",
        #     "y_target": [0, 1],
        #     "binary_cat_features": True,
        # }

        cf_method = Wachter(ml_model, hype)
        counterfactuals = cf_method.get_counterfactuals(factuals)

    elif method == "roar":


        # roar_hype = {
        #         "feature_cost": "_optional_",
        #         "lr": 0.1,
        #         "lambda_": 0.01,
        #         "delta_max": 0.01,
        #         "norm": 1,
        #         "t_max_min": 0.5,
        #         "loss_type": "BCE",
        #         "y_target": [0, 1],
        #         "binary_cat_features": True,
        #         "loss_threshold": 1e-3,
        #         "discretize": False,
        #         "sample": True,
        #         "lime_seed": 0,
        #         "seed": 0,
        #     }

        cf_method = Roar(ml_model,hype)
        counterfactuals = cf_method.get_counterfactuals(factuals)

    


    elif method == "cchvae":

        ########CCHVAE########
        # cchvae_hype = {
        #     "data_name": dataset.name,
        #     "n_search_samples": 100,
        #     "p_norm": 1,
        #     "step": 0.1,
        #     "max_iter": 1000,
        #     "clamp": True,
        #     "binary_cat_features": False,
        #     "vae_params": {
        #         "layers": [61, 512, 256, 8], #len(ml_model.feature_input_order)
        #         "train": True,
        #         "lambda_reg": 1e-6,
        #         "epochs": 5,
        #         "lr": 1e-3,
        #         "batch_size": 32,
        #     },
        # }

        cf_method = CCHVAE(ml_model, hype)
        counterfactuals = cf_method.get_counterfactuals(factuals)
    


    elif method == "revise":

        # revise_hype = {
        #     "data_name": "None",
        #     "lambda": 0.5,
        #     "optimizer": "adam",
        #     "lr": 0.01,
        #     "max_iter": 1000,
        #     "target_class": [0, 1],
        #     "binary_cat_features": False,
        #     "vae_params": {
        #         "layers": [61,128,8],
        #         "train": True,
        #         "lambda_reg": 1e-6,
        #         "epochs": 5,
        #         "lr": 1e-3,
        #         "batch_size": 32,
        #     }
        #     }

        cf_method = Revise(ml_model, dataset, hype)
        counterfactuals = cf_method.get_counterfactuals(factuals)
        


    elif method == "tvae":

        # tvae_hype = {
        #     "data_name": "None",
        #     "lambda": 0.5,
        #     "tau": 0.85,
        #     "optimizer": "adam",
        #     "lr": 0.01,
        #     "max_iter": 1000,
        #     "target_class": [0, 1],
        #     "binary_cat_features": True,
        #     "vae_params": {
        #         "layers": [61,128,128,128,8],
        #         "train": True,
        #         "lambda_reg": 1e-6,
        #         "epochs": 50
        #         ,
        #         "lr": 1e-3,
        #         "batch_size": 32,
        #     },
        # }

        cf_method = Tvae(ml_model, dataset, hype)
        counterfactuals = cf_method.get_counterfactuals(factuals)


    else:
        raise ValueError("Invalid method")


    print("Method:", cf_method.namee)
    print("Dataset:", dataset.name)

    return counterfactuals.to_numpy() 


def evaluate_counterfactuals(method_name, counterfactuals, factuals, baseline_model, changed_models, norm, sourceFile, clf ):


    original_validity = validation(counterfactuals,
                    np.argmax(baseline_model.predict(counterfactuals), axis=1),
                    baseline_model,
                    affinity_set=[[0], [1]])



    cost=np.linalg.norm(counterfactuals-factuals, ord=norm, axis=1)

    print(f'\n{method_name}\n',file=sourceFile)
    print(f"Cost {method_name}: {np.mean(cost)}",file = sourceFile)
    print(f"Cost {method_name}: {np.mean(cost)}")



    print(f"Baseline Val {method_name}: {original_validity}",file = sourceFile)
    print(f"Baseline Val {method_name}: {original_validity}")

    robust_validity =[]
    
    for i in range(len(changed_models)):
        model_1 = changed_models[i]
        validity_1 = validation(counterfactuals,
                    np.argmax(baseline_model.predict(counterfactuals), axis=1),
                    model_1,
                    affinity_set=[[0], [1]])
        robust_validity.append(validity_1)

    robust_validity = np.array(robust_validity)

    lof = clf.predict(counterfactuals).mean()

    
   
    print(f"LOF {method_name}: {lof}")
    print(f"LOF {method_name}: {lof}",file = sourceFile)

    print(f"Robust Val {method_name}: {100*np.mean(robust_validity)}",file = sourceFile)
    print(f"Robust Val {method_name}: {100*np.mean(robust_validity)}\n")
    
    result_dict = {
        'Method': [method_name],
        'Cost': [np.mean(cost)], 
        'Baseline Validity': [100*original_validity],
        'LOF': [lof], 
        'Robust Validity': [np.mean(100*robust_validity)] 
    }

    result_df = pd.DataFrame(result_dict)


    return result_df