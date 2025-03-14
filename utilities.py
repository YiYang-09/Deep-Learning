# IMPORTS
import tensorflow as tf
import tf_keras as keras
import tensorflow_probability as tfp

from tf_keras.models import Sequential, Model
from tf_keras.layers import Input, Dense, BatchNormalization, Dropout, Activation
from tf_keras.optimizers import SGD, Adam
from tensorflow_probability.python.layers import DenseVariational

from ray import train

# Set seed from random number generator, for better comparisons
from numpy.random import seed
seed(123)

import matplotlib.pyplot as plt

# CUSTOM PRIOR AND POSTERIOR FUNCTIONS FOR THE VARIATIONAL LAYER
#  Code from https://keras.io/examples/keras_recipes/bayesian_neural_networks/
# The prior is defined as a normal distribution with zero mean and unit variance.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# multivariate Gaussian distribution parametrized by a learnable parameters.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

# =======================================
# DNN related function
# =======================================
# DEEP LEARNING MODEL BUILD FUNCTION
def build_DNN(input_shape, n_hidden_layers, n_hidden_units, loss, act_fun='sigmoid', optimizer:str='sgd', learning_rate=0.01, 
            use_bn=False, use_dropout=False, use_custom_dropout=False, print_summary=False, use_variational_layer=False, kl_weight=None):
    """
    Builds a Deep Neural Network (DNN) model based on the provided parameters.
    
    Parameters:
    input_shape (tuple): Shape of the input data (excluding batch size).
    n_hidden_layers (int): Number of hidden layers in the model.
    n_hidden_units (int): Number of nodes in each hidden layer (here all hidden layers have the same shape).
    loss (keras.losses): Loss function to use in the model.
    act_fun (str, optional): Activation function to use in each layer. Default is 'sigmoid'.
    optimizer (str, optional): Optimizer to use in the model. Default is SGD.
    learning_rate (float, optional): Learning rate for the optimizer. Default is 0.01.
    use_bn (bool, optional): Whether to use Batch Normalization after each layer. Default is False.
    use_dropout (bool, optional): Whether to use Dropout after each layer. Default is False.
    use_custom_dropout (bool, optional): Whether to use a custom Dropout implementation. Default is False.
    
    Returns:
    model (Sequential): Compiled Keras Sequential model.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------      
    # Setup optimizer, depending on input parameter string
    
    if optimizer.lower() == 'sgd':
        optimizer = SGD(learning_rate = learning_rate)
    elif optimizer.lower() == 'adam':
        optimizer = Adam(learning_rate = learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # ============================================

    # Setup a sequential model
    model = Sequential()

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    # Add layers to the model, using the input parameters of the build_DNN function
    
    # Add first (Input) layer, requires input shape
    model.add(Input(shape = input_shape))

    #check if we use batch normalization
    if use_bn:
        model.add(BatchNormalization())
    
    # Add remaining layers. These to not require the input shape since it will be infered during model compile
    for _ in range(n_hidden_layers):
        
        # check if we use_variational_layer
        if use_variational_layer:
            model.add(DenseVariational(units = n_hidden_units, kl_weight = kl_weight, make_prior_fn = prior, make_posterior_fn = posterior))
        else:
            model.add(Dense(units = n_hidden_units))
            
        #check if we use batch normalization
        if use_bn:
            model.add(BatchNormalization())

        # apply activation function
        model.add(Activation(act_fun))

        # check if we use dropout
        if use_dropout:
            model.add(Dropout(0.5))
        if use_custom_dropout:
            model.add(myDropout(0.5))
        

    # Add final layer
    model.add(Dense(units = 1, activation = "sigmoid"))
    
    # Compile model
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

    # ============================================
    # Print model summary if requested
    if print_summary:
        model.summary() 
    
    return model

def train_DNN(config, training_config):
    '''
    Train a DNN model based on the provided configuration and data. 
    This is use in the automatic hyperparameter search and follows the format that Ray Tune expects.

    Parameters:
    config (dict): Dictionary with the configuration parameters for the model. This includes the parameters needed to build the model and can be 
                    manually set or generated by Ray Tune.
                    For convenience, the config dictionary also contains the training parameters, such as the number of epochs and batch size.
    training_config (dict): Dictionary with the training parameters, such as the number of epochs and batch size, and the data to use for training and validation (Xtrain, Ytrain, Xval, Yval).
    '''

    # A dedicated callback function is needed to allow Ray Tune to track the training process
    # This callback will be used to log the loss and accuracy of the model during training
    class TuneReporterCallback(keras.callbacks.Callback):
        """Tune Callback for Keras.
        
        The callback is invoked every epoch.
        """
        def __init__(self, logs={}):
            self.iteration = 0
            super(TuneReporterCallback, self).__init__()
    
        def on_epoch_end(self, batch, logs={}):
            self.iteration += 1
            train.report(dict(keras_info=logs, mean_accuracy=logs.get("accuracy"), mean_loss=logs.get("loss")))
    
    # --------------------------------------------  
    # === Your code here =========================
    # --------------------------------------------
    # Unpack the data tuple
    X_train, y_train, X_val, y_val = training_config["Xtrain"], training_config["Ytrain"], training_config["Xval"], training_config["Yval"]

    # Build the model using the variables stored into the config dictionary.
    # Hint: you provide the config dictionary to the build_DNN function as a keyword argument using the ** operator.
    model = build_DNN(**config)

    #check if class_weight is provided
    if "class_weight" in training_config and training_config["class_weight"] is not None:
        class_weight = training_config["class_weight"]
    else:
        class_weight = None
        
    # Train the model (no need to save the history, as the callback will log the results).
    # Remember to add the TuneReporterCallback() to the list of callbacks.
    history = model.fit(X_train, y_train, batch_size = training_config["batch_size"], epochs = training_config["epochs"], class_weight=class_weight,
                verbose = 2, validation_data = (X_val, y_val), callbacks = TuneReporterCallback())

    return {"history": history}

    # --------------------------------------------


# CUSTOM DROPOUT IMPLEMENTATION
# Code from https://github.com/keras-team/tf-keras/issues/81
class myDropout(keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)  # Override training=True


# CUSTOM PRIOR AND POSTERIOR FUNCTIONS FOR THE VARIATIONAL LAYER
#  Code from https://keras.io/examples/keras_recipes/bayesian_neural_networks/
# The prior is defined as a normal distribution with zero mean and unit variance.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# multivariate Gaussian distribution parametrized by a learnable parameters.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


# =======================================
# PLOTTING FUNCTIONS
# =======================================

# TRAINING CURVES PLOT FUNCTION
def plot_results(history):
    """
    Plots the training and validation loss and accuracy from a Keras history object.
    Parameters:
    history (keras.callbacks.History): A History object returned by the fit method of a Keras model. 
                                       It contains the training and validation loss and accuracy for each epoch.
    Returns:
    None
    """
    
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    
    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training','Validation'])

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training','Validation'])

    plt.show()
