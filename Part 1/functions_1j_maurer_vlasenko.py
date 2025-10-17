
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

# Activation Functions

# Sigmoid
def sigmoid(x):
    """
    Sigmoid activation function with overflow avoidance
    Inputs:
        x: Input array
    Outputs:
        Sigmoid, 1 / (1+exp(-x)) (clipped)
    """
    # Clip values into the -500,500 range to prevent overflow in exp function
    x_clipped = np.clip(x,-500,500)
    return 1.0/(1.0 + np.exp(-x_clipped))

# Derivative of sigmoid
def sigmoid_derivative(x):
    """
    Derivative of sigmoid function
    Inputs:
        x: Input array (pre-activation values)
    Outputs:
        Derivative: sigmoid(x) * (1-sigmoid(x))
    """
    s = sigmoid(x)
    return s * (1-s)

# We can use np.tanh of the hyperbolic tangent activation function

# Derivative of tanh
def tanh_derivative(x):
    """
    Derivative of tanh function
    Inputs:
        x: Input array (pre-activation values)
    Outputs:
        Derivative: 1-tanh^2(x)
    """
    return 1-np.tanh(x)**2

# ReLU activation function
def relu(x):
    """
    ReLU activation function
    Inputs:
        x: Input array
    Outputs:
        ReLU output: max(0, x)
    """
    return np.maximum(0, x)

# Derivative of ReLU
def relu_derivative(x):
    """
    Derivative of ReLU function
    Inputs:
        x: Input array (pre-activation values)
    Outputs:
        Derivative: 1 if x>0, otherwise 0
    """
    return (x > 0).astype(float)

# Function to retireve the activation function and its derivative
def get_activation_functions(activation_name):
    """
    Get activation function and its derivative based on name
    Inputs:
        activation_name: String name of activation function
    Outputs:
        tuple: (activation_function, derivative_function)
    """
    if activation_name == 'sigmoid':
        return sigmoid, sigmoid_derivative
    elif activation_name == 'tanh':
        return np.tanh, tanh_derivative
    elif activation_name == 'relu':
        return relu, relu_derivative
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")

# Function that initializes the neural network
def initialize_network(layer_sizes):
    """
    Initialize neural network weights and biases using Glorot
    Inputs:
        layer_sizes: List of integers representing the size of each layer (input, hidden layers, output) -> length should be #layers+1
    Outputs:
        tuple: (weights, biases) where weights are a list of weight matrices for each layer
               and biases a list of bias vectors for each layer
    """
    weights = []
    biases = []

    # Initialize weights and biases for each layer connection
    for i in range(len(layer_sizes)-1):
        in_size = layer_sizes[i] # number of input neurons
        out_size = layer_sizes[i + 1] # number of output neurons

        # Glorot initialization
        border = np.sqrt(6.0/(in_size + out_size))

        # Initialize weights uniformly in [-border, border]
        W = np.random.uniform(-border, border, (out_size, in_size))

        # Initial biases are 0
        b = np.zeros((out_size,1))

        # Add them to the matrix and bias lists
        weights.append(W)
        biases.append(b)

    return weights, biases

# Forward pass function
def forward(X, weights, biases, activation_fun):
    """
    Perform forward propagation through the network
    Inputs:
        X: input data
        weights: list of weight matrices
        biases: list of bias vectors
        activation_fun: activation function to use for hidden layers
    Outputs:
        tuple: (final_output, post_activations, pre_activations) where
           final_output: output of network
           post_activations: list of activations for each layer (including input)
           pre_activations: list of pre-activation values for each layer
    """
    # make sure the input is in correct shape (n_features, n_samples)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.shape[0] != len(weights[0][0]): # if features are not in rows
        X = X.T

    # Store post-activations and pre-activation values for backpropagation
    post_activations=[X]  # first "Post-activation" is the input
    pre_activations=[]      # pre-activation values

    present_input = X

    # Forward pass through all hidden layers
    num_layers = len(weights)
    for i in range(num_layers-1):  # All layers except output
        # Linear transformation: z = Wx+b
        z = np.dot(weights[i], present_input) + biases[i]
        pre_activations.append(z)

        # Apply activation function
        a = activation_fun(z)
        post_activations.append(a)
        present_input = a

    # Output layer (linear activation for regression)
    mlp_output = np.dot(weights[-1], present_input)+biases[-1]
    pre_activations.append(mlp_output)
    post_activations.append(mlp_output)

    return post_activations[-1], post_activations, pre_activations

# Loss Function
def loss(y_true, y_pred, weights, lambda_reg):
    """
    Compute L2 regularized Mean Squared Error loss
    Inputs:
        y_true: true target values
        y_pred: predicted values
        weights: list of weight matrices (for regularization)
        lambda_reg: L2 regularization parameter
    Outputs:
        tuple: (total_loss, mse_loss, reg_loss)
    """
    # Mean Squared Error loss
    mse_loss = np.mean((y_true-y_pred) ** 2)

    # L2 regularization term: sum of squared weights
    reg_loss = 0
    for W in weights:
        reg_loss += np.sum(W**2)

    # Total loss = MSE + lambda * ridge_loss
    total_loss = mse_loss + lambda_reg*reg_loss

    return total_loss, mse_loss, reg_loss


# Function that computes the backward pass
def backward(X, y_true, weights, post_activations, pre_activations,
                        activation_derivative, lambda_reg):
    """
    Perform backward propagation to compute gradients
    Inputs:
        X: input data
        y_true: true target values
        weights: list of weight matrices
        post_activations: list of post-activations from forward pass
        pre_activations: list of pre-activation values from forward pass
        activation_derivative: derivative of activation function
        lambda_reg: L2 regularization parameter
    Outputs:
        tuple: (weight_gradients, bias_gradients)
    """
    # Ensure that the shapes are correct
    if X.ndim==1:
        X = X.reshape(-1,1)
    elif X.shape[0]!=len(weights[0][0]):
        X = X.T

    if y_true.ndim==1:
        y_true = y_true.reshape(1, -1)
    elif y_true.shape[0]!=1:
        y_true = y_true.T

    n_samples = X.shape[1]

    weight_gradients = []
    bias_gradients = []

    # Start backpropagation from output layer
    # For regression with MSE loss: dL/dy_pred = 2/n_samples * (y_pred-y_true)
    y_pred = post_activations[-1]
    delta = (2.0/n_samples) * (y_pred-y_true)

    # Gradients for output layer
    # dL/dW = delta * previous_activation^T + regularization_term
    dW_output = np.dot(delta, post_activations[-2].T) + 2 * lambda_reg * weights[-1]
    db_output = np.sum(delta, axis=1, keepdims=True)

    weight_gradients.append(dW_output)
    bias_gradients.append(db_output)

    # Backward propagation through hidden layers
    num_layers = len(weights)
    for i in range(num_layers-2,-1,-1):  # From second-to-last to first layer
        # Propagate error backward: delta = W^T * delta_next * activation_derivative
        delta = np.dot(weights[i+1].T, delta) * activation_derivative(pre_activations[i])

        # Compute gradients
        if i==0: # first hidden layer connects to input
            dW = np.dot(delta, X.T) + 2 * lambda_reg * weights[i]
        else: # other hidden layers connect to previous activations
            dW = np.dot(delta, post_activations[i].T) + 2 * lambda_reg * weights[i]

        db = np.sum(delta, axis=1, keepdims=True)

        weight_gradients.append(dW)
        bias_gradients.append(db)

    # reverse to match forward order (first layer first)
    weight_gradients.reverse()
    bias_gradients.reverse()

    return weight_gradients, bias_gradients

# Flattens weights and biases for optimization
def flatten_parameters(weights, biases):
    """
    Flatten all network parameters into a single vector for optimization
    Inputs:
        weights: list of weight matrices
        biases: list of bias vectors
    Outputs:
        np.array: flattened parameter vector
    """
    parameters = [] # initialize parameter vector
    for W, b in zip(weights, biases):
        parameters.extend(W.flatten()) # flatten weight matrix
        parameters.extend(b.flatten()) # flatten bias vector
    return np.array(parameters)


def unflatten_parameters(parameters, layer_sizes):
    """
    Restore network parameters from flattened vector
    Inputs:
        parameters: Flattened parameter vector
        layer_sizes: List of layer sizes to determine shapes
    Outputs:
        tuple: (weights, biases) restored to original structure
    """
    # intializations
    idx = 0
    weights = []
    biases = []

    # Reconstruct each layer's parameters
    for i in range(len(layer_sizes)-1):
        # calculate sizes
        w_size = layer_sizes[i+1]*layer_sizes[i] # weight matrix size
        b_size = layer_sizes[i+1] # bias vector size
        # Extract and reshape weights
        W = parameters[idx:idx + w_size].reshape(layer_sizes[i+1], layer_sizes[i])
        idx += w_size
        # Extract and reshape biases
        b = parameters[idx:idx + b_size].reshape(layer_sizes[i+1], 1)
        idx += b_size
        # Append weight and bias matrix and vector
        weights.append(W)
        biases.append(b)

    return weights, biases


# Objective function (loss)
def objective_function(parameters, X, y, layer_sizes, activation_fun, lambda_reg):
    """
    Objective function for scipy.optimize.minimize

    Inputs:
        parameters: flattened parameter vector
        X: input data
        y: target values
        layer_sizes: network architecture
        activation_fun: activation function
        lambda_reg: regularization parameter

    Outputs:
        float: loss value to minimize
    """
    # Restore network structure
    weights, biases = unflatten_parameters(parameters, layer_sizes)
    # Forward pass
    y_pred, activations, z_values = forward(X, weights, biases, activation_fun)
    # Compute and return loss
    total_loss, _, _ = loss(y, y_pred.flatten(), weights, lambda_reg)

    return total_loss

# Gradient of the loss function
def gradient_function(params, X, y, layer_sizes, activation_fun, activation_derivative, lambda_reg):
    """
    Gradient function for scipy.optimize.minimize
    Inputs:
        params: flattened parameter vector
        X: input data
        y: target values
        layer_sizes: network architecture (layer dimensions)
        activation_fun: activation function
        activation_derivative: derivative of activation function
        lambda_reg: regularization parameter
    Outputs:
        np.array: Flattened gradient vector
    """
    # Restore network structure
    weights, biases = unflatten_parameters(params, layer_sizes)

    # Forward pass
    y_pred, post_activations, pre_activations = forward(X,weights,biases,activation_fun)

    # Backward pass
    weight_gradients, bias_gradients = backward(X,y,weights,post_activations,pre_activations,activation_derivative,lambda_reg)

    # Flatten gradients for optimizer
    grad = []
    for dW, db in zip(weight_gradients, bias_gradients):
        grad.extend(dW.flatten())
        grad.extend(db.flatten())

    return np.array(grad)


# Function that performs the training on the neural network
def train_network(X, y, layer_sizes, activation='tanh', lambda_reg=0.01,
                 method='L-BFGS-B', maxiter=1000, seed=42):
    """
    Train a neural network using scipy.optimize
    Inputs:
        X: input features, shape (n_samples, n_features)
        y: target values, shape (n_samples,)
        layer_sizes: list of layer sizes [input, hidden1, hidden2, ..., output]
        activation: activation function name, must be in tanh, sigmoid or relu
        lambda_reg: L2 regularization parameter
        method: optimization method for scipy.optimize.minimize
        maxiter: maximum number of iterations
    Outputs:
        tuple: (weights, biases, optimization_result, initial_loss, final_loss)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    print(f"Training network with architecture: {layer_sizes}")
    print(f"Activation: {activation}, Lambda: {lambda_reg}")

    # Get activation functions
    activation_fun, activation_derivative = get_activation_functions(activation)

    # Initialize network parameters
    weights, biases = initialize_network(layer_sizes)

    # Compute initial loss
    y_pred_initial, _, _ = forward(X, weights, biases, activation_fun)
    initial_loss, _, _ = loss(y, y_pred_initial.flatten(), weights, lambda_reg)
    print(f"Initial loss: {initial_loss:.6f}")

    # Flatten parameters for optimization
    initial_parameters = flatten_parameters(weights, biases)

    # Optimize using scipy.optimize.minimize
    result = minimize(
        fun=objective_function,
        x0=initial_parameters,
        args=(X, y, layer_sizes, activation_fun, lambda_reg),
        method=method,
        jac= lambda parameters, X, y, layer_sizes, activation_fun, lambda_reg: gradient_function(parameters, X, y, layer_sizes, activation_fun, activation_derivative, lambda_reg),
        options={'maxiter': maxiter, 'disp': True}
    )

    # Get final parameters
    final_weights, final_biases = unflatten_parameters(result.x, layer_sizes)

    # Compute final loss
    y_pred_final, _, _ = forward(X, final_weights, final_biases, activation_fun)
    final_loss, _, _ = loss(y, y_pred_final.flatten(), final_weights, lambda_reg)

    print(f"Final loss: {final_loss:.6f}")
    print(f"Optimization successful: {result.success}")
    print(f"Number of iterations: {result.nit}")

    return final_weights, final_biases, result, initial_loss, final_loss


# Function that can be used to predict values using the MLP (e.g. for testing or computing MAPE)
def predict(X, weights, biases, activation='tanh'):
    """
    Make predictions using trained network
    Inputs:
        X: input data
        weights: trained weight matrices
        biases: trained bias vectors
        activation: activation function name
    Outputs:
        np.array: predictions, shape
    """
    activation_fun, _ = get_activation_functions(activation)
    y_pred,_,_ = forward(X, weights, biases, activation_fun)
    return y_pred.flatten()

# MAPE function
def MAPE(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error
    Inputs:
        y_true: true values
        y_pred: predicted values
    Outputs:
        float: MAPE
    """
    # Avoid division by zero
    # Only commpute MAPE on the elements != 0
    condition = (y_true != 0)
    if np.sum(condition) == 0:
        return np.inf

    return 100 * np.mean(np.abs((y_true[condition]-y_pred[condition])/y_true[condition]))

# given layer_sizes, activation, lambda_reg, k, and a method, perform k-fold cross-validation
def k_fold_cross_validation(X, y, layer_sizes, activation='tanh', lambda_reg=0.01,
                           k=5, method='L-BFGS-B'):
    """
    Perform k-fold cross-validation to evaluate model performance
    Inputs:
        X: input features
        y: target values
        layer_sizes: network architecture
        activation: activation function name
        lambda_reg: regularization parameter
        k: number of folds
        method: optimization method
    Outputs:
        tuple: (avg_train_loss, avg_val_loss, avg_train_mape, avg_val_mape)
    """
    print(f"\nStarting {k}-fold cross-validation...")

    # create KFold object
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Initialize training and validation losses/ MAPEs
    train_losses = []
    val_losses = []
    train_mapes = []
    val_mapes = []

    fold = 1
    for train_idx, val_idx in kf.split(X):
        print(f"\nFold {fold}/{k}")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        weights, biases, result, _, _ = train_network(X_train, y_train, layer_sizes, activation, lambda_reg, method, maxiter=1000)

        # Make predictions
        y_train_pred = predict(X_train, weights, biases, activation)
        y_val_pred = predict(X_val, weights, biases, activation)

        # compute metrics (MSE and MAPE)
        train_loss = mean_squared_error(y_train, y_train_pred)
        val_loss = mean_squared_error(y_val, y_val_pred)
        train_mape = MAPE(y_train, y_train_pred)
        val_mape = MAPE(y_val, y_val_pred)

        # Store results
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mapes.append(train_mape)
        val_mapes.append(val_mape)

        # Print training losses and MAPE
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train MAPE: {train_mape:.4f}%, Val MAPE: {val_mape:.4f}%")

        fold+=1

    # Return average metrics
    return (np.mean(train_losses), np.mean(val_losses),
            np.mean(train_mapes), np.mean(val_mapes))


# Search for the best hyperparameter configuration
def hyperparameter_search(X_train, y_train):
    """
    Search for best hyperparameters using grid search with cross-validation
    Inputs:
        X_train: training features
        y_train: training targets
    Outputs:
        tuple: (best_parameters, all_results)
    """
    print("Starting hyperparameter search")

    # Define hyperparameter grid
    input_size = X_train.shape[1]
    layer_configs = [
        [input_size, 64, 32, 1], # 2 hidden layers: 64, 32
        [input_size, 128, 64, 1], # 2 hidden layers
        [input_size, 64, 32, 16, 1], # 3 hidden layers
        [input_size, 128, 64, 32, 1] # 3 hidden layers
    ]
    # activation functions and regularization terms
    activations = ['tanh', 'sigmoid']
    lambda_regs = [0.001, 0.01, 0.1]

    best_score = np.inf
    best_parameters = None
    results = []

    total_configs = len(layer_configs) * len(activations) * len(lambda_regs)
    config_count = 0

    # Grid search: iterate over all the layer configurations, activation functions, regularization terms
    for layers in layer_configs:
        for activation in activations:
            for lambda_reg in lambda_regs:
                config_count += 1
                print(f"\n[{config_count}/{total_configs}] Testing configuration:")
                print(f"Layers: {layers[1:-1]} (hidden)")
                print(f"Activation: {activation}")
                print(f"Lambda: {lambda_reg}")

                try:
                    # Perform cross-validation
                    avg_train_loss, avg_val_loss, avg_train_mape, avg_val_mape = k_fold_cross_validation(X_train, y_train, layers, activation, lambda_reg, k=3)

                    # Store results
                    config_result = {
                        'layers': layers,
                        'activation': activation,
                        'lambda_reg': lambda_reg,
                        'avg_val_loss': avg_val_loss,
                        'avg_val_mape': avg_val_mape,
                        'avg_train_loss': avg_train_loss,
                        'avg_train_mape': avg_train_mape
                    }
                    results.append(config_result)

                    print(f"Results: Val Loss = {avg_val_loss:.4f}, Val MAPE = {avg_val_mape:.4f}%")

                    # Update best configuration
                    if avg_val_loss < best_score:
                        best_score = avg_val_loss
                        best_parameters = {
                            'layers': layers,
                            'activation': activation,
                            'lambda_reg': lambda_reg
                        }
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

    print("\n\nHYPERPARAMETER SEARCH COMPLETE")
    print(f"Best validation loss: {best_score:.6f}")
    print(f"Best configuration:")
    print(f"Hidden layers: {best_parameters['layers'][1:-1]}")
    print(f"Activations: {best_parameters['activation']}")
    print(f"Lambda: {best_parameters['lambda_reg']}")

    return best_parameters, results