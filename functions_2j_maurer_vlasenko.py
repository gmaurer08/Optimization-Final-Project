
import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from cvxopt import matrix, solvers
from itertools import product

import warnings
warnings.filterwarnings('ignore')

SEED=123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"]=str(SEED)
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"

# Gauss Kernel
def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma*np.linalg.norm(x1-x2)**2)

# Polynomial Kernel
def poly_kernel(x1, x2, gamma, p):
    return (np.dot(x1, x2)+1)**p

# Vectorized kernels to avoid for loops
# Gaussian RBF kernel
def rbf_kernel_matrix(X1, X2, gamma: float):
    """
    Computes pairwise squared Euclidean distances efficiently using ||a-b||^2=||a||^2+||b||^2-2a^Tb
    Inputs:
        X1: matrix with shape (n1, d), n1 = number of samples, d = number of features
        X2: matrix with shape (n2, d), n2 = number of samples, d = number of features
        gamma: scalar
    Outputs:
        K: kernel matrix of shape (n1, n2)
    """
    # Find the kernel using the formula for squared euclidean distance (better for memory)
    X1_sq = np.sum(X1**2, axis=1)[:, None] # each row's squared norm
    X2_sq = np.sum(X2**2, axis=1)[None, :] # each row's squared norm
    # This constructs the full pairwise squared distance matrix
    eucl = X1_sq+X2_sq-2*(X1 @ X2.T)
    # this is the kernel matrix K with element ij being exp(-gamma*||xi-xj||^2)
    return np.exp(-gamma*eucl)

# Polynomial kernel (also vectorized)
def poly_kernel_matrix(X1, X2, p: int):
    """
    Builds a polynomial kernel matrix with element ij being (x_i^T z_j+1)^p
    Inputs:
        X1: (n1,d) matrix
        X2: (n2,d) matrix
        p: integer degree >=1
    Output:
        K: (n1,n2) polynomial kernel matrix
    """
    return (X1 @ X2.T+1.0)**p


def make_kernel_matrices(Xtr, Xte, kernel_name: str, **kernel_pars):
    """
    Builds train/train and train/test kernel matrices for the chosen kernel
    Inputs:
        Xtr: (n_train,d) training features
        Xte: (n_test,d) test features
        kernel_name: "rbf" or "poly" for Gaussian or Polynomial kernel
        kernel_pars: kernel parameters {'gamma':...} for rbf or {'p':...} for poly
    Outputs:
        K_train: (n_train,n_train) kernel on training set (+small jitter on diag)
        K_test: (n_train,n_test)  cross kernel between train and test
    """
    # chooses the right kernel and builds train/train and train/test kernel blocks
    if kernel_name=='rbf':
        K_train = rbf_kernel_matrix(Xtr, Xtr, gamma=kernel_pars['gamma'])
        K_test = rbf_kernel_matrix(Xtr, Xte, gamma=kernel_pars['gamma'])
    elif kernel_name=='poly':
        K_train = poly_kernel_matrix(Xtr, Xtr, p=kernel_pars['p'])
        K_test = poly_kernel_matrix(Xtr, Xte, p=kernel_pars['p'])
    else:
        # error if kernel_name is wrong
        raise ValueError("kernel_name must be 'rbf' or 'poly'")
    # ensure numerical stability
    K_train = K_train+1e-8*np.eye(K_train.shape[0]) # to avoid almost singular matrices
    return K_train, K_test

# This function solves the dual problem
#   { maximize  -1/2 a^T (Y Y^T * K) a + 1^T a
#   { subject to 0<=a_i<=C and Y^T a=0
# cvxopt solves a MIN problem, so we flip signs -> minimize 1/2 a^T (Y Y^T * K) a - 1^T a then take its negative
def solve_svm_dual(K_train: np.ndarray, ytr: np.ndarray, C: float):
    n = ytr.shape[0]  # number of training points in this fold
    Y = ytr.reshape(-1, 1) # column vector
    Q = (Y @ Y.T)*K_train # This is (y y^T) * K (elementwise via broadcasted product)

    # cvxopt form: min (1/2)a^T Q a + q^T a, with A@a<=b and U@a=v
    Q_cvx = matrix(Q, tc='d') # Positive semidefinite matrix for Q
    q_cvx = matrix(-np.ones(n), tc='d') # q=-1 -> objective matches the negated dual

    # Linear inequality constraints
    # box constraints 0<=a<=C encoded as: -I a<=0 and I a<=C (-a <= 0 and a <= C but in matrix form)
    M = np.vstack([-np.eye(n), np.eye(n)]) # Concatenates the two inequality blocks (matrices)
    v = np.hstack([np.zeros(n), np.full(n, C)]) # Corresponding right-hand sides (vector)
    # Linear equality constraints
    A_cvx = matrix(M, tc='d')
    b_cvx = matrix(v, tc='d')

    # Equality constraint enforces y^T a=0 (feasibility on the margin balance)
    U_cvx = matrix(ytr.reshape(1, -1), tc='d')
    v_cvx = matrix(0.0)
    
    # Solve the system (using default CVXOPT settings)
    sol = solvers.qp(Q_cvx, q_cvx, A_cvx, b_cvx, U_cvx, v_cvx)
    alphas = np.array(sol['x']).ravel() # these are the dual variables a_i
    return alphas, sol, Q # returning Q helps evaluate the dual objective if needed

# Knowing the optimal solution, compute the bias
def compute_b(alphas: np.ndarray, ytr: np.ndarray, K_train: np.ndarray, C: float):
    """
    Estimates the bias term b from KKT conditions using support vectors
    For any h with 0<a_h<C: y_h*(sum_j a_j y_j K_jh + b)=1 -> b= avg_h( y_h - sum_j a_j y_j K_jh ) (h is a SV index)
    Inputs:
        alphas: dual solution
        ytr: labels in {-1,+1} (y training)
        K_train: train kernel matrix
        C: margin parameter
    Outputs:
        b: scalar bias (averaged across margin SVs)
        sv: boolean mask of support vectors (a_i>eps)
    """
    # finds the bias b by averaging across margin SVs (0<a_h<C), fallback to all SVs if none are free
    eps = 1e-6
    sv = (alphas>eps) # mask of support vectors (nonzero alphas)
    sv_margin = (alphas>eps) & (alphas<C-eps) # mask of free (margin) support vectors
    idx = np.where(sv_margin)[0] # Convert the boolean mask into an array of indices h of free SVs to use in the averaging
    if idx.size==0: # in case there are no free SVs, use all SVs
        idx = np.where(sv)[0]
    # b = y_h - sum_j a_j y_j K_jk, averaged over chosen SVs
    b_vals = ytr[idx]-(alphas*ytr) @ K_train[:, idx]
    return float(np.mean(b_vals)), sv # returns mean bias and the SV mask
 
# Function that computes the decision scores (f(x)) BEFORE the sign function
def decision_function(alphas, ytr, K_test, b):
    """
    Computes decision scores sum_i a_i y_i K(x_i,x)+b for a batch
    Inputs:
        alphas: array long n_train
        ytr: array long n_train
        K_test: shape (n_train,n_test) matrix kernel between train and test
        b: bias
    Output:
        scores: array of length n_test with decision scores
    """
    # computes sum_i a_i y_i K(x_i,x)+b for a batch via kernel block K_test
    return (alphas*ytr) @ K_test+b

def accuracy_from_scores(scores, y_true):
    """
    Maps the decision scores to {0,1} based on sign (1 if positive, 0 if negative)
    Then computes the accuracy of y_pred compared to y_true
    Inputs:
        scores: vector of decision scores
        y_true: ground truth in {0,1}
    Output:
        accuracy: float in [0,1]
    """
    # Scores>=0 -> class 1 else 0
    y_pred = (scores>=0).astype(int)
    return float(np.mean(y_pred==y_true))

def grid_from_dict(d):
    """
    Expands a dictionary of lists parameter grid into a generator of dict combinations.
    For example, given a list of C values and a list of gamma values, it will generate
    dictionaries with keys 'C' and 'gamma' corresponding to each combination of values
    """
    # expands a dictionary of lists into a cartesian product of parameter dictionaries
    keys = list(d.keys())
    for values in product(*[d[k] for k in keys]): # cartesian product
        yield dict(zip(keys, values)) # pair keys with the current values tuple to form a parameter dictionary

def cross_validate_svm(X, y_true, kernel_name, param_grid, k=5, seed=SEED):
    """
    k-fold stratified CV to select hyperparameters by mean validation accuracy.
    For each split: build kernel, solve dual on train, compute b, score on val.
    Inputs:
        X: features matrix (standardized)
        y_true: vector of true labels in {0,1}
        kernel_name: 'rbf' or 'poly'
        param_grid: dict of lists (for example {'C':[...], 'gamma':[...]} or {'C':[...], 'p':[...]})
        k: number of folds
    Outputs:
        best_params: dict with best hyperparams
        best_mean: float mean validation accuracy for best_params
        history: list of (params, mean_acc, fold_accs)
    """
    # Stratified splits preserve class balance; we remap labels to {-1,+1} per fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    # Initializations
    best_params, best_mean = None, -np.inf # Initialize the best so far hyperparameters and best mean validation accuracy
    history = []  # Initialize list to store (params, mean_acc, fold_accs)

    # Loop over hyperparameter combinations generated by grid_from_dict
    for kernel_pars in grid_from_dict(param_grid):
        # Initialize list to collect fold val accuracies
        fold_accs = []

        # Iterate over indices for train and validation subsets
        for train_idx, val_idx in skf.split(X, y_true):

            # Extract train and validation subsets
            Xtr, Xval = X[train_idx], X[val_idx] # inputs
            ytr01, yval01 = y_true[train_idx], y_true[val_idx] # labels
            ytr = np.where(ytr01==0, -1.0, 1.0) # train labels in {-1,1}

            # Build kernel matrices for this fold
            K_train, K_val = make_kernel_matrices(Xtr, Xval, kernel_name, **kernel_pars)

            # Solve the dual with CVXOPT
            alphas, sol, Q = solve_svm_dual(K_train, ytr, C=kernel_pars['C'])

            # estimate bias using margin SVs, then evaluate on validation
            b, _ = compute_b(alphas, ytr, K_train, C=kernel_pars['C']) # bias terms
            scores_val = decision_function(alphas, ytr, K_val, b) # decision scores
            acc_val = accuracy_from_scores(scores_val, yval01) # accuracy on validation
            fold_accs.append(acc_val) # store fold accuracy
        
        # Compute mean validation accuracy and store in history
        mean_acc = float(np.mean(fold_accs))
        history.append((kernel_pars.copy(), mean_acc, fold_accs.copy()))

        # track the best hyperparameters by mean validation accuracy
        if mean_acc>best_mean:
            best_mean, best_params = mean_acc, kernel_pars.copy()
    return best_params, best_mean, history
