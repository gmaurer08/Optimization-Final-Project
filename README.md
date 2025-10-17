# Optimization-Final-Project

This repository contains the final project for the Optimization Methods for Data Science course from the Data Science master's degree at Sapienza, A.Y. 2024/2025.

### Team Members
- Géraldine Valérie Maurer, maurer.1996887@studenti.uniroma1.it
- Viktoriia Vlasenko, vlasenko.2088928@studenti.uniroma1.it

## Repository structure
```project-root/
Main
|
├── Part 1
│   ├── functions_1j_maurer_vlasenko.py          # MLP Python functions
│   └── run_1i_maurer_vlasenko.ipynb             # MLP regression from scratch Notebook
|
├── Part 2
│   ├── functions_2j_maurer_vlasenko.py          # SVM Python functions
│   └── run_2i_maurer_vlasenko.ipynb             # SVM dual optimization (CVXOPT) Notebook
|
├── Plots                                        # Plots for the first part
│   ├── plot1.png                                # Optimization Progress (initial vs. final loss)
│   ├── plot2.png                                # MSE and MAPE during Train and Test
│   ├── plot3.png                                # Validation Loss vs. Regularization Strength
│   └── plot4.png                                # Training vs. Validation Loss during Hyperparameter search
│
├── AGE_PREDICTION.csv                           # Age Regression Dataset
├── GENDER_CLASSIFICATION.csv                    # Gender Classification Dataset
│
├── Maurer_Vlasenko.pdf                          # Project Report
│
├── LICENSE                                      # MIT license
│
└── README.md                                    # README file
```

## Dataset
The dataset used in this project is the [UTKFace dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new). The over 20'000 pictures in the dataset were previously used to train a ResNet, and their features were extracted by the convolutional backbone to serve as input vectors to the models built in the project. Images of individuals with underrepresented ages in the dataset were removed to ensure a more balanced distribution. The project is split into two parts, each with its own pre-processed dataset.
- **Part 1 - Regression**: [AGE REGRESSION.csv](https://github.com/gmaurer08/Optimization-Final-Project/blob/main/AGE_PREDICTION.csv)
- **Part 2 - Classification**: [GENDER CLASSIFICATION.csv](https://github.com/gmaurer08/Optimization-Final-Project/blob/main/GENDER_CLASSIFICATION.csv)

The datasets have a header where the first columns contains the features extracted by the ResNet (labelled as “feat i”), while the last column is the ground truth (labeled as “gt”). In particular, the ground truth is a float number between 0 and 100 for AGE REGRESSION and a binary variable for GENDER CLASSIFICATION.

*Source: OMDS final project guidelines*

## Methods
In this project, we explored two supervised learning problems built entirely from first principles.
- In **Part 1**, we implemented a multilayer perceptron (MLP) for age regression using only NumPy and SciPy. The full pipeline included data standardization, analytic forward and backward propagation, and L2-regularized loss optimization with the L-BFGS-B algorithm (`scipy.optimize.minimize`). Hyperparameters such as layer depth, width, activation function (tanh or sigmoid), and regularization strength ($\lambda \in \{0.1, 0.01, 0.001\}$) were tuned using 3-fold cross-validation. Model performance was measured with MSE and MAPE, emphasizing generalization across folds.
- In **Part 2**, we studied binary gender classification using a support vector machine (SVM) with an RBF kernel. The dual optimization problem was solved both via CVXOPT and a custom Most Violating Pair (MVP) decomposition (q=2). Hyperparameters ($C \in \{0.1, 1, 10, 100\}$, $\gamma \in \{0.01, 0.1, 1.0\}$) were chosen by 5-fold stratified cross-validation. Both implementations shared a consistent preprocessing pipeline (standardization, label mapping, reproducibility controls) and reported metrics such as accuracy, precision, recall, and F1 score.

## Results
For the **MLP regression**, validation MAPEs clustered around 23–24%, with the best configuration being a [32, 64, 32, 16, 1] architecture using sigmoid activations and $\lambda = 0.01$. This model achieved a training MAPE of 23.3% and test MAPE of 22.8%, with nearly identical MSEs (~95 train, ~92 test), demonstrating stable generalization and limited overfitting. Regularization played an important role, with a smaller value ($\lambda=0.001$) encouraging overfitting, while a too large value ($\lambda= 0.1$) reduced flexibility.

<img width="1022" height="375" alt="image" src="https://github.com/user-attachments/assets/38d406c6-b87d-419f-a64e-d912f6ff458f" />

<img width="867" height="795" alt="image" src="https://github.com/user-attachments/assets/e7da061e-46a1-48e3-9955-c9a05774e71a" />

For the **SVM classifier**, the optimal hyperparameters were $C=1$, $\gamma=0.1$, with a 92.3% train accuracy and 90.0% test accuracy and balanced class performance (F1≈0.90). The MVP solver matched the performance of the CVXOPT baseline while being more computationally efficient (0.35s vs. 1.6s), confirming both correctness and scalability. Confusion matrices revealed symmetric errors, indicating well-calibrated margins and no bias toward either class.

<img width="1234" height="205" alt="image" src="https://github.com/user-attachments/assets/68da8dcd-d4c8-46c9-94c5-ae8d42260a9f" />

## Conclusions

Building both models from scratch provided a deep understanding of optimization and regularization dynamics in supervised learning. The MLP experiment showed how moderate network depth and mid-range L2 penalties lead to a positive trade-off between bias and variance. The SVM study confirmed that RBF kernels with properly tuned parameters achieve strong, generalizable performance. Implementing both the dual QP and MVP solvers highlighted the practical balance between exact optimization and computational efficiency. Overall, the project delivered interpretable, reproducible, and well-generalized models for both regression and classification tasks.
