# Research-Project-FL
# Securing Transactions: A Novel Federated Approach to Credit Card Fraud Detection Using Metaheuristic Optimization Techniques

## 📘 Description
This repository contains the Contribution and Methodology 
This repository contains the implementation and dataset used in the study on A Novel Federated Approach to Credit Card Fraud Detection Using Metaheuristic Optimization Techniques.

## 📘 Contributions
    Federated Learning (FL) is a decentralized machine learning framework in which multiple clients collaboratively train a shared model under the coordination of a central server, without exposing their raw data (Sarker, 2021). This privacy-preserving setup introduces new challenges, particularly in hyperparameter optimization, which remains a prominent area of ongoing research (Kairouz et al., 2021). Communication overhead significantly influences model performance in FL environments. To address this, we examine a communication-efficient strategy that applies local hyperparameter tuning prior to the federation phase, aiming to reduce transmission costs—a critical bottleneck in FL systems (Nilsson et al., 2018). Furthermore, we present a set of reproducible empirical benchmarks to evaluate federated optimization techniques that incorporate metaheuristic approaches. The key contributions of this study are outlined as follows:
•Addressing Class Imbalance: The Synthetic Minority Oversampling Technique (SMOTE), a widely used resampling method, has been used to tackle the issue of unbalanced data. This technique ensures that minority classes are adequately represented in the dataset, leading to more robust and generalizable models.
•Metaheuristic Optimization Strategies: To enhance the baseline model’s performance, this research applies seven metaheuristic optimization strategies. These methods, which include algorithms such as Genetic Algorithm, Particle Swarm Optimization, and Ant Colony Optimization, are used to optimize model parameters and improve performance.
•Pre-Communication Model Optimization: Optimized models were developed with numerous changes prior to any server communication. This pre-communication optimization significantly reduces the quantity of communication required for model training in a federated learning environment. Consequently, this approach minimizes network bandwidth usage and enhances training efficiency.
• Federated Learning Efficiency: To assess the effectiveness of improved model within the Federated Learning framework, the learning process was performed several times with the optimized global model. This iterative technique guarantees that models are appropriate for dispersed learning environments

## 📘 Methodology

The proposed federated learning (FL) model for credit card fraud detection (CCFD) is implemented in three phases:
1. Preprocessing Phase
- Challenge: Credit card fraud datasets are highly imbalanced (fraudulent transactions ≪ legitimate ones, e.g., 1:1000 ratio).
- Problem: Classifiers tend to favor the majority class, misclassifying fraudulent transactions as legitimate.
- Solution:
- Apply Synthetic Minority Oversampling Technique (SMOTE) to balance the dataset.
- Reduce skewed label distribution to improve fraud detection accuracy.
- Address heterogeneity to prevent misconvergence of the global model in FL.
2. Optimization Phase
- Issue: Standard optimization methods (e.g., extended SGD) in FL incur high communication costs.
- Approach:
- Employ metaheuristic optimization algorithms (seven in total) to minimize communication overhead.
- Models are locally optimized and repeatedly updated before server communication.
- Metaheuristics balance exploration and exploitation, enabling efficient convergence without relying on domain-specific heuristics.

3. Federated Learning Phase
- Evaluation: Train and test the globally optimized model within the FL framework.
- Model Architecture (CNN):
- Input Layer: Transaction feature vector.
- Conv Layer 1: 28 filters + ReLU → extract low-level features (frequency, spending behavior).
- Conv Layer 2: 32 filters + ReLU → capture intermediate patterns.
- Conv Layer 3: 64 filters + ReLU → detect anomalies/outliers.
- Max Pooling: Applied after Conv Layer 2 to reduce dimensions and mitigate overfitting.
- Flatten Layer: Converts multidimensional data into 1D vector.
- Fully Connected Layer: Linear transformation + Dropout (50%) → reduce overfitting.
- Output Layer: Sigmoid activation → binary classification (fraudulent vs. non-fraudulent).






## 📂 Dataset Information
- **Source:**
[[Dataset -1]	 Machine Learning Group – ULB, “Credit card fraud detection: Anonymized credit card transactions labeled as fraudulent or genuine,” 2018. [Online]. Available: https://www.kaggle.com/mlg-ulb/creditcardfraud. [Accessed 24 Aug 2023].
[Dataset -2]	 T. Avci, “Fraud detection on bank payments,” 2020. [Online]. Available: https://www.kaggle.com/code/turkayavci/fraud-detection-on-bank-payments/input. [Accessed 24 Aug 2023].
[Dataset -1]	 Gksj7, “Creditcardcsvpresent,” 2019. [Online]. Available: https://github.com/gksj7/creditcardcsvpresent. [Accessed 24 Aug 2023].
- **Format:** [CSV]
- **Size:**
Name	      Instances	  Features	Normal	 Fraudulent
Dataset-1	284.807	  31	       284.315	 492
Dataset-2	594.643	  10	       587.443	 7200
Dataset-3	3075	  12	       2627	 448

- **Description** The experiment's outcomes on three publicly accessible datasets confirm our suggested algorithms' effectiveness.
   [Dataset-1]: a publicly available dataset from Kaggle, originally provided by the Machine Learning Group – ULB (2018), which contains real, anonymized credit card transactions conducted by European cardholders. The dataset comprises 284,807 transactions recorded in September 2013. It is notably imbalanced, with only 492 instances labeled as fraudulent, and contains no missing values. Of the 30 features included, only transaction amount and transaction time are known; the remaining features have been anonymized for confidentiality
   [Dataset-2]: There is no publication of private or legal client transactions or personal information in BankSim data sets (Avci, 2020). Academics and others can therefore utilize it to develop and analyze fraud detection techniques. Even for people who have access to their own data, synthetic data has the benefit of being simpler, quicker, and less expensive for testing. We argue that BankSim generates data that closely resembles the pertinent components of real-world data. Based on a sample of aggregated transactional data supplied by a Spanish bank, BankSim is an agent-based bank payment simulator. 594643 instances of 7200 fraudulent transactions are included in the sample.
   [Dataset-3]:  An overall of 3075 samples, 448 of which were positive samples, are included in the Creditcardcsvpresent data set (Gksj7, 2019). 
]

 ## Citation of the Datasets

- Dataset 1 (Kaggle ULB, 2018) → Widely used across 10+ studies (Abdul Salam 2024, Benchaji 2021, Itoo 2020, Makki 2019, Puh 2019, Xuan 2018, Yang 2019, Zheng 2021, Zhu 2020).
- Dataset 2 (BankSim, Avci 2020) → Used in Avci (2020).
- Dataset 3 (Creditcardcsvpresent, Gksj7 2019) → Used in Gksj7 (2019).


## 💻 Code Information
- **Main Scripts:** [List key files, e.g., `train.py`, `preprocess.py`]
- **Functions:**
## 💻 Function Overview

### `input_spec()`
Defines the input specification for the model:
- **Features:** Tensor of shape `[None, 28]` with type `float64`
- **Labels:** Tensor of shape `[None, 1]` with type `int64`
- Purpose: Ensures reproducibility by clearly declaring expected input shapes and data types.

---

### `build_model()`
Constructs a **Sequential Keras model** with the following architecture:
- Input layer: 28 features
- Dense layer: 32 units, ReLU activation
- Dense layer: 64 units, ReLU activation
- Dense layer: 32 units, ReLU activation
- Output layer: 1 unit, Sigmoid activation
- *(Dropout layers are included but commented out — can be enabled for regularization)*

---

### `compile_model()`
Compiles the Keras model with:
- **Loss:** `BinaryCrossentropy` (for binary classification tasks)
- **Optimizer:** `Adam` (adaptive learning rate optimization)
- **Metrics:** `BinaryAccuracy` (evaluates classification performance)

Purpose: Prepares the model for training by linking architecture, loss function, optimizer, and evaluation metrics.

## The model_fn() function defines how to create a TensorFlow Federated (TFF) model from a compiled Keras model.

- Step 1 – Compile Model
compile_model() prepares the base Keras model with architecture, optimizer, and training configuration
- Step 2 – Clone Model
tf.keras.models.clone_model() creates a fresh copy of the compiled model.
👉 This prevents shared state issues when multiple federated clients use the model.
- Step 3 – Wrap in TFF Model
tff.learning.from_keras_model() converts the cloned Keras model into a TFF model.
- input_spec(): Defines the expected input structure (shapes, types) for federated datasets.
- Loss Function: BinaryCrossentropy() is used for binary classification tasks.
- Metrics: Tracks BinaryAccuracy, Precision, and Recall during training/evaluation.

##build_federated_averaging_process()
- Defines the federated training algorithm (FedAvg).
- Uses SGD with learning rate = 0.1 and momentum = 0.2 for both clients and server.
- Training Loop
- Runs for EPOCHS = 100.
- Each round:
- Clients train locally on their datasets.
- Updates are averaged on the server.
- Metrics are collected in train_hist.
- Runtime Measurement
- time.time() is used to measure total training runtime.
- Federated Evaluation
- build_federated_evaluation(model_fn) evaluates the final global model.
- federated_metrics contains validation performance across Alice and Bob’s datasets.

## get_shape() → Extracts the shape of each weight tensor in the model.
## set_shape() → Reshapes flat arrays back into the original layer weight shapes.
## evaluate_nn() → Evaluates the model for a given set of weights.
- Uses accuracy as the metric.
- Objective function = 1 - accuracy (minimization).
####### PSO Setup:
- Particles: 10 candidate solutions.
- Dimensions: 5153 (total number of weights).
- Bounds: Each weight lies between -1.0 and +1.0.
- Options:
- c1=0.4 → cognitive coefficient (particle’s own experience).
- c2=0.6 → social coefficient (swarm’s best experience).
- w=0.4 → inertia weight (balance between exploration/exploitation).
- Optimization:
- Runs for 10 iterations.
- Finds the best weight vector (pos).
- Applies optimized weights to the model.
- Runtime:
- Measured with time.time().
- Prints total optimization time.
##### ACO Setup 
# Define hyperparameters
q0 = 0.2
alpha = 0.7
beta = 0.1
rho = 0.2
num_ants = 3
num_iterations = 5

# Compile global model
global_model = compile_model()
m = global_model.count_params()
lb = np.full(m, -1)
ub = np.full(m, 1)

# Run ACO
import time
start = time.time()
best_position, best_fitness = aco_fl(objective_function, num_ants, num_iterations, 
                                     q0, alpha, beta, rho, lb, ub, global_model)
end = time.time()

print(f"Best Position: {best_position}")
print(f"Best Fitness: {best_fitness}")
print(f"Runtime: {end - start} seconds")

#### CSO Setup
# Define hyperparameters
num_cats = 5
num_iterations = 10

# Compile global model
global_model = compile_model()
m = global_model.count_params()
lb = np.full(m, -1)
ub = np.full(m, 1)

# Run CSO
import time
start = time.time()
best_position, best_fitness = cso_fl(objective_function, num_cats, num_iterations, lb, ub, global_model)
end = time.time()

print(f"Best Position: {best_position}")
print(f"Best Fitness: {best_fitness}")
print(f"Runtime: {end - start} seconds")

###### HHO Setup
# Build global model
model = compile_model()

# Set HHO parameters
n = 10
m = model.count_params()
alpha = 0.3
beta = 0.6
gamma = 0.9
lb = np.full(m, -1)
ub = np.full(m, 1)

# Run HHO
import time
start = time.time()
g_best, x_best = hho_federated_learning(objective_function, n, m, alpha, beta, gamma, lb, ub, 10)
end = time.time()

print(f"Best Fitness: {g_best}")
print(f"Best Solution: {x_best}")
print(f"Runtime: {end - start} seconds")

#### FHO Setup
epoch = 10       # Number of iterations
pop_size = 10    # Population size

import time
start = time.time()

opt_model = GTO.Matlab101GTO(epoch, pop_size)
best_position, best_fitness = opt_model.solve(problem_dict1)

print(f"Solution: {best_position}, Fitness: {best_fitness}")

end = time.time()
print(f"Runtime of the program is {end - start}")

 ####### HBO Setup
# Define range for hyperparameters
learning_rates = [0.001, 0.0005]
num_filters_values = [32]

# Optimize the CNN model
epochs = 10



##- make_tf_dataset():
- Drops unused columns.
- Splits into train/test.
- Balances data with SMOTE.
- Converts to TensorFlow dataset.
- Shuffles and batches.
- Alice & Bob:
Each client has its own train/validation datasets, simulating federated learning participants.
- Parameters:
- EPOCHS = 100 → training iterations.
- BATCH_SIZE = 64 → batch size for training.
- Validation datasets use batch size = 1 for evaluation.



## Requirements 
To reproduce the experiments, install the following dependencies:- Python 3.9 (recommended)
- numpy==1.23.5
- pandas==1.5.3
- scikit-learn==1.2.2
- imbalanced-learn==0.10.1
- pyswarms==1.3.0
- tensorflow==2.12.0
- tensorflow-federated==0.71.0
- nest-asyncio==1.5.6

##Hardware & OS
 Intel Core i7 1.80 GHz CPU
- 16 GB RAM
- Windows 10 (64-bit) operating system

## ⚙️ Usage Instructions
1. Create a new environment
2. TensorFlow Installation
%%shell
pip install sklearn
pip install pandas
pip install matplotlib
pip install tensorflow

pip uninstall --yes tensorboard tb-nightly

pip install --quiet --upgrade tensorflow-federated 
if it`s error
pip install --user --quiet --upgrade tensorflow-federated 

pip install --quiet --upgrade nest-asyncio
pip install --quiet --upgrade tensorboard 
 
3. Run Coding on Jupyter:  
   Final_First_Dataset.ipynb
   Final_Second_Dataset.ipynb
