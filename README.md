# Traffic Speed Predicto

## 🧭 Project Overview
The **TrafficSpeedPredicto** project aims to apply predictive models to Guangzhou traffic speed data. A dataset with randomly generated missing values is created, and the **BGCP (Bayesian Gaussian CP Decomposition)** algorithm is employed to impute the missing entries. The imputation quality of BGCP is then evaluated, and the completed dataset is used for model training. The trained model is subsequently tested on the original (fully observed) dataset. By comparing model performance, the reliability of BGCP’s imputation on this dataset is demonstrated.

The project implements two simple time-series prediction models — **LSTM** and **LSTM_CNN** — to forecast traffic speeds.

---

## 📊 Dataset
- **Name:** Guangzhou Speed Data  

- **Shape:** Tensor of size **214 × 61 × 144**  
  - `214` — number of road segments  
  - `61` — number of days  
  - `144` — number of time intervals per day  
  
- Random missing values are introduced for imputation experiments.

  ![image-20251113163305417](./datasets/75eb7ec7b388b9720b6b144d105f7e75.png)



---

## 🔧 Methodology
1. **Data Preprocessing:**  
   - Random missing entries are generated.  
   
     ```python
     import time
     import os
     import scipy.io
     import numpy as np
     # Set random seed for reproducibility
     np.random.seed(1000)
     # Load the original dense tensor
     dense_tensor = scipy.io.loadmat('./Guangzhou-data-set/tensor.mat')['tensor']
     # Get tensor dimensions
     dim = dense_tensor.shape
     # Missing rate
     missing_rate = 0.4  # Random missing (RM)
     # Generate sparse tensor with missing values
     sparse_tensor = dense_tensor * np.round(np.random.rand(dim[0], dim[1], dim[2]) + 0.5 - missing_rate)
     # Print the tensor dimensions
     print(dim)
     # Define the output directory
     save_dir = './Guangzhou-data-setMitMissingValue'
     os.makedirs(save_dir, exist_ok=True)  # Create directory if it does not exist
     # Save the sparse tensor as a .mat file
     scipy.io.savemat(os.path.join(save_dir, 'sparse_tensor.mat'), {'tensor': sparse_tensor})
     ```
   
   - BGCP algorithm is applied for tensor completion.  
   
     ```python
     def BGCP(dense_tensor, sparse_tensor, factor, burn_iter, gibbs_iter, output_log_path):
         """Bayesian Gaussian CP (BGCP) decomposition."""
         dim = np.array(sparse_tensor.shape)
         rank = factor[0].shape[1]
         if np.isnan(sparse_tensor).any() == False:
             ind = sparse_tensor != 0
             pos_obs = np.where(ind)
             pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
         elif np.isnan(sparse_tensor).any() == True:
             pos_test = np.where((dense_tensor != 0) & (np.isnan(sparse_tensor)))
             ind = ~np.isnan(sparse_tensor)
             pos_obs = np.where(ind)
             sparse_tensor[np.isnan(sparse_tensor)] = 0
         show_iter = 200
         tau = 1
         factor_plus = []
         for k in range(len(dim)):
             factor_plus.append(np.zeros((dim[k], rank)))
         temp_hat = np.zeros(dim)
         tensor_hat_plus = np.zeros(dim)
         for it in range(burn_iter + gibbs_iter):
             tau_ind = tau * ind
             tau_sparse_tensor = tau * sparse_tensor
             for k in range(len(dim)):
                 factor[k] = sample_factor(tau_sparse_tensor, tau_ind, factor, k)
             tensor_hat = cp_combine(factor)
             temp_hat += tensor_hat
             tau = sample_precision_tau(sparse_tensor, tensor_hat, ind)
             if it + 1 > burn_iter:
                 factor_plus = [factor_plus[k] + factor[k] for k in range(len(dim))]
                 tensor_hat_plus += tensor_hat
             if (it + 1) % show_iter == 0 and it < burn_iter:
                 temp_hat = temp_hat / show_iter
                 print('Iter: {}'.format(it + 1))
                 print('MAPE: {:.6}'.format(compute_mape(dense_tensor[pos_test], temp_hat[pos_test])))
                 print('RMSE: {:.6}'.format(compute_rmse(dense_tensor[pos_test], temp_hat[pos_test])))
                 temp_hat = np.zeros(sparse_tensor.shape)
                 print()
         factor = [i / gibbs_iter for i in factor_plus]
         tensor_hat = tensor_hat_plus / gibbs_iter
         print('Imputation MAPE: {:.6}'.format(compute_mape(dense_tensor[pos_test], tensor_hat[pos_test])))
         print('Imputation RMSE: {:.6}'.format(compute_rmse(dense_tensor[pos_test], tensor_hat[pos_test])))
         print()
         return tensor_hat, factor
     ```
   
     **Imputation Performance：**
   
     Imputation MAPE: 0.0835415
     Imputation RMSE: 3.6097
   
2. **Models Implemented:**  
   - **LSTM:** A baseline recurrent neural network model for time-series prediction.  
   
     ```python
     model = Sequential()
     model.add(LSTM(input_shape=(TIME_STEPS, INPUT_SIZE),
                        units=64,
                        return_sequences=True))
     model.add(Activation('tanh'))
     model.add(Dropout(0.5))
     model.add(LSTM(units=256))
     model.add(Activation('tanh'))
     model.add(Dropout(0.5))
     model.add(Dense(OUTPUT_SIZE))
     ```
   
   - **LSTM_CNN:** A hybrid model combining convolutional and recurrent layers to capture both spatial and temporal dependencies. 
   
     ```python
     model = Sequential()
     # First convolutional layer
     model.add(TimeDistributed(Conv1D(40,
                      kernel_size[1],
                      strides=1,
                      padding='valid'), input_shape=[time_steps, 214, 1]))
     model.add(TimeDistributed(BatchNormalization()))
     model.add(TimeDistributed(Activation('relu')))
     # Second convolutional layer
     model.add(TimeDistributed(Conv1D(40, kernel_size[1], padding='valid')))
     model.add(TimeDistributed(BatchNormalization()))
     model.add(TimeDistributed(Activation('relu')))
     # Third convolutional layer
     model.add(TimeDistributed(Conv1D(40, kernel_size[0], padding='valid')))
     model.add(TimeDistributed(BatchNormalization()))
     model.add(TimeDistributed(Activation('relu')))
     # Flatten
     model.add(TimeDistributed(Flatten()))
     model.add(Dense(214))
     model.add(Dropout(0.5))
     
     model.add(LSTM(64, return_sequences=True))
     model.add(Activation('tanh'))
     model.add(Dropout(0.5))
     model.add(LSTM(256))
     model.add(Activation('tanh'))  
     model.add(Dropout(0.5))
     model.add(Dense(214))
     
     model.summary()
     ```
   
3. **Evaluation:**  
   
   - Comparison between ground truth and BGCP-imputed data.  
   - Comparison of the performance on the original dataset between the model trained on the original data and the model trained on the BGCP-imputed data. 

---

## 🧠 Technologies Used
- **BGCP Implementation** — Bayesian Gaussian CP Decomposition for tensor completion  

- **TensorFlow** & **Keras** — Deep learning frameworks  

- **NumPy** & **Pandas** — Data handling and manipulation  

- **Matplotlib** **&Dash**— Visualization  

  

---

## 📂 Prediction

##### standard for evaluation(**MSE_loss, MAE, RMSE, Cosine Similarity**)

Lstm from Original Data:[0.009148206561803818, 0.06146997958421707, 0.09377942979335785, 0.9879339337348938]

Lstm from Imputed Data:[0.026158226653933525, 0.12577585875988007, 0.15631181001663208, 0.9837740063667297]

lstm_CNN from Original Data:[0.00955184455960989, 0.06182552129030228, 0.09529214352369308, 0.9871290922164917]

lstm_CNN from Imputed Data: [0.01881762407720089, 0.10619371384382248, 0.13531117141246796, 0.9793261885643005]

### Future Work

Explored potential model extensions (e.g., Graph Convolutional Networks and Attention Mechanisms) and other Imputation_algorithms.
