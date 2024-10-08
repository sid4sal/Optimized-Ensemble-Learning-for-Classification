# **Optimized Ensemble Learning for Classification of GST Data**  
*Binary Classification of GST data with AI/ML Techniques*  
**Team-ID = GSTN_715**

## **Overview**  
This project is based on the binary classification of GST data. The task was to develop AI/ML models to predict the target class using machine learning and deep learning techniques. We implemented an Ensemble Model using multiple classifiers to enhance performance on the given datasets.

### **Table of Contents**  
1. [Approach](#approach)
    - [Problem Definition](#problem-definition)    
    - [Analyzing the Dataset](#analyzing-the-dataset)  
    - [Dataset Processing](#dataset-processing)  
    - [Explored Solutions](#explored-solutions)  
    - [Final Solution](#final-solution)  
2. [Results](#results)  
    - [Improvements/Future Scope](#improvementsfuture-scope)  
3. [Code Documentation and Manual](#code-documentation-and-manual)  
    - [Environment Setup](#environment-setup)  
    - [Project Structure](#project-structure)  
    - [Training the Model](#training-the-model)  
    - [Making Predictions](#making-predictions)  
- [References](#references)

## **Approach**  
Explanation of the approach and the steps taken in model development.  

### **Problem Definition**  
The task is to perform classification on the provided dataset using AI/ML techniques. The already split dataset [training (60%) and testing (20%)] was given. Both train and test data contains two csv files, one for inputs and other for labels.  

### **Analyzing the Dataset**  
We performed detailed analysis of the given dataset:
- **Input data**: Contains an 'ID' column along with 22 features named 'Column0' through 'Column21'.
- **Target data**: Contains an 'ID' column and the 'target' column specifying the true class of the samples (either 0 or 1).
- **Class Imbalance**: There is an imbalance in the target label for both the train and test datasets. The majority class (0) outnumbers the minority class (1).
- **Missing Values**: Some features contain a large proportion of missing (NA) values.
- **Homogeneous Features**: There are some features where most of the values are the same.
- **Outliers**: There are outliers in some of the features.
- **Correlations**: It is observed from the correlation matrix that very few columns are highly correlated and those which are highly correlated has categorical data so it is chosen to not omit any columns/variables. Column18 is highly correlated with the output/target.  

- **Feature Distribution**: Distributations of Features in the Train Dataset  
    <img src="https://drive.google.com/thumbnail?id=1GOrKudIFHXCaNo-4pvqUNwC2HsfbLGFW&sz=w1017" alt="feature_distributation_train.png" width="700"/>  

- **Class Distribution**: Distributation of the Classes in the Train Dataset:  
    <img src="https://drive.google.com/thumbnail?id=1QDMc_by-oF5KIb0UceQdLdQ7a4_ChFfq&sz=w567" alt="distributation_train_labels.png" width="400"/>  

- **Missing Values Heatmap**: Heatmap of Missing values in the Train Dataset:  
    <img src="https://drive.google.com/thumbnail?id=1CRVPgk4DkJEK2U4lLe9ngax0QS1XCMYy&sz=w857" alt="missing_data_train.png" width="500"/>  

- **Correlation Matrix**: Correlation Between Features and Target:  
    <img src="https://drive.google.com/thumbnail?id=1sMabPZWKx4gmpgB6pGxWumrN235-cal8&sz=w1033" alt="correlation_matrix_train.png" width="600"/>  


### **Dataset Processing**  
We identified that the below techniques are most suitable for this dataset by analysing the dataset in detail and making predictions using different preprocessing techniques with a basic random forest classifier.  
- **Outlier Handling**: We capped extreme values using Z-Score and Interquartile Range (IQR) analysis.  
- **Feature Engineering**: For certain columns, new features were created to track whether a value is NA (for those columns who has most of the values as NA) or to track whether a value is the most frequent value (for those columns who has most of the values as the same value).  
- **Scaling**: Categorical columns were scaled using min-max normalization, and continuous columns were normalized using Z-normalization (this is chosen because the outliers are taken care of beforehand).  
- **Imputation**: NA values were handled with a Bayesian ridge regression imputation method. Other techniques like non-linear regression were explored but failed (didn't converge) due to significant missing data in some columns.

### **Explored Solutions**  
Several machine learning and deep learning approaches were evaluated:  
- **Random Forest Classifier** : ML model  
- **Random Forest with Class Weights** : To account for class imbalance.   
- **Deep Neural Network (DNN)** : Tuned for best performance without overfitting.  
- **Deep Neural Network with Class Weights** : To account for class imbalance.  
- **Autoencoder**: Pretraining an autoencoder neural network (which encodes the training samples into n-dimensions then decodes it) then using a Deep Neural Network on top of it for final prediction.  
- **Large Deep Neural Network** : Neural Network with depth and more neurons.  
- **Ensemble Model**: Ensemble of one Deep Neural Network Model and one XGB Classifier Model (One robust for class 1 and other for class 0) with Meta Neural Network Model for final prediction.  
- **Anomaly Detection (Autoencoder)**: Train an autoencoder neural network on only class 1, then classify based on reconstruction error (so that class 1 samples will have low error while class 0 will have large error).  
- **Deep Metric Learning (DML)**: Training a Siamese neural network to learn an n-dimensional embedding space using contrastive loss. Then using these projections for classification.  

The models produced varying results, and the ensemble model had the best performance.

Note: If needed, the code of these models is also included in our project files (inside the "Model_Scripts" folder).

### **Final Solution**  
*Final Model: Ensemble Model*  
Our final solution was an ensemble of two models with a meta-model on top. This solution combines the strength of multiple models and addresses the weaknesses of individual models. The model is structured as follows: 
1. **XGBoost Classifier**: First Model in our Ensemble model is the XGB classifier trained to classify class 1 robustly, i.e. every example classified as class 1 highly likely belongs to class 1, but the same cannot be said for class 0. The overall performance of this model is very low.  
2. **Deep Neural Network**: Second Model in our Ensemble model is the Deep Neural Network model specifically optimized to classify samples of class 0 robustly, i.e. every sample that this model predicts as class 0 is highly likely to be truly class 0, but the same cannot be said for class 1. The overall performance of this model is poor.  
3. **Meta Neural Network**: A neural network that takes input from these two models as input features to make the final prediction. It learns how to combine other model's outputs to produce a more accurate final prediction. This model has high overall performance despite the poor performance of the individual models.  

- **Architecture of Meta Model**  
    <img src="https://drive.google.com/thumbnail?id=1i_YTkrGFbsO7WaYKMaKDs7_3Hlip078D&sz=w4524" alt="ensemble_model_architecture_diagram.png" width="700"/>  

## **Results**  

- **Metrics Used**  
    For this task, we have used the following metrics to determine the performance of our models:
    - Accuracy  
    - Precision  
    - Recall  
    - F1 Score  
    - AUC-ROC  
    - Confusion Matrix  
    - Binary Cross Entropy Loss  

- **Comparing models**  
    The comparison of performance of different models on the Test Dataset:  
    <img src="https://drive.google.com/thumbnail?id=1dXPuXIEqDakgT0AgGQf1N8lW4oFMfxhS&sz=w2497" alt="model_compare.png" width="800"/>  
    The table shows that the ensemble model is the best performing one considering all the metrics used.

- **Ensemble Model Performance**  
    The results of `Ensemble Model` (our best-performing model):  
    - **Performance of Ensemble model on the Train vs Test Datasets**  
        <img src="https://drive.google.com/thumbnail?id=1hU4sztYClTHbAKVKxE3jMb5BLrSO7rYb&sz=w2004" alt="ensemble_model_train_test_results.png" width="350"/>  
        This shows that model is not overfitting.

    - **Performance of Component Models inside the Ensemble model** on the Test Dataset  
        <img src="https://drive.google.com/thumbnail?id=12BC8I9wlZQeIbw4uzqeFPaDmhtEKyld5&sz=w2287" alt="ensemble_model_results.png" width="400"/>  
        Both Model 1 and Model 2 has low performance, but combined model has a high performance.  

    - **Final Performance Report of Ensemble Model** on the Test Dataset  
        <img src="https://drive.google.com/thumbnail?id=16Y17BiD0nkOjtEbWr4yJbFV3qbye3DIV&sz=w2374" alt="ensemble_model_report.png" width="400"/>  
        <img src="https://drive.google.com/thumbnail?id=1Ec85vzbBU3NguWZPLFcdUG07m0MPI2NE&sz=w536" alt="ensemble_model_confusion_matrix.png" width="400"/>  

### **Improvements/Future Scope**  
Following are some of the ways which could improve the model in future:
- Expand the ensemble with more diverse models (e.g., deep metric learning, anomaly detection models).  
- Experiment with different loss functions like triplet loss in Siamese networks and log loss in neural networks for more control.  
- Apply more advanced preprocessing techniques to enhance data quality.

## **Code Documentation and Manual**

### **Environment Setup**  
The project utilizes **Conda** for managing dependencies, because of the ease and flexibility. You can use any other environment manager, but install the proper versions of the required libraries.  

Follow the steps below to set up your environment:  
1. Install **Miniconda**: Follow instructions [on the official page here](https://docs.anaconda.com/miniconda/miniconda-install/).  
2. Navigate to the project folder and run:  
   ```bash  
   conda env create -n your_env_name -f environment.yml  
   ```  
3. Activate the environment:
    ```bash  
   conda activate your_env_name  
   ```  
   This will create and activate the required environment with all necessary dependencies.

### **Project Structure**  
The project directory is structured as follows:  

```
Optimized-Ensemble-Learning-for-Classification  
├── dataset_original/  
│   ├── Test_20/  
│   └── Train_60/  
├── dataset_preprocessed/  
├── Model_Scripts/  
├── preprocessing_config/  
├── saves/  
│   └── save_final_ensemble_model/  
├── tf_logs/  
├── tf_models/  
├── environment.yml  
├── Final_Model_Predict.ipynb  
├── Final_Model_Predict_Script.py  
└── Final_Model_Train.ipynb  
```

Below are the details of each folder in the project:
```
Optimized-Ensemble-Learning-for-Classification
├── dataset_original
│   ├── Test_20 (Place the Test input and labels here)
│   └── Train_60 (Place the Train input and labels here)
├── dataset_preprocessed (Preprocessed Dataset will be saved here)
├── Model_Scripts (Contains the code for other Models that we have tried)
├── preprocessing_config (Configuration files of the preprocessing while training will be saved here)
├── saves (Contains Saves of different models and preprocessing-configuration)
│   └── save_final_ensemble_model (Our Final Ensemble Model is saved here and this is used to make predictions)
│       ├── preprocessing_config
│       ├── tf_logs
│       └── tf_models
├── tf_logs (Training Logs of the models will be saved here)
├── tf_models (Models will be saved here after training)
├── environment.yml (List of the requirements to run the model)
├── Final_Model_Predict.ipynb (Python Notebook for making predictions on a given dataset using our final ensemble model)
├── Final_Model_Predict_Script.py (Python Script for making predictions on a given dataset using our final ensemble model)
└── Final_Model_Train.ipynb (Python Notebook to train our final ensemble model on the training set)  
```  

**Note**: Add the training and testing dataset files (`X_Train_Data_Input.csv`, `Y_Train_Data_Target.csv`, `X_Test_Data_Input.csv`, `Y_Test_Data_Target.csv`) inside the directory `dataset_original/Train_60` and `dataset_original/Test_20` respectively.  

**Note**: In the `Saves` folder each individual folder (named after the model) contains three folders: `tf_models`, `tf_logs` and `preprocessing_config`. In case you want to do more testing, after training the model, its respective files will be stored in the outside folders: `tf_models`, `tf_logs` and `preprocessing_config`. You need to make a new folder inside `Saves` folder and copy these folders there, then you can make the prediction on the new model. Predict the new saved model, by changing the save folder name in the `Final_Model_Predict.ipynb` notebook or passing the new save folder path argument in the predict script `Final_Model_Predict_Script.py`. By default you should use our pretrained final model save `"save_final_ensemble_model"`.  

### **Training the Model**  
To recreate the training results, Model Training can be done using the `Final_Model_Train.ipynb` python notebook. This notebook will train the model on the given training dataset and give prediction results on the test dataset.  
1. Activate the environment:  
   ```bash  
   conda activate your_env_name  
   ```  
2. Run the Jupyter Notebook.
    ```bash
    jupyter notebook
    ```
    Inside Jupyter, navigate to project folder and open the `Final_Model_Train.ipynb` file.  
3. Before proceeding, you need to have the training and testing dataset files (`X_Train_Data_Input.csv`, `Y_Train_Data_Target.csv`, `X_Test_Data_Input.csv`, `Y_Test_Data_Target.csv`) inside the directory `dataset_original/Train_60` and `dataset_original/Test_20`. Or alternatively, you can change the paths in the notebook to your stored files.  
4. Run the individual cells one by one (if needed, make changes according to your need) to train the model. The code in the notebook is well commented.  

### **Making Predictions**  
You can make predictions using our pretrained model in two ways:

#### **1. Python Notebook**  
- Run the `Final_Model_Predict.ipynb` notebook.  
    ```bash  
    # Activate the environment
    conda activate your_env_name  

    # Run the Jupyter Notebook
    jupyter notebook  

    # Inside Jupyter, navigate to project folder and open the `Final_Model_Train.ipynb` file.
    ```
- Change the `x_test_csv_path='dataset_original/Test_20/X_Test_Data_Input.csv'` path to your testing input file. Or keep it the same if you want to test it on the testing dataset.    
- Change the `pred_out = 'predictions.csv'` if needed. Predictions by the model will be saved in this file location.  
- Run the individual cells one by one to make predictions. The code in the notebook is well commented.  
- Optionally, You can also see the results of the predictions by providing the true labels in the `y_test_csv_path = 'dataset_original/Test_20/Y_Test_Data_Target.csv'`  

#### **2. Python Script**  
- Activate the environment and navigate/change directory into the project folder:
    ```bash
    conda activate your_env_name
    ```  
- To only make predictions:  
Run the prediction script with the following command  
    ```bash  
    python Final_Model_Predict_Script.py --test_data_input "dataset_original/Test_20/X_Test_Data_Input.csv" --save_folder "saves/save_final_ensemble_model/" --output_csv "predictions.csv"
    ```  
    This makes predictions and saves them in the csv file.  

- To make predictions and compare with true labels:
Run the prediction script with the following command  
    ```bash  
    python Final_Model_Predict_Script.py --test_data_input "dataset_original/Test_20/X_Test_Data_Input.csv" --save_folder "saves/save_final_ensemble_model/" --output_csv "predictions.csv" --true_labels "dataset_original/Test_20/Y_Test_Data_Target.csv"
    ```  
    This makes the prediction and saves them in a csv file then compares them with the true labels to display the results.

- Arguments to the script:
    - --test_data_input = Path to the test dataset (CSV file)
    - --save_folder = Path to the saved model folder (models and preprocessing config files).
    - --output_csv = Path to save the predictions (CSV file).
    - --true_labels = Optional path to the true labels (target labels) for evaluation (CSV file).  

For further details, see the comments within the script and notebooks.

## **References**  
1. G. E. Hinton, R. R. Salakhutdinov, Reducing the Dimensionality of Data with Neural Networks. Science 313, 504-507 (2006). DOI: 10.1126/science.1127647  

2. H. He and E. A. Garcia, "Learning from Imbalanced Data," in IEEE Transactions on Knowledge and Data Engineering, vol. 21, no. 9, pp. 1263-1284, Sept. 2009, doi: 10.1109/TKDE.2008.239.  

3. Bromley, J., Guyon, I., LeCun, Y., Säckinger, E., & Shah, R. (1993). Signature Verification using a “Siamese” Time Delay Neural Network. In J. Cowan, G. Tesauro, & J. Alspector (Eds.), Advances in Neural Information Processing Systems (Vol. 6). Morgan-Kaufmann.  

4. Breiman, L. Random Forests. Machine Learning 45, 5–32 (2001).

5. Chen, Chao & Breiman, Leo. (2004). Using Random Forest to Learn Imbalanced Data. University of California, Berkeley.  

6. Butcher, B., & Smith, B. J. (2020). Feature Engineering and Selection: A Practical Approach for Predictive Models: by Max Kuhn and Kjell Johnson. Boca Raton, FL: Chapman & Hall/CRC Press, 2019, xv + 297 pp., $79.95(H), ISBN: 978-1-13-807922-9. The American Statistician, 74(3), 308–309.  

7. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.

8. R. Hadsell, S. Chopra and Y. LeCun, "Dimensionality Reduction by Learning an Invariant Mapping," 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06), New York, NY, USA, 2006, pp. 1735-1742, doi: 10.1109/CVPR.2006.100.  

9. Zhou, C., & Paffenroth, R. C. (2017). Anomaly Detection with Robust Deep Autoencoders. Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 665–674.

10. Baldi, P. (2012). Autoencoders, Unsupervised Learning, and Deep Architectures. In I. Guyon, G. Dror, V. Lemaire, G. Taylor, & D. Silver (Eds.), Proceedings of ICML Workshop on Unsupervised and Transfer Learning (Vol. 27, pp. 37–49). PMLR.  

11. S. Chopra, R. Hadsell and Y. LeCun, "Learning a similarity metric discriminatively, with application to face verification," 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), San Diego, CA, USA, 2005, pp. 539-546 vol. 1, doi: 10.1109/CVPR.2005.202.  