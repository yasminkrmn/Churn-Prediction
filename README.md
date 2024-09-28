# Customer Churn Prediction Project

### Overview
This project aims to predict customer churn for a bank using a deep learning model. By analyzing customer data, we can identify potential churners and help the bank take preventive measures to retain them.

#### Table of Contents
* Features
* Technologies Used
* Data Collection
* Data Preprocessing
* Exploratory Data Analysis (EDA)
* Model Development
* MLflow Integration
* User Interface
* Installation
* Usage
* Contributing

#### Features
Predict customer churn using key metrics such as Credit Score, Age, Tenure, Balance, and more.

User-friendly web interface built with Streamlit for easy interaction and data input.

Data storage options: SQLite for saving user inputs and predictions.

Model performance evaluation and logging for improved accuracy.

Integration with MLflow for tracking experiments and managing the machine learning lifecycle.

#### Technologies Used
* Python
* Streamlit
* Pandas
* Keras (TensorFlow)
* Scikit-learn
* SQLite (for data storage)
* MLflow (for experiment tracking)
* Custom exception handling and logging utilities


#### Data Collection
The dataset used for training the model includes various customer features, such as:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Credit Card Ownership
* Active Member Status
* Estimated Salary


#### Data Preprocessing
* Data Splitting: 
  * The dataset was split into training and testing sets to prevent data leakage, ensuring that the model is evaluated on unseen data.

* Scaling and Encoding:
  * RobustScaler was used to scale numerical features, which is less sensitive to outliers.
  * OneHotEncoder was employed to convert categorical features (e.g., Geography, Gender) into a numerical format.

* Pipelines: 
  * Scikit-learn pipelines were utilized to streamline the preprocessing steps, maintaining the order of operations and improving code readability.

#### Exploratory Data Analysis (EDA)
Conducted statistical analyses to uncover patterns, trends, and relationships within the data.
Visualizations and summary statistics were created to provide insights into customer behavior and characteristics.

#### Model Development
Built a deep learning model using Keras.
The model is trained to classify whether a customer is likely to churn based on input features.
Implemented a prediction pipeline for making predictions on new user inputs.

#### MLflow Integration
* Experiment Tracking: Log and track model parameters, metrics, and artifacts using MLflow.
* Model Registry: Register models and manage different versions for reproducibility and deployment.
* Visualization: Use MLflow's UI to visualize the training process, compare runs, and evaluate model performance.

  
How to Use MLflow
Run the MLflow tracking server:

      mlflow ui

Open your web browser and go to http://localhost:5000 to access the MLflow UI.

View and compare different model training runs, parameters, and performance metrics.

#### User Interface
The web application built with Streamlit allows users to:

* Input customer data through sliders, select boxes, and number inputs.
* Submit the data to receive a churn prediction.
* View the prediction result and save their input data to a CSV file or SQLite database.

#### Installation
To set up the project locally, follow these steps:

Clone the repository:

    git clone https://github.com/yasminkrmn/Churn-Prediction.git


Install the required packages:

    pip install -r requirements.txt

Run the Streamlit application:

        streamlit run app.py

(Optional) To use MLflow, ensure MLflow is installed and run the following command in a separate terminal:

    mlflow ui

Usage

1. Open your web browser and go to http://localhost:8501 to access the application.

2. Fill in the customer details using the provided inputs.
3. Click the "Predict" button to see if the customer is likely to churn.
4. Optionally, the user input data will be saved to a specified file or database for further analysis.

#### Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please fork the repository and submit a pull request.