# API-Deployment-for-Predicting-Long-Term-Deposit-Subscription

## Project Overview
This project aims to predict whether a bank customer will subscribe to a term deposit based on their demographic and transactional data. The model was developed and deployed as an API using FastAPI, allowing for real-time predictions. The project includes data preprocessing, model training, and deployment. Various test cases were conducted to ensure the accuracy and reliability of the prediction model.

### Dataset Description
The dataset used in this project contains various attributes related to bank customers that could potentially influence their decision to subscribe to a term deposit. Each record includes demographic details such as the customer’s age, job type, marital status, and education level. Financial information is also provided, such as whether the customer has defaulted on credit, has a housing or personal loan, and the last contact duration in seconds. Behavioral data includes the number of contacts made during the current and previous campaigns, the outcome of the previous campaign, and how many days have passed since the customer was last contacted. Additional information includes the month and day of the week when the last contact occurred, as well as the communication type (e.g., cellular or telephone). The target variable indicates whether the customer subscribed to a term deposit, with yes representing a subscription and no indicating otherwise.

### Step 1: Model Training
- The .ipynb file contains the code for data preprocessing, feature encoding, and training a machine learning model (e.g., Decision Tree) to predict term deposit subscriptions.
- The trained model is saved as dt_model.pkl, along with encoders for categorical features (enc_contact.pkl, enc_day.pkl, enc_month.pkl, OHE_encoder.pkl) and a scaler (scaler.pkl).

### Step 2: Prediction Code
- dt_predict.py contains the code to load the saved model and make predictions. This script can be integrated with deployment frameworks to handle incoming prediction requests.

### Step 3: Deployment with FastAPI
- The project is deployed as an API using FastAPI. After setting up FastAPI, you can run the server and send POST requests to make predictions.
- To start the FastAPI server, run:
  uvicorn dt_predict:app --reload
  This will host the API locally, allowing you to send test cases through POST requests.

## Project Structure
├── .ipynb       # Model training and feature engineering

├── dt_predict.py        # Prediction script for FastAPI deployment

├── dt_model.pkl         # Trained model file

├── enc_contact.pkl      # Categorical feature encoder for 'contact'

├── enc_day.pkl          # Categorical feature encoder for 'day_of_week'

├── enc_month.pkl        # Categorical feature encoder for 'month'

├── OHE_encoder.pkl      # One-hot encoder for categorical features

├── scaler.pkl           # Scaler for feature normalization

└── README.md            # Project documentation

Testing
The deployment was tested with three cases, as outlined below:
1. Test Case 1: Predicts "no" for a single customer instance.
2. Test Case 2: Predicts "yes" for a different customer, which matches the actual label.
3. Test Case 3: Another "yes" prediction, successfully matching the actual label.
These tests confirm that the model performs well in identifying customers likely to subscribe or not. Please refer to Usecase_ScreenShot_and_Explanation.pdf for detailed explanations and screenshots of each test case.

Author
Davin Edbert Santoso Halim
