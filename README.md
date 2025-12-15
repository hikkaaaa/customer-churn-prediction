The dataset is taken from kaggle's telco customer churn prediction 

Three models were trained: Random Forest, Decision Classifier, XGBoost

Among these, Random Forest Classifier showed the best accuracy

Using this model:
1) Upload the dataset into the google colab "customer_churn_prediction.ipynb"
2) Run all
3) You will have two pickle files in your google colab files: "encoders.pkl" and "customer_churn_model.pkl"
4) Download the files and use it for your predictions:
   The code below is the example of the way how prediction works, this example is also shown at the very end of the ipynb file.
   By changing some parts of the code according to your own needs, you can use the model for predicting customer churn prediction

with open("customer_churn_model.pkl", "rb") as f:
  model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["feature_names"]

input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

input_data_df = pd.DataFrame([input_data])

with open("encoders.pkl", "rb") as f:
  encoders = pickle.load(f)

#encode categorical features using the saved encoders
for column, encoder in encoders.items():
  input_data_df[column] = encoder.transform(input_data_df[column])

#make predictions using the loaded model
prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

print(prediction)

#results
print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Prediction Probability: {pred_prob}")
