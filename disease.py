import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import joblib
import streamlit as st


@st.cache_resource
def load_model_and_encoders():
    model = tf.keras.models.load_model('disease_model.h5')
    mlb = joblib.load('mlb_encoder.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, mlb, label_encoder


@st.cache_resource
def load_additional_data():
    precaution_data = pd.read_csv("/Users/jintubiswakarma/Desktop/ML/disease predictor/symptom_precaution.csv")
    description_data = pd.read_csv("/Users/jintubiswakarma/Desktop/ML/disease predictor/symptom_Description.csv")
    return precaution_data, description_data


precaution_data, description_data = load_additional_data()


model, mlb, label_encoder = load_model_and_encoders()


dataset_path = "/Users/jintubiswakarma/Desktop/ML/disease predictor/dataset.csv"
dataset = pd.read_csv(dataset_path)
dataset.fillna('', inplace=True)
dataset['Symptoms'] = dataset.iloc[:, 1:].apply(lambda x: [s for s in x if s], axis=1)

# Split dataset into training and testing
X = mlb.transform(dataset['Symptoms'])
y = label_encoder.transform(dataset['Disease'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate model
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
report = classification_report(y_test, predicted_classes, target_names=label_encoder.classes_, output_dict=True)
st.write(f"Model Precision: {report['macro avg']['precision']:.2f}")
st.write(f"Model Recall: {report['macro avg']['recall']:.2f}")
st.write(f"Model F1-score: {report['macro avg']['f1-score']:.2f}")

# Function to get precautions
def get_precautions(disease):
    row = precaution_data[precaution_data['Disease'] == disease]
    if not row.empty:
        return row.iloc[0, 1:].dropna().tolist()
    return ["No precautions available"]

# Function to get description
def get_description(disease):
    row = description_data[description_data['Disease'] == disease]
    if not row.empty:
        return row.iloc[0]['Description']
    return "No description available"

# Streamlit Web App
st.title("Disease Prediction System")

# Get user symptoms
symptoms = st.multiselect("Select Symptoms", mlb.classes_)

if st.button("Predict Disease"):
    if symptoms:
        # Transform symptoms
        input_data = mlb.transform([symptoms])
        prediction = model.predict(input_data)[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction)[-3:][::-1]
        top_3_diseases = label_encoder.inverse_transform(top_3_indices)
        top_3_probs = prediction[top_3_indices]

        st.subheader("Top Predictions:")
        for disease, prob in zip(top_3_diseases, top_3_probs):
            if prob > 0.2:  # Confidence threshold
                st.write(f"**{disease}** ({prob*100:.2f}% confidence)")
                st.write(f"Description: {get_description(disease)}")
                st.write("Precautions:")
                for precaution in get_precautions(disease):
                    st.write(f"- {precaution}")
    else:
        st.write("Please select at least one symptom.")
