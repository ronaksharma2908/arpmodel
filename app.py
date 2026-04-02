import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score   #

# Load dataset
data = pd.read_csv("antibiotic_resistance.csv")

# Features and target
X = data[['IMIPENEM', 'CEFTAZIDIME', 'GENTAMICIN', 'AUGMENTIN', 'CIPROFLOXACIN']]
y = data['Location']

# Train model (you could also load a pre-trained model with joblib)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
# Evaluate accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Streamlit UI
st.title("Antibiotic Resistance Location Predictor")

imipenem = st.number_input("IMIPENEM", min_value=0, max_value=1, step=1)
ceftazidime = st.number_input("CEFTAZIDIME", min_value=0, max_value=1, step=1)
gentamicin = st.number_input("GENTAMICIN", min_value=0, max_value=1, step=1)
augmentin = st.number_input("AUGMENTIN", min_value=0, max_value=1, step=1)
ciprofloxacin = st.number_input("CIPROFLOXACIN", min_value=0, max_value=1, step=1)

if st.button("Predict Location"):
    sample = [[imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin]]
    prediction = model.predict(sample)
    st.write("Predicted Location:", prediction[0])
def predict_location(sample):
   # Here you would use the model to predict the location based on the input sample
    return "Predicted location: App is running"

def run_app():
    sample = "test_sample"  # Replace with actual sample data
    prediction = predict_location(sample)
    return prediction

if __name__ == "__main__":
    result = run_app()
    print(result)
    
# Save model after training
joblib.dump(model, "arpmodel.pkl")

# Later, load it in app.py
model = joblib.load("arpmodel.pkl")
import matplotlib.pyplot as plt
import streamlit as st

# After training your model
importance = model.feature_importances_

# Plot feature importance
fig, ax = plt.subplots()
ax.bar(X.columns, importance)
ax.set_title("Feature Importance")
ax.set_ylabel("Importance Score")

# Show plot in Streamlit
st.pyplot(fig)
st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
uploaded_file = st.file_uploader("Upload a CSV", type="csv")
if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    predictions = model.predict(new_data)
    st.write(predictions)



