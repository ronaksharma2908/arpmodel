import streamlit as st
import pandas as pd
df = pd.read_csv("antibiotic_resistance.csv")   
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score   
st.markdown("### Welcome! 👋")
st.write("Welcome! This app predicts antibiotic resistance location using ML and suggests effective antibiotics.")


# Load dataset
data = pd.read_csv("antibiotic_resistance.csv")

# Features and target
X = data[['IMIPENEM', 'CEFTAZIDIME', 'GENTAMICIN', 'AUGMENTIN', 'CIPROFLOXACIN']]
y = data['Location']
for col in X.columns:
    print(f"{col}: min={df[col].min()}, max={df[col].max()}")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
print("Predictions:", le.inverse_transform(y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Streamlit UI
st.title("Antibiotic Resistance Location Predictor")

import streamlit as st

imipenem = st.number_input("Enter IMIPENEM value", min_value=0, max_value=40, step=1)
ceftazidime = st.number_input("Enter CEFTAZIDIME value", min_value=0, max_value=32, step=1)
gentamicin = st.number_input("Enter GENTAMICIN value", min_value=0, max_value=30, step=1)
augmentin = st.number_input("Enter AUGMENTIN value", min_value=0, max_value=35, step=1)
ciprofloxacin = st.number_input("Enter CIPROFLOXACIN value", min_value=0, max_value=35, step=1)

if st.button("Predict Location"):
    sample = [[imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin]]
    prediction = model.predict(sample)
    st.write("Predicted Location:", prediction[0])
    st.success(f"Predicted Location: {le.inverse_transform(prediction)[0]}")
    st.write("Input values:", sample) 

def predict_location(sample):
   # Here you would use the model to predict the location based on the input sample
    return "Predicted location: App is running"

def run_app():
    sample = "test_sample"  # Replace with actual sample data
    prediction = predict_location(sample)
    return prediction
inputs = {
    "IMIPENEM": imipenem,
    "CEFTAZIDIME": ceftazidime,
    "GENTAMICIN": gentamicin,
    "AUGMENTIN": augmentin,
    "CIPROFLOXACIN": ciprofloxacin
}

suggested = [ab for ab, val in inputs.items() if val == 0]

st.markdown("### Decision Support")
st.write("Suggested Antibiotics (Sensitive):", suggested)


if __name__ == "__main__":
    result = run_app()
    print(result)
    
# Save model after training
joblib.dump(model, "arpmodel.pkl")
joblib.dump(le, "label_encoder.pkl")


# Later, load it in app.py
model = joblib.load("arpmodel.pkl")
le = joblib.load("label_encoder.pkl")
sample = [[imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin]]
prediction = model.predict(sample)
st.success(f"Predicted Location: {le.inverse_transform(prediction)[0]}")

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
    st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
import networkx as nx
import matplotlib.pyplot as plt

st.markdown("### Resistance Gene Network")

G = nx.Graph()
antibiotics = ["IMIPENEM","CEFTAZIDIME","GENTAMICIN","AUGMENTIN","CIPROFLOXACIN"]
for ab in antibiotics:
        G.add_node(ab)

    # Example: connect antibiotics if their values are correlated
corr = df[antibiotics].corr()
for i in antibiotics:
        for j in antibiotics:
            if i != j and corr.loc[i, j] > 0.7:  # threshold for strong correlation
                G.add_edge(i, j)

    # Draw graph
fig, ax = plt.subplots()
nx.draw(G, with_labels=True, node_color="lightblue", node_size=2000, font_size=12, ax=ax)
st.pyplot(fig)
st.header("Antibiotic Network Visualization")
st.pyplot(fig)


# Example: connect nodes if both resistant in same sample
# for _, row in df.iterrows():
#     resistant = [ab for ab in antibiotics if row[ab] == 1]
#     for i in range(len(resistant)):
#         for j in range(i+1, len(resistant)):
#             G.add_edge(resistant[i], resistant[j])

# plt.figure(figsize=(6,6))
# nx.draw(G, with_labels=True, node_color="lightblue", font_weight="bold")
# st.pyplot(plt)
