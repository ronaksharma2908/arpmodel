from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


# loading dataset
data = pd.read_csv("antibiotic_resistance.csv")

# extracting first 5 rows of the dataset
print(data.head())

# extracting column names of the dataset
print(data.columns)
print(data.isnull().sum())

for col in data.columns:
    print(col, data[col].unique()[:10])  
# encoding categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])
print(data.columns)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load your dataset (already done earlier)
# data = pd.read_csv("data/your_dataset.csv")

# Features (inputs) = all antibiotic columns
X = data[['IMIPENEM', 'CEFTAZIDIME', 'GENTAMICIN', 'AUGMENTIN', 'CIPROFLOXACIN']]

# Target (output) = Location (or whichever column you want to predict)
y = data['Location']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
sample = [[1, 0, 1, 0, 1]]  # Replace with actual values for IMIPENEM, CEFTAZIDIME, GENTAMICIN, AUGMENTIN, CIPROFLOXACIN
prediction = model.predict(sample)
print("Predicted Location:", prediction[0])








