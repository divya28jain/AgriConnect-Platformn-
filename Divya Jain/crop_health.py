import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
df = pd.read_csv("synthetic_crop_disease_dataset.csv")
df = df.drop(columns=['image_id'])
le_crop = LabelEncoder()
df['crop_type'] = le_crop.fit_transform(df['crop_type'])
le_disease = LabelEncoder()
df['disease_label_encoded'] = le_disease.fit_transform(df['disease_label'])
X = df.drop(columns=['disease_label', 'disease_label_encoded'])
y = df['disease_label_encoded']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
report = classification_report(y_test, y_pred, target_names=le_disease.classes_, output_dict=True)

results_df = pd.DataFrame({
    'Actual_Label': le_disease.inverse_transform(y_test),
    'Predicted_Label': le_disease.inverse_transform(y_pred)
})
results_df.to_csv('crop_disease_predictions.csv', index=False)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report.csv', index=True)

print("\nResults saved to 'crop_disease_predictions.csv' and 'classification_report.csv'")

import joblib
joblib.dump(clf, "crop_disease_model.pkl")
 
import joblib
import os

filename = "crop_disease_model.pkl"

# 1. Check if file exists and its size
if os.path.exists(filename):
    print(f"File exists: {filename}")
    print(f"Size: {os.path.getsize(filename)} bytes")
else:
    print("File not found.")

# 2. Try loading it just to validate
try:
    obj = joblib.load(filename)
    print("✅ File is a valid joblib model.")
    print("Object type:", type(obj))
except Exception as e:
    print("❌ File is not valid or corrupted.")
    print("Error:", e)
