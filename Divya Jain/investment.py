import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_excel("preprocessed_agri_investment_data.csv.xlsx")

# Keep a copy of the original (for human-readable output later)
original_df = df.copy()
# Encode categorical features
label_encoders = {}
for col in ['Crop_Name', 'State']:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode the target variable
target_encoder = LabelEncoder()
df['Investment_Recommendation_encoded'] = target_encoder.fit_transform(df['Investment_Recommendation'])

# Prepare features and target
feature_cols = ['Crop_Name_encoded', 'State_encoded', 'Estimated_Revenue', 'Estimated_Profit',
                'Calculated_ROI_%', 'Risk_Index']
X = df[feature_cols]
y = df['Investment_Recommendation_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)   
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Decode predictions
y_pred_labels = target_encoder.inverse_transform(y_pred)
y_actual_labels = target_encoder.inverse_transform(y_test)

# Match back with original values for interpretability
result_df = X_test.copy()
result_df['Actual_Label'] = y_actual_labels
result_df['Predicted_Label'] = y_pred_labels

# Add human-readable columns back from the original dataframe
decoded_crop = label_encoders['Crop_Name'].inverse_transform(X_test['Crop_Name_encoded'])
decoded_state = label_encoders['State'].inverse_transform(X_test['State_encoded'])
result_df['Crop_Name'] = decoded_crop
result_df['State'] = decoded_state

# Reorder columns
result_df = result_df[['Crop_Name', 'State', 'Estimated_Revenue', 'Estimated_Profit',
                       'Calculated_ROI_%', 'Risk_Index', 'Actual_Label', 'Predicted_Label']]

# Save to CSV
result_df.to_csv("final_agri_predictions.csv", index=False)
print("✅ Final result with original labels saved as 'final_agri_predictions.csv'")

# Save the model
joblib.dump(model, "investment_model.pkl")
print("✅ Model saved")
