import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
df = pd.read_csv("crop_yield.csv")
print(df.head())
  
print(df.columns)

# Map prices and costs
market_price_dict = {
    'Arecanut': 250000,        
    'Arhar/Tur': 6200,
    'Castor seed': 5300,      
    'Coconut ': 11000,         
    'Cotton(lint)': 52000,    
}
cost_cultivation_dict = {
    'Arecanut': 244365,         
    'Arhar/Tur': 71370,
    'Castor seed': 40000,
    'Coconut ': 155836,
    'Cotton(lint)': 82200,
}
df['Market_Price'] = df['Crop'].map(market_price_dict)
df['Cost_of_Cultivation'] = df['Crop'].map(cost_cultivation_dict)
df['Revenue'] = df['Yield'] * df['Market_Price']
df['Profit'] = df['Revenue'] - df['Cost_of_Cultivation']
df['ROI'] = (df['Profit'] / df['Cost_of_Cultivation']) * 100
df_clean = df.dropna(subset=['ROI']).copy()
# Label encode categorical features
le_crop = LabelEncoder()
df_clean['Crop_encoded'] = le_crop.fit_transform(df_clean['Crop'])
le_state = LabelEncoder()
df_clean['State_encoded'] = le_state.fit_transform(df_clean['State'])
le_season = LabelEncoder()
df_clean['Season_encoded'] = le_season.fit_transform(df_clean['Season'])
# Features and target
features = ['Crop_encoded', 'State_encoded', 'Season_encoded', 'Area',
            'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield', 'Revenue']
X = df_clean[features]
y = df_clean['ROI']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

RandomForestRegressor
?i
RandomForestRegressor(random_state=42)
# Evaluation
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Create a DataFrame with actual vs predicted
results_df = X_test.copy()
results_df['Actual_ROI'] = y_test.values
results_df['Predicted_ROI'] = y_pred

results_df = results_df.merge(df_clean[['Crop', 'State', 'Season', 'Crop_encoded', 'State_encoded', 'Season_encoded']],
                              on=['Crop_encoded', 'State_encoded', 'Season_encoded'],
                              how='left')
# Save to CSV
results_df.to_csv("predicted_results_with_names.csv", index=False)
# Save model
joblib.dump(model, "roi_predictor.pkl")
