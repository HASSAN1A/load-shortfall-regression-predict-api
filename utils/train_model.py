"""
    Simple file to create a Sklearn model for deployment in our API
    Author: Explore Data Science Academy
    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.
"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Madrid_wind_speed', 'Seville_clouds_all', 'Bilbao_wind_deg',
                                        'Seville_rain_1h', 'Barcelona_rain_3h', 'Valencia_snow_3h',
                                        'Bilbao_pressure', 'Bilbao_weather_id', 'Valencia_temp_min']]


# Fit model
model = RandomForestRegressor(n_estimators=100, random_state=0 , n_jobs=1)
print ("Training Model...")
model.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/load_shortfall_simple_lm_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(model, open(save_path,'wb'))
