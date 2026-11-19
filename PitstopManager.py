from urllib.request import urlopen
import certifi
import json
import ssl
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class OpenF1API:
    """Helper class for interacting with OpenF1 API"""
    
    def __init__(self):
        self.base_url = "https://api.openf1.org/v1"
    
    def get_data(self, endpoint, params=None):
        """Fetch data from OpenF1 API"""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
    
    def get_dataframe(self, endpoint, params=None):
        """Fetch data and return as pandas DataFrame"""
        data = self.get_data(endpoint, params)
        if data:
            return pd.DataFrame(data)
        return pd.DataFrame()

def apply_weights(x_poly, w):

    return np.dot(x_poly,w)
def polynomial_linear_regression(x_poly, ys):
    model = LinearRegression(fit_intercept=False)

    model.fit(x_poly, ys)

    return model.coef_
def polynomial_features(x):

    polynomial_features= PolynomialFeatures(degree=3)
    x_poly = polynomial_features.fit_transform(x[:, np.newaxis])
    return x_poly


api = OpenF1API()
print("✅ API client initialized!")

driver_numbers = [63, 12, 16, 44, 3, 6, 1, 81, 30, 41, 10, 43, 87, 51, 27, 5, 55, 23, 14, 18, 11, 77, 31]

race_df = api.get_dataframe('laps', {
    'session_key': '11326'
})

stint_df = api.get_dataframe('stints',{
    'session_key': '11326'
})

race_df = race_df[race_df['is_pit_out_lap'] == False]

temp_df = race_df.iloc[0:0].copy()
for driver in driver_numbers:
    driver_laps = race_df[race_df['driver_number'] == driver]
    driver_laps = driver_laps.sort_values('lap_number').copy()

    driver_laps['rolling_median'] = driver_laps['lap_duration'].rolling(window=5, center=True).median()

    outlier_threshold = 2
    is_valid_lap = (driver_laps ['lap_duration'] - driver_laps['rolling_median'] <= outlier_threshold) | driver_laps['rolling_median'].isna()

    clean_laps = driver_laps[is_valid_lap]
    if not clean_laps.empty:
        clean_laps = clean_laps.drop(columns = ['rolling_median'])
        temp_df = pd.concat([temp_df,clean_laps], ignore_index = True)

temp_df = temp_df.drop(index=0)
temp_df = temp_df.reset_index(drop = True)

mean_lap_time = temp_df.loc[:, 'lap_duration'].mean()
temp_df = temp_df[temp_df['lap_duration'] < (1.02 * mean_lap_time)]
temp_df = temp_df[temp_df['lap_duration'] > (0.98 * mean_lap_time)]

cleaned_race_df = temp_df.copy()

size = cleaned_race_df.shape



soft_tyre_data = []
medium_tyre_data = []
hard_tyre_data = []

for i in range(size[0]):
    lap_time = cleaned_race_df.iloc[i].loc['lap_duration'] 
    fuel_adjusted_lap_time = lap_time + (0.03 * (cleaned_race_df.iloc[i].loc['lap_number'] - 1))

    index = stint_df.index[
        (stint_df['driver_number'] == cleaned_race_df.iloc[i].loc['driver_number'])
        & (stint_df['lap_start'] <= cleaned_race_df.iloc[i].loc['lap_number'])
        & (stint_df['lap_end'] >= cleaned_race_df.iloc[i].loc['lap_number'])
                           ].start
    tyre_wear = (stint_df.loc[index, 'tyre_age_at_start']
    + cleaned_race_df.iloc[i].loc['lap_number'] 
    - stint_df.loc[index, 'lap_start'])

    if (stint_df.loc[index, 'compound']) == "SOFT":
        soft_tyre_data.append([tyre_wear, fuel_adjusted_lap_time])
    elif (stint_df.loc[index, 'compound']) == 'MEDIUM':
        medium_tyre_data.append([tyre_wear, fuel_adjusted_lap_time])
    elif (stint_df.loc[index, 'compound']) == 'HARD':
        hard_tyre_data.append([tyre_wear, fuel_adjusted_lap_time])

soft_tyre_data = np.array(soft_tyre_data)
medium_tyre_data = np.array(medium_tyre_data)
hard_tyre_data = np.array(hard_tyre_data)

xs = medium_tyre_data[:, 0]
ys = medium_tyre_data[:, 1]

x_smooth = np.linspace(xs.min(), xs.max(), 200)
x_smooth_poly = polynomial_features(x_smooth)
x_poly = polynomial_features(xs) # get the polynomial features

w = polynomial_linear_regression(x_poly, ys) # find the fitted weights

y_pred = apply_weights(x_smooth_poly, w)

plt.figure(figsize=(10, 6))
plt.plot(xs, ys, marker='o', linestyle='None', color='blue', alpha=0.5, label='Actual Laps (Medium)')
plt.plot(x_smooth, y_pred, color='red', linewidth=2.5, label='Degree 3 Polynomial Fit')

plt.title('Medium Tyre Degradation')
plt.xlabel('Tyre Age (Laps)')
plt.ylabel('Fuel-Adjusted Lap Time (Seconds)')
plt.grid(True)
plt.legend()
plt.show()

print(f"y = {w[3]:.6f}x^3 + {w[2]:.6f}x^2 + {w[1]:.6f}x + {w[0]:.6f}")
#ssl_context = ssl.create_default_context(cafile=certifi.where())

#response = urlopen('https://api.openf1.org/v1/laps?session_key=11400&driver_number=63&lap_number=8', context=ssl_context)
#data = json.loads(response.read().decode('utf-8'))
#print(data)