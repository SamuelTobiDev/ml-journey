import numpy as np
import pandas as pd

# [age, fare, class, sex,  family size]
jack = np.array([20, 7.25,3,0,1])
rose = np.array([17, 71.28,1,1,2])

print("Jack", jack)
print("Rose", rose)
print("Jack shape", jack.shape)
print("Rose shape", rose.shape)

# scale jack's features by 2
jack_scaled = jack * 2
print("Jack scaled", jack_scaled)

combined = jack + rose
print("Combined features", combined)


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df['Age'] = df['Age'].fillna(df['Age'].median())

# Each row of your DataFrame is a vector

first_passenger = df.iloc[0][['Age','Fare','Pclass']].values
print("First passenger vector", first_passenger)
print("First passenger shape", first_passenger.shape)
print("type of first passenger", type(first_passenger))