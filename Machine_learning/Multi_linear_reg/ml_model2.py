import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
df=pd.read_csv("trial_data_nv2.csv")
median_bedroom=math.floor(df.bedroom.median())
df.bedroom=df.bedroom.fillna(median_bedroom)
reg=linear_model.LinearRegression()
reg.fit(df[['area','bedroom','age']],df.price)
print(df)

#let's do some predictions on model

#==============1===============
#when area=3000,bedroom=3,age=15 
print("when area=3000,bedroom=3,age=15 = ",reg.predict([[3000,3,15]]))

#==============2===============
#when area=2500,bedroom=4,age=5 
print("when area=2500,bedroom=4,age=5 ",reg.predict([[2500,4,5]]))


