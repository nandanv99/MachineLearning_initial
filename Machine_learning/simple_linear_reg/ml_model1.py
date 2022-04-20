from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("trial_data_nv.csv")
print(df)
plt.xlabel('areas')
plt.ylabel('price')
#plt.scatter(df.areas,df.price)

reg=linear_model.LinearRegression()
reg.fit(df[['areas']],df.price)

#y=mx+c 
#finding m 
print("slope of whole data is :",reg.coef_)
#finding c
print("intercept is :",reg.intercept_)
lst=[3300,5000,7000,6700]
lst_price=[]
for i in lst:
    lst_price.append(reg.predict([[i]]))
    
plt.scatter(df.areas,df.price,color='green')
plt.scatter(lst,lst_price,color='red')
plt.plot(df.areas,reg.predict(df[['areas']]),color="blue")