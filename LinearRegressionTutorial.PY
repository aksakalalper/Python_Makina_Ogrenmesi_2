from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


#class created
class LinearRegressionModels():
    def __init__(self):
        self.lm=LinearRegression()
    #simple linear regression model
    def linearRegression(self):
        #reading files
        file="Student_Marks.csv"
        data=pd.read_csv(file)
        df=pd.DataFrame(data)
        y=df['Marks'] #we want to predict that 
        x=df.drop('Marks',axis=1)
        model=self.lm.fit(x,y) #learning process
        print("tahmin edilen değer: ",model.predict([[5,4]]))
        print("modelin skoru: ",model.score(x,y))
        print("modelin coef sabiti: ",model.coef_)
        print("modelin intercept katsayısı: ",model.intercept_)
    #preparing data for learning process
    def preProcessingTrainTestSplit(self):
        #reading files
        file="Audi_A1_listings.csv"
        data=pd.read_csv(file)
        df=pd.DataFrame(data)
        df=df.drop(['index', 'Type' ,'Number_of_Owners', 'href', 'MileageRank', 'PriceRank', 'PPYRank', 'Score'],axis=1)
        df['Engine']=df['Engine'].str.replace('L',' ')
        df['Engine']=pd.to_numeric(df['Engine'])
        df=pd.get_dummies(df,columns=['Transmission','Fuel'])
        y=df['Price(£)']
        x=df.drop('Price(£)',axis=1)
        model=self.lm.fit(x,y)
        print(df.info(),"\n",df.describe(),"\n",df.columns)
        print(df.head(5))
        print("model skoru", model.score(x,y))
        #train-test-split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
        trainModel=self.lm.fit(x_train,y_train)
        print("train model skoru: ", trainModel.score(x_test,y_test))
    def mseRMSEmae(self):
        #reading files
        file="insurance.csv"
        data=pd.read_csv(file)
        df=pd.DataFrame(data)
        print(df.info(),"\n",df.describe(),"\n",df.columns)
        print(df.head(5))
        
linearregressionmodels=LinearRegressionModels()
linearregressionmodels.mseRMSEmae()


    