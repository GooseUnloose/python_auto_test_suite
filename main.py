from auto_tester import *
from model_class import *
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

if __name__ == '__main__':
    
    df = pd.read_csv("test.csv")
    df['gender'] = df['gender'].replace(['Male','Female'],[0,1])
    
    df.head()
    
    test = Testing_Environment(df)
    
    test.set_X('gender')
    test.set_y('gender')
    
    
    model = Model_Class(DecisionTreeClassifier(),{'max_depth': [10, 20],'min_samples_split': [2, 5],})


    X_train, X_test, y_train, y_test = train_test_split(test.get_X(),test.get_y())
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    
    
    
    