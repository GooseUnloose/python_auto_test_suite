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
    
    test.auto_set_cpu_cores()
    
    test.set_X('gender')
    test.set_y('gender')
    
    model = Model_Class(DecisionTreeClassifier(),{'max_depth': [10, 20],'min_samples_split': [2, 5],})

    output = test.train_test(model,log_results=True)
    
    
    