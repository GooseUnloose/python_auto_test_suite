from auto_tester import *
from model_class import *
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

if __name__ == '__main__':
    
    df = pd.read_csv("test.csv")
    df['gender'] = df['gender'].replace(['Male','Female'],[0,1])
    
    df.head()
    
    test = Testing_Environment(df)
    
    model = Model_Class(LogisticRegression(),{})
    
    test.auto_set_cpu_cores()
    
    test.parallel_layer_distribution(model)
    
    test.set_X('gender')
    test.set_y('gender')

    output = test.train_test(model,log_results=True)
    
    
    
    
    