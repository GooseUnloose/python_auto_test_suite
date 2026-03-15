class Model_Class:
    
    def __init__(self,model,params:dict):
        self.model = model
        self.params = params
        
    def get_params(self):
        return self.params
        
    def get_model(self):
        return self.model
    
    
    #Model agnostic methods for predicting and fitting
    def predict(self,X_test):
        return self.model.predict(X_test)
        
    def fit(self,X_train, y_train):
        return self.model.fit(X_train,y_train)
        
    
    