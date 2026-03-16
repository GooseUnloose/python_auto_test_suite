class Model_Class:
    
    def __init__(self,model,params:dict):
        self.model = model
        self.params = params
        
        self.model_name = "".join(char for char in model.__repr__() if char.isalpha())
        
    def get_params(self):
        return self.params
        
    def get_model(self):
        return self.model
    
    def name(self):
        return self.model_name
    
    #Model agnostic methods for predicting and fitting
    def predict(self,X_test):
        return self.model.predict(X_test)
        
    def fit(self,X_train, y_train):
        return self.model.fit(X_train,y_train)
        
    
    