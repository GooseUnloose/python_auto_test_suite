import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from model_class import Model_Class as model_class

import os,math,psutil


from sklearn import datasets

def sum_parameter_configurations(parameters:dict):
    
    parameter_sum = 1
    
    for parameter in parameters.values():
        parameter_sum = parameter_sum * len(parameter)
    
    return parameter_sum
    

#Returns a list of core utilisation for each layer of parallelism
def parallel_layer_distribution(df:pd.DataFrame, 
                                parameters:dict, 
                                cv:int, 
                                cpu_cores:int = 0,
                                cpu_auto:bool = False, 
                                layer_count: int = 3, 
                                core_overhead_factor:float = 3.0) -> list[int]:
    '''Determines parellisation layer distribution
    
    keyword arguments:
    
    df -- dataframe for testing
    cpu_cores -- number of CPU Cores the system has (default = 0)
    '''
    
    #Core decision logic
    
    if cpu_cores > 0:
        cpu_auto = False
    
    if cpu_auto == True:
        cpu_cores = os.cpu_count()
        
    if cpu_cores == 0:
        return [0,0,0]

        
    
    #Error handling
    if type(df) != pd.DataFrame:
        raise ValueError("df Must be a pd.DataFrame object")
    
    
    #One core is reserved for the device's operating system, whilst the others are listed as avaliable cores for parallelsation
    total_cpu_count = cpu_cores
    avaliable_cpu_count = total_cpu_count - 1
    
    
    #Parameter tuning layer
    parameter_permutations = sum_parameter_configurations(parameters)
    
    parameter_workers = min(avaliable_cpu_count,parameter_permutations)
    parameter_workers = memory_safety('',3.0,parameter_workers)
    cores_per_parameter = math.floor(avaliable_cpu_count / parameter_workers)
    
    
    #Fold layer
    fold_workers = min(cv,cores_per_parameter)
    fold_workers = memory_safety('',3.0,fold_workers)
    cores_per_fold = math.floor(cores_per_parameter / fold_workers)
    
    
    #Model layer
    if cores_per_fold >= 2:
        model_workers = cores_per_fold
        
    else:
        model_workers = 1
    
    
    return [parameter_workers,fold_workers,model_workers]



def memory_safety(df:pd.DataFrame,memory_overhead:float, worker_count:int) -> int:
    
    try: 
        dataframe_memory = os.path.getsize(df)
        
    except FileNotFoundError:
        
        dataframe_memory = 1
        
        
    
    
    estimated_memory_per_worker = dataframe_memory * memory_overhead
    avaliable_memory = psutil.virtual_memory().available / 1.e6

    max_workers = math.floor(avaliable_memory / estimated_memory_per_worker)
    actual_workers = min(worker_count,max_workers)
    
    if actual_workers < 2:
        actual_workers = 1
    
    return actual_workers


#
def auto_tester(X:pd.DataFrame, 
                y:pd.DataFrame, 
                model:model_class, 
                cv_folds: int, 
                cpu_cores:int,
                random_seed: int,
                stratify:bool,
                shuffle:bool,
                cpu_auto:bool = False,
                ): 
    
    parallel_layers = parallel_layer_distribution(pd.DataFrame.join(X,y),model.params,cv_folds,cpu_cores,cpu_auto)
    X_train,X_test, y_train, y_test = train_test_split(X, y, shuffle = shuffle, stratify = stratify, random_state = random_seed)
    
    GridSearchCV(model.model, model.params, n_jobs=parallel_layers[0], cv=cv_folds)
    
    
    
params = {
    'test1':[1,2],
    'test2':[1,2],
}

sum_parameter_configurations(params)
parallel_layer_distribution(pd.DataFrame(),params,cv=2,cpu_auto=False,cpu_cores=65)


class Testing_Environment:
    
    def __init__(self, dataframe:pd.DataFrame, cv_folds:int = 5, hp_type = GridSearchCV, cpu_cores:int = 4):
        self.dataframe = dataframe
        self.cv_folds = cv_folds
        self.hp_type = hp_type
        self.cpu_cores = cpu_cores
    
        #Attributes to be defined later
        self.X = None
        self.y = None
        self.parallel_layers = [1,1,1]
        self.dataframe_file_size = 0
    
        
    #Setters
    def set_hp_type(self,hp_type):
        self.hp_type = hp_type
    
    def set_dataframe_file_size(self,file_path:str):
        try: 
            self.dataframe_file_size = os.path.getsize(file_path)
    
        except FileNotFoundError:
            self.dataframe_file_size = 1
            
    def set_X(self,column:str):
        self.X = self.dataframe.drop(column,axis=1)
        
    def set_y(self,column:str):
        self.y = self.dataframe[column]
        
    def set_cpu_cores(self,cores:int):
        self.cpu_cores = cores
    
    def auto_set_cpu_cores(self):
        self.cpu_cores =  os.cpu_count()
    
    
    #Getters
    def get_X(self):
        return self.X
    
    def get_y(self):
        return self.y
    
    def get_cpu_cores(self):
        return self.cpu_cores
    
    def get_dataframe_file_size(self):
        return self.dataframe_file_size
    
    
    def get_hp_type(self):
        return self.hp_type
    
    #Parameter permutation summation
    def sum_parameter_configurations(self, model:model_class):
        parameter_sum = 1
    
        for parameter in model.get_params().values():
            parameter_sum = parameter_sum * len(parameter)
    
        return parameter_sum
    
    
    #Memory safety per layer parallelisation
    def memory_safety(self,memory_overhead:float, worker_count:int):
        
        estimated_memory_per_worker = self.dataframe_file_size * memory_overhead
        avaliable_memory = psutil.virtual_memory().available / 1.e6

        max_workers = math.floor(avaliable_memory / estimated_memory_per_worker)
        actual_workers = min(worker_count,max_workers)
    
        if actual_workers < 2:
            actual_workers = 1
    
        return actual_workers
    
    
    #Parallel layer distribution
    def parallel_layer_distribution(self, model:model_class, core_overhead_factor:float = 3.0):
        
        avaliable_cpu_count = self.cpu_cores - 1
        
        if avaliable_cpu_count== 0:
            self.parallel_layers = [0,0,0]
        
        #Parameter tuning layer
        parameter_permutations = sum_parameter_configurations(model)
        parameter_workers = min(avaliable_cpu_count,parameter_permutations)
        parameter_workers = memory_safety(core_overhead_factor,parameter_workers)
        cores_per_parameter = math.floor(avaliable_cpu_count / parameter_workers)
        
        #Fold layer
        fold_workers = min(self.cv_folds,cores_per_parameter)
        fold_workers = memory_safety(core_overhead_factor,fold_workers)
        cores_per_fold = math.floor(cores_per_parameter / fold_workers)
    
    
        #Model layer
        if cores_per_fold >= 2:
            model_workers = cores_per_fold
        
        else:
            model_workers = 1
            
        self.parallel_layers = [parameter_workers, fold_workers, model_workers]
        
    #Runs the test,training based on the specifications of the testing environment
    def train_test(self,model:model_class, test_size:float = 0.3, random_state = 1234):
        
        X_train, X_test, y_train, y_test = train_test_split(self.get_X(),self.get_y(),test_size = test_size, random_state = random_state)
        
        
        
        hyperparameter_tuner = self.hp_type(estimator = model.get_model(),param_grid = model.get_params())
        return hyperparameter_tuner.fit(X_train,y_train)
        
          

