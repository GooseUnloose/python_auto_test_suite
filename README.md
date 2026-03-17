# python_auto_test_suite
A formalised version of an automated test, train, result script I used during my thesis. Build using only the Pandas Library and Scikit-learn as dependencies
>!!!THIS PROJECT IS IN CONTINOUS DEVELOPMENT!!! I aim to improve this through periodic updates, but it is currently in a working state

## The Why
When training ML models, particularly on desktop or lower spec machines, the biggest gatekeeper of productivity is: Time. 
Time spent for the model to train on your dataset; Time spent for evaluating model performance; Time setting up the testing environment itself. This project seeks to reduce development time through an automated testing environment that utilises a cascading core distribution system for smart, effective parallelism for model training.

The repo consists of two classes, the tesing environment and a generic model class.

## Testing Environment

The testing environment is a configurable class designed to allow users to perform a complete test suite on a dataset with minimal method calls.


### Parallelisation Cascading
Parallelisation Cascading is the process of leveraging a device's avaliable cores in a, locally optimal, distribution across all 'layers' of the training stage. Focusing on providing more cores in sections that provide the most impact. Memory safety is also considered during the core distribution, ensuring that there is sufficient memory to make the parallelisation possible at each layer.</br>

#### We can break down a full training scenario into three 'layers':
##### Layer 1: Hyper parameter tuning
This layer benefits the most from parallelisation as it contains the most test permutations, based on the number of parameters to test and the tuning method selected. This layer is at the top of our pile, as such it will get all avaliable cores before passing down the remainders to the following layer.
##### Layer 2: Cross Validation
Cross validation is similarly taxing, dependant on the number of folds selected, this gets the remaineder of the free cores. Parallelisation here is based on the number of free cores for each parameter permutation. In a typical desktop this is usually serialised as all of the cores have been reserved for the hp tuning layer.
##### Layer 3: Model layer
All of the remaining cores avaliable per fold are reserved for the model layer, however by this section almost all of them have been reserved already, thus it is typically serialised as well. This layer benefits the least from parallelism than the other two layers - based on there being a reasonable amount of folds and tuning parameters selected. If both of the above layers <b>are</b> serialised, this layer takes all of the free cores instead.


## Model Class

The model class is a generic class used in for the testing_environment. It aims to provide a library agnostic interface for machine learning models to interact with the testing environment through familiar behaviours.

## The How
This section details how to setup and run a model training environment using the repository:

A dataframe of the chosen dataset is defined as normal, and a testing environment object is also created, with the dataframe passed as a required parameter
```
df = pd.read_CSV(...)
test = Testing_Environment(df)
```
The X and Y subsets can then be defined in the testing_environment
```
test.set_X('column_name')#The name y column to be removed from the X subset
test.set_y('column_name')#The name of the y column
```
Avaliable cores for parallelism can be explcitly stated using a getter, or an automatic detection method can be called to identify all avalible cores on the system
```
test.auto_set_cpu_cores() #automatic function
test.set_cpu_cores(number) #explicit setter
```
To create a model_class object the model class and a dictionary of testing parameters are passed as parameters
```
model = Model_Class(LogisticRegression(),{parameters:...})
```
Calculating the core distribution across training layers can be done through the parallel_layer_distribution() method
```
test.parallel_layer_distribution(LogisticRegression()) #The model is passed into this as we use the number of
                                                        parameter permutations to determine number of cores for
                                                        tuning layer
```
To train the model using the environment, the train function is called, with the model passed as a required parameter
```
output = test.train_test(model,log_results=True) #log_results will output a txt file of the model's performance across all folds
```

Following these steps will produce a trained model using the best performing parameters across the whole training phase. As well as a log of the model's performance in an external .txt file
