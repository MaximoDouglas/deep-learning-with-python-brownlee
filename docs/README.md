# Deep-Learning Learning
Lessons, projects and notes taken from my reading of the Jason Brownlee's book: Deep Learning with python

## Installations (ubuntu)
1. **pip3**: sudo apt install python3-pip
2. **scikit-learn, numpy and scipy**: pip3 install -U scikit-learn
3. **theano**: pip3 install --user theano
4. **tensorflow**: pip3 install --user tensorflow
5. **BLAS**: sudo apt-get install liblapack-dev
6. **Masking tensorflow warning**: export TF_CPP_MIN_LOG_LEVEL=2
7. **python3-tk**: sudo apt install python3-tk
8. **matplotlib**: pip3 install --user matplotlib
9. **pandas**: pip3 install --user pandas
10. **keras**: pip3 install --user keras

## Lessons learned (30-12-2018)
### Libraries
1. **Theano** and **Tensorflow** are two numerical libraries largely used to develop deep learning models.
2. Is possible to make models directly using Theano and Tensorflow, but the project can get too complex.
3. The **Keras** library (a python library used to make deep learning models) have as its purpose modulating and masking the complexity of Theano or Tensorflow, depending on which of them are set to be its **backend**.
4. The construction of deep learning models in Keras can be summarized as:
- **Defining the model**: The keras library uses the idea of **model** and the main type of a model is the **Sequencial**. The Sequencial consists of a stack of layers, where each layer are added following the sequence that the developer wants the computing to be executed.
- **Compiling the model**: As the name says, this is the model compilation, where it uses the underlying framework (Theano or Keras) to optimize the computing that the model executes. In the compilation, the **loss function** and **optimizers** are defined. In this step, the **compile()** function is called on the model.
- **Fiting the model**: After compiled, the model can be fit on training data, using the **fit()** function. This step can be done one batch of the data at a time or passing all the model training data at once. It is in this step that the computation happens in fact.
- **Making predictions**: Where the defined, compiled and fitted model serves its purpose, beeing used to make preddictions in new data based in its previous training. This is made by calling on model functions like **evaluate()** and **predict()**.

### Multilayer perceptrons
1. **Weights**: "The weights on the inputs are very much like the **coefficients** used in a **regression equation**. Like **linear regression**, each neuron also has a bias which can be thought of as an input that always has the value 1.0 and it too must be weighted. Like linear regression, larger weights indicate **increased complexity** and **fragility** of the model. It is desirable to **keep weights in the network small** and regularization techniques can be used."
2. **Multiclass classification**: "In this case a softmax activation function may be used to output a probability
of the network predicting each of the class values. Selecting the output with the highest probability can be used to produce a crisp class classification value".
- "The softmax function takes an un-normalized vector, and normalizes it into a probability distribution. That is, prior to applying softmax, some vector elements could be negative, or greater than one; and might not sum to 1; but after applying softmax, each element x is in the interval [0,1], and sum to 1. Softmax is often used in neural networks, to map the non-normalized output to a probability distribution over predicted output classes".
3. **Data Preparation**:
- **Datatype**: It's important to know, before anything, that data must be numerical. If there is some categorical data, it has to be mapped (converted) to a numerical value. One Hot Encoding is a common transformation of this type, that transforms categorical value to a real-valued representation.
- **Data-scaling**: Data-scaling is important, but it can cause overfitting in the model.

### Multiclass Classification
1. **kFold vs Stractified-kFold**: If you want that folds do not adapt to the percentage of the classes, just use kFold (in this case, you have to make the One Hot Enconding, passing the class in a dummy format to the model evaluation with cross_val_score). If you want that the folds consider the percentege of the classes in the fold splits, you have to use Stractfied-kFold, which takes a 1D array (not a hot encond format) with the classes. So, using kFold, **y** have to be on a one hot enconding format (see **/code/local_env/part_three/010_mcc_irisflowers-book.py**), whilst Stractfied-kFold takes a list with the classes integer values (see **/code/local_env/part_three/011_mcc_irisflowers-maximo_1.py**).

### Regression
1. The result (mean and standard deviation) of the cross_val_score applied in a KerasRegressor is a negative number, 'cause this is the mean (and std) of the loss values, so, this is the value that we want to minimize (as this is negative, it is maximized instead). The change (in the book the result is positive) was made to use other libraries that minimize the loss (maximizing the result). 

### Important links:
1. [iloc](https://stackoverflow.com/questions/19155718/select-pandas-rows-based-on-list-index);
2. [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html);
