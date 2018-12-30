# Deep-Learning Learning
Lessons, projects and notes taken from my reading of the Jason Brownlee's book: Deep Learning with python

## Installations (ubuntu)
1. **pip3**: sudo apt install python3-pip
2. **scikit-learn, numpy and scipy**: pip3 install -U scikit-learn
3. **theano**: pip3 install --user theano
4. **tensorflow**: pip3 install --user tensorflow
5. **BLAS**: sudo apt-get install liblapack-dev
6. **Masking tensorflow warning**: export TF_CPP_MIN_LOG_LEVEL=2

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
1. Weights: "The weights on the inputs are very much like the **coefficients** used in a **regression equation**. Like **linear regression**, each neuron also has a bias which can be thought of as an input that always has the value 1.0 and it too must be weighted. Like linear regression, larger weights indicate **increased complexity** and **fragility** of the model. It is desirable to **keep weights in the network small** and regularization techniques can be used."
