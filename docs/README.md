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

## Lessons learned
### Libraries
1. **Theano** and **Tensorflow** are two numerical libraries largely used to develop deep learning models.
2. Is possible to make models directly using Theano and Tensorflow, but the project can get too complex.
3. The **Keras** library (a python library used to make deep learning models) have as its purpose modulating and masking the complexity of Theano or Tensorflow, depending on which of them are set to be its **backend**.
4. The construction of deep learning models in Keras can be summarized as:
- **Defining the model**: The Keras library uses the idea of **model** and the main type of a model is the **Sequencial**. The Sequencial consists of a stack of layers, where each layer are added following the sequence that the developer wants the computing to be executed.
- **Compiling the model**: As the name says, this is the model compilation, where it uses the underlying framework (Theano or Keras) to optimize the computing that the model executes. In the compilation, the **loss function** and **optimizers** are defined. In this step, the **compile()** function is called on the model.
- **Fiting the model**: After compiled, the model can be fit on training data, using the **fit()** function. This step can be done one batch of the data at a time or passing all the model training data at once. It is in this step that the computation happens in fact.
- **Making predictions**: Where the defined, compiled and fitted model serves its purpose, being used to make predictions in new data based in its previous training. This is made by calling on model functions like **evaluate()** and **predict()**.

### Multilayer perceptrons
1. **Weights**: "The weights on the inputs are very much like the **coefficients** used in a **regression equation**. Like **linear regression**, each neuron also has a bias which can be thought of as an input that always has the value 1.0 and it too must be weighted. Like linear regression, larger weights indicate **increased complexity** and **fragility** of the model. It is desirable to **keep weights in the network small** and regularization techniques can be used."
2. **Multiclass classification**: "In this case a softmax activation function may be used to output a probability
of the network predicting each of the class values. Selecting the output with the highest probability can be used to produce a crisp class classification value".
- "The softmax function takes an un-normalized vector, and normalizes it into a probability distribution. That is, prior to applying softmax, some vector elements could be negative, or greater than one; and might not sum to 1; but after applying softmax, each element x is in the interval [0,1], and sum to 1. Softmax is often used in neural networks, to map the non-normalized output to a probability distribution over predicted output classes".
3. **Data Preparation**:
- **Datatype**: It's important to know, before anything, that data must be numerical. If there is some categorical data, it has to be mapped (converted) to a numerical value. One Hot Encoding is a common transformation of this type, that transforms categorical value to a real-valued representation.
- **Data-scaling**: Data-scaling is important, but it can cause overfitting in the model.

### Multiclass Classification
1. **kFold vs Stractified-kFold**: If you want that folds do not adapt to the percentage of the classes, just use kFold (in this case, you have to make the One Hot Encoding, passing the class in a dummy format to the model evaluation with cross_val_score). If you want that the folds consider the percentege of the classes in the fold splits, you have to use Stractfied-kFold, which takes a 1D array (not a hot encond format) with the classes. So, using kFold, **y** have to be on a one hot encoding format (see **/code/local_env/part_three/010_mcc_irisflowers-book.py**), whilst Stractfied-kFold takes a list with the classes integer values (see **/code/local_env/part_three/011_mcc_irisflowers-maximo_1.py**).

### Regression
1. The result (mean and standard deviation) of the cross_val_score applied in a KerasRegressor is a negative number, 'cause this is the mean (and std) of the loss values, so, this is the value that we want to minimize (as this is negative, it is maximized instead). The change (in the book the result is positive) was made to use other libraries that minimize the loss (maximizing the result).

### Predictions
1. Predictions takes as argument the input X (to be predicted) as a numpy array or a numpy array of lists (when the model takes more then one input value (in a model that the data have 8 features, the second option would be used (a numpy array of lists))).
2. Predictions can be made without re-compiling an loaded model.
3. There are basically two ways of predicting models:
- **model.predict(X)**: which returns one or more numpy arrays of predictions.

   - If it is a multi-class classifier, for example, it will return, for a single entry X to be predict, a numpy array of probabilities of each class being the right one.

   - If it is a binary classifier, it will return a float value, which can be read as: the chosen class is the most next to this value. So, if the return is 0.9, the most probable class is 1.

   - If it is a regression model, the output will be the predicted value for the X entry.

- **model.predict_classes(X)**: which returns the index of the predicted class in the array of classes.

4. Examples:
- **model.predict(X)** example (pima indians diabetes dataset):
```python
  prediction = loaded_model.predict(numpy.array([[1,85,66,29,0,26.6,0.351,31]]))
  print(prediction)
```
will produce a output like:
```console
  [[0.09788511]]
```
- **model.predict_classes(X)** example (pima indians diabetes dataset):
```python
  prediction = loaded_model.predict_classes(numpy.array([[1,85,66,29,0,26.6,0.351,31]]))
  print(prediction)
```
will produce a output like:
```console
  [[0]]
```

### Saving and loading models:
1. As said before, it is not necessary to re-compile the model to make predictions, this is possible because predictions does not messes up with evaluations or updates in the weights. Re-compiling is just necessary when:
- It's wanted to change: Loss function; Optimizer / Learning rate; Metrics.
- The loaded model was not compiled yet (or this information is unknown).
- When it's necessary to evaluate the loaded model.
- When it's wanted to train the loaded model, with the same or other parameters.
2. [Important Link](https://stackoverflow.com/questions/47995324/does-model-compile-initialize-all-the-weights-and-biases-in-keras-tensorflow) - Stackoverflow topic where I learned most of it.

### Dropout regularization - copied from the book
1. Generally use a small dropout value of 20%-50% of neurons with 20% providing a good
starting point. A probability too low has minimal effect and a value too high results in
under-learning by the network.
2. Use a larger network. You are likely to get better performance when dropout is used
on a larger network, giving the model more of an opportunity to learn independent
representations.
3. Use dropout on input (visible) as well as hidden layers. Application of dropout at each
layer of the network has shown good results.
4. Use a large learning rate with decay and a large momentum. Increase your learning rate
by a factor of 10 to 100 and use a high momentum value of 0.9 or 0.99.
5. Constrain the size of network weights. A large learning rate can result in very large
network weights. Imposing a constraint on the size of network weights such as max-norm
regularization with a size of 4 or 5 has been shown to improve results.

### CNN Best Practices
1. Input Receptive Field Dimensions: The default is 2D for images, but could be 1D
such as for words in a sentence or 3D for video that adds a time dimension.
2. Receptive Field Size: The patch should be as small as possible, but large enough to
see features in the input data. It is common to use 3 × 3 on small images and 5 × 5 or
7 × 7 and more on larger image sizes.
3. Stride Width: Use the default stride of 1. It is easy to understand and you don’t need
padding to handle the receptive field falling off the edge of your images. This could be
increased to 2 or larger for larger images.
4. Number of Filters: Filters are the feature detectors. Generally fewer filters are used at
the input layer and increasingly more filters used at deeper layers.
5. Padding: Set to zero and called zero padding when reading non-input data. This is
useful when you cannot or do not want to standardize input image sizes or when you want
to use receptive field and stride sizes that do not neatly divide up the input image size.
6. Pooling: Pooling is a destructive or generalization process to reduce overfitting. Receptive
field size is almost always set to 2 × 2 with a stride of 2 to discard 75% of the activations
from the output of the previous layer.
7. Data Preparation: Consider standardizing input data, both the dimensions of the
images and pixel values.
8. Pattern Architecture: It is common to pattern the layers in your network architecture.
This might be one, two or some number of convolutional layers followed by a pooling layer.
This structure can then be repeated one or more times. Finally, fully connected layers are
often only used at the output end and may be stacked one, two or more deep.
9. Dropout: CNNs have a habit of overfitting, even with pooling layers. Dropout should be
used such as between fully connected layers and perhaps after pooling layers.

### TODO
1. Study Dropout Regularization - search for "norm of the weights"
2. Study [SGD](https://keras.io/optimizers/).
3. Study sgd's momentum.
4. Write the Dropout section with my words.
5. Write the CNN Best Practices section with my words.
6. Study word embeddings.
7. Study Recurrent Neural Networks.
8. Study LSTM.

### Important links:
1. [iloc](https://stackoverflow.com/questions/19155718/select-pandas-rows-based-on-list-index);
2. [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html);
3. **CNN - Error on input Dimensions**: Runing the first cnn example direct from the book, I got ```Negative dimension size caused by subtracting 5 from 1 for 'conv2d_1/convolution' (op: 'Conv2D') with input shapes: [?,1,28,28], [5,5,28,32].```. This error was caused by the divergence between the code from the book and the input shape specification that keras 2 uses. Instead of .reshape(?, 1, 28, 28) I used .reshape(?, 28, 28, 1) and, on the input_dim I used input_dim=(28, 28, 1) and it worked. A detailed explanation in this [important link](https://stackoverflow.com/questions/41651628/negative-dimension-size-caused-by-subtracting-3-from-1-for-conv2d).
