# Electrostatic Classifier
A classifier inspired by electrostatics. A demo can be found [here](https://prajwalsouza.github.io/Experiments/Binary-Classification.html).  

[![A binary classification demo](https://github.com/prajwalsouza/Electrostatic-Classifier/blob/master/Images/demo.png "A binary classification demo")](https://prajwalsouza.github.io/Experiments/Binary-Classification.html)
## Usage
```python
from classifier import ElectroStaticClassifier
clf = ElectroStaticClassifier()

trainingdata = [[1,1], [1,2], [2,1], [3,1], [2,2], [0,3]]
traininglabels = [1, 1, 1, 0, 0, 0]
weights = [0.1, 0.5, 0.4, 0.2, 0.3, 0.1]

clf.fit(trainingdata, traininglabels, weights) #If weights are not given, every data point is assigned the same weight value

clf.predict([0.5, 2]) 
>>> [1]
```
## Dependencies 
1. [Numpy](http://www.numpy.org/) : A fundamental package for scientific computing with Python.
