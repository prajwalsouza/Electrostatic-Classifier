import numpy as np
import math
from collections import Counter

class ElectroStaticClassifier:
	def __init__(self, inverse_Distance_Order=2):
		self.inverse_Distance_Order = inverse_Distance_Order

	def fit(self, X, y, weights=None):
		if hasattr(weights, "__len__"):
			self.weights = weights
		else:
			self.weights = np.ones(len(y))
		

		self.numOfClasses = len(Counter(y))
		classes = list(Counter(y).keys())

		if len(X) != len(y):
			raise ValueError('Number of data points must match number of labels.')
		if len(X) != len(self.weights):
			raise ValueError('Number of data points must match number of weights.')

		classPoints = {}
		for k in range(self.numOfClasses):
			classPoints[classes[k]] = []

		for k in range(len(X)):
			for m in range(self.numOfClasses):
				if y[k] == classes[m]:
					classPoints[classes[m]].append([X[k],self.weights[k]])

		# print(classPoints)

		self.data = X
		self.classes = classes
		self.classPoints = classPoints

	def forceFactor(self, pointofInterest, weight, pointdata):
		pos = pointdata
		pOI = pointofInterest

		dist = pOI - pos
		distance = np.linalg.norm(dist)
		force = (weight*1)/(distance**self.inverse_Distance_Order)

		returnAr = np.zeros(len(pos))

		for h in range(len(pos)):
			returnAr[h] = abs(force*dist[h]/distance)

		return returnAr

	def predict(self, newpoint):
		try:
			dim = len(newpoint[0])
			newpoints = newpoint
		except:
			newpoints = [newpoint]

		if len(self.data[0]) != len(newpoints[0]):
			raise ValueError('Dimension of the new datapoint must be same as that of the datapoints in the training set.')

		newpoints = np.asarray(newpoints)

		returnarray = []
		for p in newpoints:
			forceValues = np.zeros(self.numOfClasses)
			for m in range(self.numOfClasses):
				sumarray = np.zeros(len(self.data[0]))
				for point in self.classPoints[self.classes[m]]:
					value = self.forceFactor(p, point[1], point[0])
					sumarray = sumarray + value

				fvalue = np.linalg.norm(sumarray)
				forceValues[m] = fvalue

			mvalue = max(forceValues)
			maxindex = [i for i, j in enumerate(forceValues) if j == mvalue]
			classtype = self.classes[maxindex[0]]
			returnarray.append(classtype)

		return np.asarray(returnarray)

	


