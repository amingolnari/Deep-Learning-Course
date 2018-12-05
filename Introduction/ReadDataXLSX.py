"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
Pandas Version : 0.23.4
Date : 5/12/2018

Load Data From Excel File with Pandas
Code 0
"""

import pandas
import numpy as np

def Readxlsx(Fname, ShName):
	# Read Excel File
	Excel = pandas.read_excel(Fname, sheet_name = ShName)
	return Excel

def LoadData(Excel, X, Y, Shuffle = True):
	# Shuffle Data
	if Shuffle:
		idx = np.random.permutation(len(Excel))
	else:
		idx = np.arange(len(Excel))
	X[0:, 0] = Excel.X1[idx].real
	X[0:, 1] = Excel.X2[idx].real
	X[0:, 2] = Excel.X3[idx].real
	Y[0:, 0] = Excel.Y1[idx].real
	Y[0:, 1] = Excel.Y2[idx].real
	# Normalize 0-1
	X[:, 0] = (X[:, 0] - np.min(X[:, 0])) / (np.max(X[:, 0]) - np.min(X[:, 0]))
	X[:, 1] = (X[:, 1] - np.min(X[:, 1])) / (np.max(X[:, 1]) - np.min(X[:, 1]))
	X[:, 2] = (X[:, 2] - np.min(X[:, 2])) / (np.max(X[:, 2]) - np.min(X[:, 2]))
	X = np.expand_dims(X, axis = 2)
	return X, Y


if __name__ == "__main__":
	Excel = Readxlsx('Files/SAMPLE.xlsx', 'Data')
	X = np.zeros((len(Excel), 3), dtype = 'float32')
	Y = np.zeros((len(Excel), 2), dtype = 'float32')
	X, Y = LoadData(Excel, X, Y, Shuffle = True)
