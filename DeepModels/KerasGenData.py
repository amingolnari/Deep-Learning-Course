"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
Keras Version : 2.2.4
Date : 10/12/2018

Generate batches of tensor image data from directory
Code 303
"""

from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

def GenData(Path, RGBmode = True):
	gen = image.ImageDataGenerator()
	if RGBmode:
		color_mode = 'rgb'
	else:
		color_mode = 'grayscale'

	TrainData = gen.flow_from_directory(Path + 'Train', shuffle = True,
	                                    color_mode = color_mode,
	                                    batch_size = 490, # All Images in Folder
	                                    target_size = (100, 100), # Resize Image to (64, 64, Channel)
	                                    class_mode = 'binary') # Cat vs Dog (2 Classes)
	TestData = gen.flow_from_directory(Path + 'Test', shuffle = False,
	                                    color_mode = color_mode,
	                                    batch_size = 210,
	                                    target_size = (100, 100),
	                                    class_mode = 'binary')
	# Make Data and Labels
	(XTrain, YTrain) = next(TrainData)
	(XTest, YTest) = next(TestData)
	
	return (XTrain, YTrain), (XTest, YTest)

def ShowRandomSamples(Data):
	idx = np.random.randint(0, Data.shape[0], 36)
	_, ax = plt.subplots(6, 6, figsize = (12, 12), facecolor = 'k')
	i = 0
	for r in range(6):
		for c in range(6):
			ax[r][c].imshow(Data[idx[i],:,:,:])
			ax[r][c].axis('off')
			i += 1
	plt.show()
	
	return

def main():
	Path = 'Data/'
	(XTrain, YTrain), (XTest, YTest) = GenData(Path)
	XTrain /= 255.0
	XTest /= 255.0
	ShowRandomSamples(XTrain)
	
	return

if __name__ == '__main__':
	main()
