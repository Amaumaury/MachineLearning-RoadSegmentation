# Project Road Segmentation
Road Segmentation Project done in the Machine Learning class at EPFL in the fall of 2017.

## Dependencies & Installation
- [python](https://www.python.org) version 3.6
- [numpy](http://www.numpy.org) for array and tensor manipulation.
- [scipy](https://www.scipy.org) and [scikit-image](http://scikit-image.org) for image processing.
- [matplotlib](https://matplotlib.org) for plots.
- [keras](https://keras.io) version 2.0.8 for machine learning with the TensorFlow backend.

All dependencies can be installed with [anaconda](https://www.anaconda.com/download/#macos) for python 3.6.
Numpy, Scipy, Scikit-image and matplotlib are already installed in the basic conda package.
To install keras use the command line 'conda install keras'.

## Running our model
We saved our trained model in the file gold_train_model_hsv.hdf5
Simply run the file run.py which will load the model and predict on the tests images and produce the same submission.csv as the one given on Kaggle. This script will take approximately 10-15mn to preprocess the images and predict from the model.
If you wish to train a new model, simply run the train_deepcnn.py which will save the model under 'train_model.hdf5'. This script will take long to run (1h30 - 2h) and will require a GPU. To test it modify the path of the model in run.py.

## Files description
- pred_images is the directory were predicted images will be saved.
- provided is the directory with the provided code for the project.
- test_set_images is the directory containing the test images.
- training is the directory containing the train images, grountruth and hsv images preprocessed.
- gold_train_model_hsv.hdf5 is our final model.
- preprocessing.py contains our preprocessing functions.
- run.py is the script to predict on the test images.
- train_deepcnn.py is the script to train a new model.


## The Team - Les Semi-Croustillants
- Vincenzo Bazzucchi (vincenzo.bazzucchi@epfl.ch)
- Amaury Combes (amaury.combes@epfl.ch)
- Alexis Montavon (alexis.montavon@epfl.ch)
