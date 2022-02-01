# Extraction, Training and Offline test of Neural Network
Codes used to extract, train and test (offline test) a Feedforward Neural Network with EMG arm data

The file named "totalFeatExtracted1.csv" contains the database of the features extracted (using "ExtractorCaract.py") of a set of 
right arm EMG signals acquired through a MYO Armband. Every row represents a set of features exctracted
from a single EMG signal, while the columns indicates the corresponding feature, being the last column 
the ID of the hand movement performed by the user.

The ID hand movements or grasps corresponding to the IDs are:
1 - Hand in rest
2 - Cylinder
3 - Fist 
4 - fine grip
5 - fixed hook
6 - horns
7 - palm out
8 - palm inward
9 - pistol
10 - Spherical


"NNTrain.py" is used to train the Neural Network using the database "totalFeatExtracted1.csv"

"NNTrain-Test.py" is used to test the accuracy of the generated Neural Network model.
