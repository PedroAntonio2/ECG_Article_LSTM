"""
Myocardial Infarction Localization and Blocked
Coronary Artery Identification Using a Deep
Learning Method
"""

from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.utils import class_weight