from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

digits = datasets.load_digits()
print(digits.target)