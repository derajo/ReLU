import requests


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'

ds = requests.get(url).text
print(len(ds))
# Header is Variance, Skewness, Kurtosis. Entropy of the Wavelet Transformed image
header = 'WTi_var,WTi_skew,WTi_kurt,img_entropy,class\n'
with open('../../data/binary_class_bank_note.csv', 'w+') as f:

	f.write(header+ds)