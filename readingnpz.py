from numpy import load

#checking this npz file out of curiosity, seems to contain mean and std probably used in normalization
data = load('data/y_stats.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])