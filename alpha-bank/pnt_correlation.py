import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def plot_variable(x, y, title, x_label, y_label):
    _, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.scatter(x=x, y=y)
    plt.show()


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def correlation(x, y, xlabel, ylabel):
    print('\n', xlabel, '-', ylabel)
    print('Covariance:\n', np.cov(x, y))
    print('Pearson Correlation\n', stats.pearsonr(x, y))
    print('Spearman Correlation\n', stats.spearmanr(x, y))
    print('Fisher-Z Transformation\n', np.arctan(stats.pearsonr(x, y)))
    print('Kendall Correlation\n', stats.kendalltau(x, y))
    print('Weighted Kendall\n', stats.weightedtau(x, y))
    print('Cosine Similarity\n', cosine_similarity(x, y))


train_data = pd.read_csv('Alpha_Train.csv')
test_data = pd.read_csv('Alpha_Test.csv')
pnt_data = pd.read_csv('Greek_Pointers.csv')

tr = len(train_data)
ts = len(test_data)
test_data['Pnt_Open'] = pnt_data.iloc[tr:ts+tr]['Pnt_Open'].values
test_data['Pnt_Close'] = pnt_data.iloc[tr:ts+tr]['Pnt_Close'].values
test_data.to_csv('new_Alpha_Test.csv', index=False)

print(test_data)
exit(0)




print(train_data.shape, test_data.shape, pnt_data.shape)

for i in range(345):
    if train_data.iloc[i]['Date'] != pnt_data.iloc[i]['Date']:
        print(train_data.iloc[i]['Date'])


train_test_data = train_data['Close'].tolist() + test_data['Close'].tolist()
print(train_test_data)

roi_df = pd.DataFrame({
    'alpha': train_test_data,
    'point': pnt_data['Pnt_Close']
})

correlation(roi_df['alpha'], roi_df['point'], 'Alpha', 'Pointers')
plot_variable(roi_df['alpha'], roi_df['point'], 'Alpha Bank - Pointers', 'Alpha', 'Pointers')