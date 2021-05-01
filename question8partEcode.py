# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 22:16:30 2021

@author: thedr
"""

test_dataset = datasets.MNIST('./data', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
device = torch.device("cpu")
model = Net().to(device)
X = []
y= []
for data in test_dataset:
    #print(data[0])
    array = model.to_feature(torch.reshape(data[0],(1,1,28,28))).detach().numpy()
    X.append(array[0])
    y.append(int(data[1]))
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
print('plotting')
# We want to get TSNE embedding with 2 dimensions
X = np.array(X)#.reshape((10000,64))
n_components = 2
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(X)
tsne_result.shape
# (1000, 2)
# Two dimensions for each of our images
 
# Plot the result of our TSNE with the label color coded
# A lot of the stuff here is about making the plot look pretty and not TSNE
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=1)
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)



final = np.zeros((4*28,8*28))
fig, ax = plt.subplots(4,8)
for i in range(4):
    I0 = X[np.random.randint(0,999)]
    used = []
    for j in range(8):
        index  = 0 
        smallest_score = 999
        for image in X:
            if ((np.linalg.norm(image - I0) < smallest_score) and (index not in used)):
                smallest = image
                smallest_score = np.linalg.norm(image - I0)
                small_index = index
            index += 1
        final[i*28:(i+1)*28,j*28:(j+1)*28] = test_dataset[small_index][0]
        used.append(small_index)

for image_set in ax:
    

# sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=5)
# lim = (tsne_result.min()-5, tsne_result.max()+5)
# ax.set_xlim(lim)
# ax.set_ylim(lim)
# ax.set_aspect('equal')
# ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)