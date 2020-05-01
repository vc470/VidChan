# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:07:40 2019

@author: VChan
"""

###############################################################################
###############################################################################
### CLUSTERING
# Let's try clustering using "subsetdrugdensityonlycol": 

from sklearn.cluster import KMeans
from sklearn import datasets

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import pairwise_distances_argmin
## FOr using any distance measures....
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn import metrics


subsetdrugdensityonlycol_clustering = subsetdrugdensityonlycol[['BUYER_COUNTY','BUYER_STATE','year_x','densitybins',
                                           'WA_PER','BA_PER','IA_PER','H_PER','Poverty_Percent','crimerate',
                                           'Median_Income','ltHS','HS','sCorA','B+','ObesityPercent','democrat','republican','Unemp_Yearly_Rate']]

# https://www.youtube.com/watch?v=gSSd2M12OfQ  # YouTube tutorial about clustering

# https://nikkimarinsek.com/blog/7-ways-to-label-a-cluster-plot-python  # Color coding blog

# CLUSTERING FOR THE ENTIRE PERIOD:  first model

featuresmain = subsetdrugdensityonlycol[['WA_PER','BA_PER','IA_PER','H_PER','Poverty_Percent','crimerate',
                                           'Median_Income','HS','sCorA','B+','ObesityPercent','democrat','Unemp_Yearly_Rate']]
targetsmain = subsetdrugdensityonlycol[['densitybins']]

modelmain = KMeans(n_clusters=4) # 4 density bins. So we want four clusters.

KMmodelmain = modelmain.fit(featuresmain)
#KMmodel

KMmodelmain.labels_

print("Centers: ", KMmodelmain.cluster_centers_)  
print("Labels: ", KMmodelmain.labels_)
print("Intertia (L2norm dist):", KMmodelmain.inertia_)

plt.scatter(featuresmain.ObesityPercent,featuresmain.Poverty_Percent)

colormap = np.array(['Red','Blue','Yellow','Green'])

subsetdrugdensityonlycol['label'] = KMmodelmain.labels_
subsetdrugdensityonlycol['color'] = colormap[KMmodelmain.labels_]


plt.scatter(featuresmain.ObesityPercent,featuresmain.Poverty_Percent, c = colormap[KMmodelmain.labels_],s=5) # not so good visual
plt.xlabel("Obesity Rate")
plt.ylabel("Poverty Rate")

plt.scatter(featuresmain.Median_Income,featuresmain.H_PER, c = colormap[KMmodelmain.labels_],s=5) # pretty good visual
plt.xlabel("Median Income")
plt.ylabel("Hispanic Percentage")

subsetdrugdensityonlycol.to_csv(r'C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/ALMOST FINAL-VID/subsetdrugdensityonlycol.csv',index=False)

### CLUSTERING : SECOND  MODEL
featuresmainsecondmodel = subsetdrugdensityonlycol[['Poverty_Percent','crimerate','Median_Income','HS','ObesityPercent','democrat','Unemp_Yearly_Rate']]
modelmainsecondmodel = KMeans(n_clusters=4) # 4 density bins. So we want four clusters.
KMmodelmainsecondmodel = modelmainsecondmodel.fit(featuresmainsecondmodel)

KMmodelmainsecondmodel.labels_
colormap = np.array(['Red','Blue','Yellow','Green'])

plt.scatter(featuresmainsecondmodel.ObesityPercent,featuresmainsecondmodel.Poverty_Percent, c = colormap[KMmodelmainsecondmodel.labels_],s=5) # not so good visual
plt.xlabel("Obesity Rate")
plt.ylabel("Poverty Rate")

plt.scatter(featuresmainsecondmodel.Median_Income,featuresmainsecondmodel.Poverty_Percent, c = colormap[KMmodelmainsecondmodel.labels_],s=5) # pretty good visual
plt.xlabel("Median Income")
plt.ylabel("Hispanic Percentage")

### CLUSTERING : THIRD  MODEL
featuresmain3model = subsetdrugdensityonlycol[['Median_Income','HS']]
modelmain3model = KMeans(n_clusters=4) # 4 density bins. So we want four clusters.
KMmodelmain3model = modelmain3model.fit(featuresmain3model)

KMmodelmain3model.labels_
colormap = np.array(['Red','Blue','Yellow','Green'])

plt.scatter(featuresmain3model.Median_Income,featuresmain3model.HS, c = colormap[KMmodelmain3model.labels_],s=5) # pretty good visual
plt.xlabel("Median Income")
plt.ylabel("High School Completion Rate")


# For 2006 clustering: 

cluster2006 = subsetdrugdensityonlycol_clustering[subsetdrugdensityonlycol_clustering['year_x']==2006]

cluster2006.to_csv(r'C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/ALMOST FINAL-VID/cluster2006.csv',index=False)

#target = cluster2006.iloc[:,0:4].values
#features = cluster2006.iloc[:, 5:18].values # featured data:

features = cluster2006[['WA_PER','BA_PER','IA_PER','H_PER','Poverty_Percent','crimerate',
                                           'Median_Income','ltHS','HS','sCorA','B+','ObesityPercent','democrat','republican','Unemp_Yearly_Rate']]
targets = cluster2006[['densitybins']]

model = KMeans(n_clusters=4) # 4 density bins. So we want four clusters.

KMmodel = model.fit(features)
#KMmodel

KMmodel.labels_

print("Centers: ", KMmodel.cluster_centers_)  
print("Labels: ", KMmodel.labels_)
print("Intertia (L2norm dist):", KMmodel.inertia_)

plt.scatter(features.ObesityPercent,features.Poverty_Percent)
colormap = np.array(['Red','Blue','Yellow','Green'])

plt.scatter(features.ObesityPercent,features.Poverty_Percent, c = colormap[KMmodel.labels_],s=40) # not so good visual

plt.scatter(features.Median_Income,features.H_PER, c = colormap[KMmodel.labels_],s=40) # pretty good visual

plt.scatter(features.Median_Income,features.Poverty_Percent, c = colormap[KMmodel.labels_],s=40) # pretty good visual


plt.scatter(features.WA_PER,features.HS, c = colormap[KMmodel.labels_],s=40) # not so good visual



# For 2009 clustering: 

cluster2009 = subsetdrugdensityonlycol_clustering[subsetdrugdensityonlycol_clustering['year_x']==2009]

#target = cluster2009.iloc[:,0:4].values
#features = cluster2009.iloc[:, 5:18].values # featured data:

features2 = cluster2009[['WA_PER','BA_PER','IA_PER','H_PER','Poverty_Percent','crimerate',
                                           'Median_Income','ltHS','HS','sCorA','B+','ObesityPercent','democrat','republican','Unemp_Yearly_Rate']]
targets2 = cluster2009[['densitybins']]

model2 = KMeans(n_clusters=4) # 4 density bins. So we want four clusters.

KMmodel2 = model2.fit(features2)


#KMmodel

KMmodel2.labels_

print("Centers: ", KMmodel2.cluster_centers_)  
print("Labels: ", KMmodel2.labels_)
print("Intertia (L2norm dist):", KMmodel2.inertia_)

plt.scatter(features2.ObesityPercent,features2.Poverty_Percent)
colormap2 = np.array(['Red','Blue','Yellow','Green'])

plt.scatter(features2.Median_Income,features2.Poverty_Percent, c = colormap2[KMmodel2.labels_],s=40)

plt.scatter(features2.Median_Income,features2.H_PER, c = colormap2[KMmodel2.labels_],s=40)


####################### EVALUATION FOR CLUSTERING ANALYSIS:
# https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a

from sklearn.metrics import silhouette_score

# Main data: .53496
predsmain = modelmain.fit_predict(featuresmain)
scoremain = silhouette_score (featuresmain, predsmain, metric='euclidean')
scoremain

# Main Data: Second Model- 0.5350
predsmainsecondmodel = modelmainsecondmodel.fit_predict(featuresmainsecondmodel)
scoremainsecondmodel = silhouette_score (featuresmainsecondmodel, predsmainsecondmodel, metric='euclidean')
scoremainsecondmodel

# Main Data: Second Model- 0.5350
predsmain3model = modelmain3model.fit_predict(featuresmain3model)
scoremain3model = silhouette_score (featuresmain3model, predsmain3model, metric='euclidean')
scoremain3model


# 2006 data: .53704
preds = model.fit_predict(features)
score = silhouette_score (features, preds, metric='euclidean')
score

# 2009 data: .55055
preds2 = model2.fit_predict(features2)
score2 = silhouette_score (features2, preds2, metric='euclidean')
score2



