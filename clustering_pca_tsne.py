# clustering, result visualization using PCA and t-SNE
# https://archive.ics.uci.edu/ml/datasets/wine+quality
# Wine Quality Data Set
# PCA analysis is famous for dimension reduction and data visualization
# here we will use it to visualize the clustering outputs from "clustering-methods-comparison"
# it can give us insights to evaluate if our clusters look reasonable
# the begining of the code is the same from repository "clustering-methods-comparison"
# the visualization is treated in the last loop of this module with the functions from visutils.py

import pandas as pd
from utils import extract, encode
from pipelines import buildpipe
from paramsearch import grid_matrix, customsearch
from visutils import visu_pca_2d, visu_pca_3d, visu_tsne_2d

# import data
df = pd.read_csv('Wine_Quality_Data.csv')
# define target feature
target = 'color'
# remove irrelevant features
df.drop(columns=["quality"], inplace=True)
# extract data columns per type
numeric, categorical, ordinal, binary, label = extract(df.copy(), ordinal_features=None, target=target)
# define x and y
X = df.drop(columns=target)
# list initialization
best_param_res = []
models = []
names = []
# loop over all evaluated models
for name, model, param_grid in grid_matrix():
    # build main pipe
    mainpipe = buildpipe(name, model, numeric, categorical, ordinal, binary)
    # custom quick search of hyperparameters
    best_param = customsearch(X, name, model, param_grid, mainpipe, n_iter=100)
    # store variables
    best_param_res.append(best_param)
    names.append(name)
    models.append(model)
# in real-world industry, true labels are likely to be unknown
# let's try to visalize the clusters
# first we scale the data
df_visu, _, _, _, _ = encode(df=df.copy(), ordinal_features=None, target=target)
# then loop over each model and plot data
for name, model, best_param in list(zip(names, models, best_param_res)):
    mainpipe = buildpipe(name, model, numeric, categorical, ordinal, binary)
    model.set_params(**best_param)
    print(name + " fitting for best parameters : " + str(best_param))
    model = mainpipe.fit(X)
    try:
        labels = model.predict(X)
    except:
        labels = model[name].labels_
    # add clustering results to df_visu
    df_visu["labels"] = labels
    # plot data
    visu_pca_2d(df_visu, "labels", name)
    visu_pca_3d(df_visu, "labels", name)
    visu_tsne_2d(df_visu, "labels", name)

