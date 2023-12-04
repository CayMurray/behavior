## MODULES/INITIALIZATION ##

import numpy as np
import pandas as pd
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from matlab import engine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

eng = engine.start_matlab()
FS_path = 'data/FS.mat'
HC_path = 'data/HC.mat'

FS_labels = ['US_PRE','FS','US+1','US+2','US+3',
            'HC_PRE','HC_POST','HC_PRE+1','HC_POST+1',
            'HC_PRE+2','HC_POST+2','HC_PRE+3','HC_POST+3']
HC_labels = ['US_PRE','US+1','US+2','US+3',
            'HC_PRE','HC_POST','HC_PRE+1','HC_POST+1',
            'HC_PRE+2','HC_POST+2','HC_PRE+3','HC_POST+3']


## FUNCTIONS ##

def convert_to_pd(path,context_labels):
    data = eng.load(path)
    df_totmat,anim_index = pd.DataFrame(data['TOTMAT']),data['anim_index']
    total_neurons = [i for sublist in anim_index for i in sublist]

    start = 0
    ctimes = [int(i) for sublist in data['ctimes'] for i in sublist]

    columns = []

    for val, (context,end) in enumerate(zip(context_labels,ctimes)):
        if val == 0:
            adjust = 0
        else:
            adjust = 1

        expanded_labels = [context]*(end-start+adjust)
        columns.append(expanded_labels)
        start = end + 1

    df_master = pd.DataFrame(columns=[label for i in columns for label in i])

    for rat_id, rat in enumerate(set(total_neurons)):
        new_index = [neuron == rat for neuron in total_neurons]
        rat_neurons = np.array(df_totmat[new_index],dtype=np.float64)

        for i in range(rat_neurons.shape[0]):
            indv_neurons = rat_neurons[i,:]
            df_master.loc[f"rat_{rat_id}_{i}"] = indv_neurons

    return df_master


def reduce_dim(df_master,mode):
    z_scaler = StandardScaler()
    scaled_data = z_scaler.fit_transform(df_master.T.values)

    if mode == 'PCA':
        pca = PCA(n_components=2)
        calcium_pcs = pca.fit_transform(scaled_data)
        df_reduce = pd.DataFrame(data=calcium_pcs,columns=['PC1','PC2'])

    elif mode == 'UMAP':
        reducer = umap.UMAP(n_components=2,random_state=42)
        embedding = reducer.fit_transform(scaled_data)
        df_reduce = pd.DataFrame(data=embedding,columns=['UMAP1','UMAP2'])

    df_reduce['context_ids'] = [context for context in df_master.columns]

    return df_reduce


def visualize(HC_pc,HC_umap,FS_pc,FS_umap,desired_contexts):
    num_contexts = len(desired_contexts)
    fig,axs = plt.subplots(nrows=num_contexts, ncols=4, figsize=(20,5*num_contexts))

    if num_contexts == 1:
        axs = axs.reshape(1, -1)

    for i,context in enumerate(desired_contexts):
        df_HC_pcs = HC_pc[HC_pc['context_ids'].isin(context)]
        df_FS_pcs = FS_pc[FS_pc['context_ids'].isin(context)]

        df_HC_umaps = HC_umap[HC_umap['context_ids'].isin(context)]
        df_FS_umaps = FS_umap[FS_umap['context_ids'].isin(context)]

        sns.scatterplot(ax=axs[i,0],data=df_HC_pcs,x='PC1',y='PC2',hue='context_ids',palette='tab10')
        axs[i,0].set_title(f'HC_PCs')
        sns.scatterplot(ax=axs[i,1],data=df_FS_pcs,x='PC1',y='PC2',hue='context_ids',palette='tab10')
        axs[i,1].set_title(f'FS_PCs')
        sns.scatterplot(ax=axs[i,2],data=df_HC_umaps,x='UMAP1',y='UMAP2',hue='context_ids',palette='tab10')
        axs[i,2].set_title(f'HC_UMAPs')
        sns.scatterplot(ax=axs[i,3],data=df_FS_umaps,x='UMAP1',y='UMAP2',hue='context_ids',palette='tab10')
        axs[i,3].set_title(f'FS_UMAPs')

    plt.tight_layout()
    plt.show()


def rf(df_master,contexts,mode):
    df_master = df_master[df_master['context_ids'].isin(contexts_to_predict)]
    X = df_master[[f'{mode}1',f'{mode}2']]
    Y = df_master[['context_ids']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    unique_labels = sorted(set(y_test['context_ids']))

    rf = RandomForestClassifier(n_estimators=100,random_state=42)
    rf.fit(X_train,y_train)
    predictions = rf.predict(X_test)
    print(classification_report(y_test,predictions))

    cm = confusion_matrix(y_test,predictions)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, fmt='g',annot=True, cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


## CONVERT TO PANDAS ##
df_HC = convert_to_pd(path=HC_path,context_labels=HC_labels)
df_FS = convert_to_pd(path=FS_path,context_labels=FS_labels)


## DIMENSIONALITY REDUCTION + VISUALIZATION ##

#df_HC_pcs = reduce_dim(df_HC,'PCA')
#df_FS_pcs = reduce_dim(df_FS,'PCA')
#df_HC_umaps = reduce_dim(df_HC,'UMAP')
df_FS_umaps = reduce_dim(df_FS,'UMAP')

contexts_to_visualize = [('HC_PRE','HC_POST'), ('HC_PRE','HC_POST+3')]
#visualize(HC_pc=df_HC_pcs, HC_umap=df_HC_umaps, FS_pc=df_FS_pcs, FS_umap=df_FS_umaps, desired_contexts=contexts_to_visualize)


## MACHINE LEARNING ##

contexts_to_predict = ['HC_PRE','HC_POST']
rf(df_FS_umaps, contexts=contexts_to_predict, mode='UMAP')






