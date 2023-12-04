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


def get_pcs(df_master):

    z_scaler = StandardScaler()
    scaled_data = z_scaler.fit_transform(df_master.T.values)
    pca = PCA(n_components=2)

    calcium_pcs = pca.fit_transform(scaled_data)
    df_pc = pd.DataFrame(data=calcium_pcs,columns=['PC1','PC2'])
    context_ids = [context for context in df_master.columns]

    desired_contexts = ['HC_POST+2','HC_POST+3']
    df_pc['context_ids'] = context_ids
    df_pc = df_pc[df_pc['context_ids'].isin(desired_contexts)]

    sns.scatterplot(data=df_pc,x='PC1',y='PC2',hue='context_ids',palette='tab10')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Context', bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    return df_pc


def rf(df_pc):

    X = df_pc[['PC1','PC2']]
    Y = df_pc[['context_ids']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    unique_labels = sorted(set(y_test['context_ids']))

    rf = RandomForestClassifier(n_estimators=100,random_state=42)
    rf.fit(X_train,y_train)
    predictions = rf.predict(X_test)
    cm = confusion_matrix(y_test,predictions)

    print(classification_report(y_test,predictions))
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, fmt='g',annot=True, cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


df_HC = convert_to_pd(path=HC_path,context_labels=HC_labels)
df_FS = convert_to_pd(path=FS_path,context_labels=FS_labels)

df_FS_pcs = get_pcs(df_FS)
df_HC_pcs = get_pcs(df_HC)

rf(df_FS_pcs)






