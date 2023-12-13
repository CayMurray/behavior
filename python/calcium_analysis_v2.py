## MODULES AND ENGINE ##

import os
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
from sklearn.metrics import classification_report,confusion_matrix,homogeneity_score,completeness_score

eng = engine.start_matlab()


## CLASSES ##

class DimensionReduction:
    
    def __init__(self,*args):
        self.modes = args

    def get_components(self,data):
        reduced_dict = {}
        contexts = data.columns[:-1]
        data = data.drop(['rat_id'],axis=1).T

        for mode in self.modes:
            z_scaler = StandardScaler()
            scaled_data = z_scaler.fit_transform(data)

            if mode == 'PCA':
                algorithm = PCA(n_components=2)
            elif mode == 'UMAP':
                algorithm = umap.UMAP(n_components=2,random_state=42)

            reduced_components = algorithm.fit_transform(scaled_data)
            df_reduced = pd.DataFrame(data=reduced_components,columns=[f'{mode}_1',f'{mode}_2'])
            df_reduced['context_ids'] = contexts
            reduced_dict[mode] = df_reduced

        return reduced_dict
    

class RandomForest:

    def __init__(self,data,desired_contexts=['HC_PRE','HC_POST'],num_trees=100,random_state=42):
        self.dict = data
        self.num_trees = num_trees
        self.random_state = random_state
        self.contexts = desired_contexts
        
    def train_rf(self):
        for (group,modes) in self.dict.items():
                for mode,data in modes.items():
                    data = data[data['context_ids'].isin(self.contexts)]
                    X = data.drop(['context_ids'],axis=1)
                    Y = data['context_ids']
                    X_train,x_test,Y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=self.random_state,shuffle=True)
                    rf = RandomForestClassifier(n_estimators=self.num_trees,random_state=self.random_state)
                    rf.fit(X_train,Y_train)
                    predicted = rf.predict(x_test)
                    cm = confusion_matrix(y_test,predicted)
                    unique_labels = sorted(set(Y))

                    fig,ax = plt.subplots(figsize=(20,10))
                    sns.heatmap(ax=ax, data=cm, fmt='g',annot=True, cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
                    ax.set_title(f'{group} {mode}',pad=20,fontsize=20)
                    ax.set_xlabel('Predicted',labelpad=20,fontsize=20)

        plt.show()


class CorrelationAnalysis:

    def __init__(self,input):
        self.dict = input
        self.contexts = ['HC_PRE','HC_POST',
                         'HC_POST+1','HC_POST+2','HC_POST+3']

    def get_vectors(self):
        for (group,modes) in self.dict.items():
                for mode,data in modes.items():
                    contexts = self.contexts.copy()

                    if group == 'non-FS':
                        contexts = contexts.pop(0)

                    data = data[data['context_ids'].isin(contexts)]




class DataProcessor:

    def __init__(self,*args,reduce_list=['PCA']):
        self.datasets = args
        self.path = 'data/calcium'
        self.dict = {}
        self.labels = ['US_PRE','FS','US+1','US+2','US+3',
        'HC_PRE','HC_POST','HC_PRE+1','HC_POST+1',
        'HC_PRE+2','HC_POST+2','HC_PRE+3','HC_POST+3']
        self.reduce_list = reduce_list

    def prepare_data(self):
        self.load_data()
        self.construct_df()
        self.apply_dimension_reduction()

        return self.dict

    def load_data(self):
        for group in self.datasets:
            data = eng.load(f'{self.path}/{group}.mat')
            self.dict[group] = {}
            self.dict[group]['data'] = np.array(data['TOTMAT'],dtype=np.float32)
            self.dict[group]['ctimes'] = self.double_to_list(data['ctimes'],np.uint32)
            self.dict[group]['rat_indices'] = self.double_to_list(data['anim_index'],np.uint8)

    def construct_df(self):
        for (group,d) in self.dict.items():
            context_list = []
            labels = self.labels.copy()

            if group != 'FS':
                labels.pop(1)

            intervals = [(1,d['ctimes'][0])]
            intervals.extend((d['ctimes'][i]+1,d['ctimes'][i+1]) for i in range(len(d['ctimes'])-1))
            contexts_and_intervals = dict(zip(labels,intervals))
            
            for val, (context, (start, end)) in enumerate(contexts_and_intervals.items()):
                num_of_times = end - start + 1
                context_list.extend([context] * num_of_times)

            df_master = pd.DataFrame(data=d['data'],columns=context_list)
            df_master['rat_id'] = self.dict[group]['rat_indices']
            self.dict[group]['data'] = df_master

    def apply_dimension_reduction(self):
        for group in self.datasets:
            reducer = DimensionReduction(*self.reduce_list)
            df_input = self.dict[group]['data']
            reduced_dict = reducer.get_components(df_input)
            self.dict[group].update(reduced_dict)
            
            for context in ['data','ctimes','rat_indices']:
                del self.dict[group][context]

    @staticmethod
    def double_to_list(matlab_array,num_type):
        return list(np.array(matlab_array,dtype=num_type).reshape(-1))
    

class Analysis:

    def __init__(self,contexts_to_analyze):
        self.contexts = contexts_to_analyze

    def visualize_data(self,input):
        for (group,modes) in input.items():
            for mode,data in modes.items():
                data = data[data['context_ids'].isin(self.contexts)]
                fig,ax = plt.subplots(figsize=(15,10))
                sns.scatterplot(ax=ax,data=data,x=f'{mode}_1',y=f'{mode}_2',hue='context_ids',palette='tab10')
                ax.set_title(f'{group} - {mode}')
        plt.show()

    def predict_labels(self,input):
        classifier_instance = RandomForest(input,self.contexts)
        classifier_instance.train_rf()


## PROCESS AND ANALYZE DATA ##

context_processor = DataProcessor('FS',reduce_list=['PCA'])
#prepared_data = context_processor.prepare_data()

#analyzer = Analysis(contexts_to_analyze=['HC_PRE','HC_POST'])
#analyzer.visualize_data(prepared_data)
#analyzer.predict_labels(prepared_data)


