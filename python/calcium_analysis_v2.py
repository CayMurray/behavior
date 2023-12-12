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

eng = engine.start_matlab()


## CLASSES ##

class DataProcessor:

    def __init__(self,*args,reduce_list=['PCA']):
        self.datasets = args
        self.path = 'data'
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
    
class Analysis:

    def __init__(self,contexts_to_analyze):
        self.contexts = contexts_to_analyze

    def visualize_data(self,input):
        for (group,modes) in input.items():
            for mode,data in modes.items():
                data = data[data['context_ids'].isin(self.contexts)]
                fig,ax = plt.subplots(figsize=(15,10))
                sns.scatterplot(ax=ax,data=data,x=f'{mode}_1',y=f'{mode}_2',hue='context_ids',palette='tab10')

        plt.show()


## PROCESS AND ANALYZE DATA ##

context_processor = DataProcessor('FS','HC',reduce_list=['PCA'])
prepared_data = context_processor.prepare_data()

visualizer = Analysis(contexts_to_analyze=['HC_PRE','HC_POST'])
visualizer.visualize_data(prepared_data)




