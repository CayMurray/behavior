import os
import numpy as np
import pandas as pd
from matlab import engine
from collections import Counter

eng = engine.start_matlab()

class DataProcessor:

    def __init__(self,*args):
        self.datasets = args
        self.path = 'data'
        self.dict = {}
        self.labels = ['US_PRE','FS','US+1','US+2','US+3',
        'HC_PRE','HC_POST','HC_PRE+1','HC_POST+1',
        'HC_PRE+2','HC_POST+2','HC_PRE+3','HC_POST+3']

    def double_to_list(self,matlab_array,num_type):
        return list(np.array(matlab_array,dtype=num_type).reshape(-1))

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

            df_master = pd.DataFrame(data=self.dict[group]['data'],columns=context_list)
            df_master['rat_id'] = self.dict[group]['rat_indices']

memes = DataProcessor('FS','HC')
memes.load_data()
memes.construct_df()