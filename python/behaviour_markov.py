import numpy as np
import pandas as pd
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from pyvis.network import Network

df_master = pd.read_csv('data/pre.csv').drop(['time'],axis=1)

for col in df_master.columns:
    df = df_master[col]
    unique_states = df.unique()

    t_counts = {state:Counter() for state in unique_states}
    t_matrix = pd.DataFrame(index=unique_states,columns=unique_states,dtype=np.float64).fillna(0)

    for current,n in zip(df,df[1:]):
        t_counts[current][n] += 1

    for state in unique_states:
        t_total = sum(t_counts[state].values())
        
        for next_state in unique_states:
            if t_total > 0:
                t_matrix.at[state,next_state] = t_counts[state][next_state] / t_total

    G = nx.DiGraph()
    net = Network()

    for state in unique_states:
        G.add_node(state)

    for i in t_matrix.index:
        for j in t_matrix.columns:
            w = t_matrix.at[i,j]
            
            if w > 0:
                weight = -np.log(w)
                G.add_edge(i,j,weight=weight)
    
    #centrality_measures = dict(sorted(nx.degree_centrality(G).items(),key=lambda x: x[1],reverse=True))
    print(f'{col} irreducibility: {nx.is_strongly_connected(G)}')
    
    net.from_nx(G)
    #net.show('markov.html',notebook=False)
    
    