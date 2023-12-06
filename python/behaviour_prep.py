import pandas as pd

df = pd.read_excel('data/Sabin.xlsx',sheet_name='Sated_Beh')
columns_to_drop = [df.columns[i] for i in range(len(df.columns)) if i%3==2]
df = df.drop(columns_to_drop,axis=1).iloc[1:,:]

rat_id = 0
new_columns = []

for i,column in enumerate(df.columns):

    if i%2 == 0:
        new_columns.append(f'behaviour_{rat_id}')
    else:
        new_columns.append(f'transition_{rat_id}')
        rat_id+=1

df.columns = new_columns
discrete_states = []

for i in range(int(len(df.columns)/2)): 
    current_behaviour = df[f'behaviour_{i}'].iloc[0]
    indv_dict = {f'rat_{i}':[current_behaviour]}
    
    for t in range(0,600,10):
        filtered_df=df[(df[f'transition_{i}'] >= t) & (df[f'transition_{i}'] < t+10)].filter(like=str(i))
        
        if not filtered_df.empty:
            current_behaviour = filtered_df[f'behaviour_{i}'].iloc[-1]
    
        indv_dict[f'rat_{i}'].append(current_behaviour)

    discrete_states.append(pd.DataFrame(indv_dict))

df_time = pd.DataFrame({'time':[i for i in range(0,610,10)]})
df_final = pd.concat([df_time,*discrete_states],axis=1)
df_final.to_csv('data/sabin_pre.csv',index=False)
