import numpy as np
import pandas as pd

data = pd.read_json('./Disbiome/association.txt')
columns = np.array(data.columns).reshape(-1)
filtered_columns = columns[[2,3,7,8,9]]
filtered_data = data[filtered_columns].drop_duplicates()

# filtered_data.to_csv('./Disbiome/filtered_data.csv')
disease = pd.read_json('./Disbiome/disease.txt')
microbe = pd.read_json('./Disbiome/organism.txt')
disease_temp = disease[['disease_id', 'name']]
# disease_temp.to_csv('./Disbiome/temp_disease.csv')
microbe_temp = microbe[['organism_id', 'name']]
# microbe_temp.to_csv('./Disbiome/microbe_temp.csv')

dis_temp = disease['disease_id'].to_numpy()
dis_map = {dis_temp[i]:i for i in range(len(dis_temp))}
mic_temp = microbe['organism_id']
mic_map = {mic_temp[i]:i for i in range(len(mic_temp))}

dis_map_re = {v:k for k, v in dis_map.items()}
mic_map_re = {v:k for k, v in mic_map.items()}

ass = filtered_data[['disease_id', 'organism_id', 'qualitative_outcome']].to_numpy()
A = np.zeros((1622, 374))

for item in ass:
    if item[-1] == 'Elevated':
        A[mic_map[item[1]], dis_map[item[0]]] = 1
    else:
        A[mic_map[item[1]], dis_map[item[0]]] = -1

microbe_names = microbe_temp['name'].to_numpy()
disease_names = disease_temp['name'].to_numpy()

A_pd = pd.DataFrame(A, index=microbe_names, columns=disease_names)









