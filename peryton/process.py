import pandas as pd
import numpy as np
from .mesh import get_Mesh_sim_mat
data = pd.read_csv('./peryton/Peryton-results.tsv',sep='\t')
filtered_data = data[['disease_name', 'microbe_scientific_name', 'relationship_name', 'disease_mesh_id', 'disease_mesh_heading']]

disease = filtered_data.iloc[:, [0,3,4]]
disease_u = disease.drop_duplicates().to_numpy()

microbe = filtered_data['microbe_scientific_name'].drop_duplicates().to_numpy()
disease_u.sort(axis=0)
microbe.sort()

meshids = list(map(lambda x: x.split(':')[-1].strip(), disease_u[:, 1]))
meshids1 = meshids
dis_sim = get_Mesh_sim_mat(meshids)

dis_names = disease_u[:, 0]

dis2mesh = {dis_names[i]:meshids1[i] for i in range(len(disease_u))}

dis_dic = {dis_names[i]:i for i in range(len(dis_names))}
mic_dic = {microbe[i]:i for i in range(len(microbe))}
filtered_data.to_csv('./peryton/filtered_data.csv')
np.save('./peryton/dis_dic.npy', dis_dic)
np.save('./peryton/mic_dic.npy', mic_dic)

ass = filtered_data.iloc[:, [0,1,2]].to_numpy()
A = np.zeros((microbe.shape[0], dis_names.shape[0]))
for item in ass:
    A[mic_dic[item[1]], dis_dic[item[0]]] = 1

for item in ass:
    if item[2] == 'Increased':
        A[mic_dic[item[1]], dis_dic[item[0]]] = 1

np.save('./peryton/mic_dis associatino.npy', A)

pd.DataFrame(A, index=microbe, columns=disease_u[:,0]).to_csv('./peryton/sign_final_ass.csv')


