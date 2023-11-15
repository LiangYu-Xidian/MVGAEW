import numpy as np
import pandas as pd

mdad = pd.read_excel('./graph2mda/MDAD/microbes.xlsx')
mdad.columns = ['1', '2', '3']
mdadmap = mdad[mdad['3'].notna()]
f_id = mdadmap['1'].to_numpy()
f_id_0 = np.array([item - 1 for item in f_id])
adj = np.loadtxt('./graph2mda/MDAD/adj.txt', dtype=int)
microbe = adj[:, 1].max()
drug = adj[:, 0].max()

A = np.zeros((int(microbe), int(drug)))
for item in adj:
    A[item[1] - 1, item[0] - 1] = 1
# np.save('./graph2mda/MDAD/mic_drug_ass.npy', A)

drug_sim = pd.read_csv('./graph2mda/MDAD/drug_similarity.txt', sep='\t', header=None).to_numpy()
mic_fun_mada = mic_fun_DSS(drug_sim, A)
# np.save('./graph2mda/MDAD/mic_fun_mdad.npy', mic_fun_mada)
f_fun_sim = mic_fun_mada[f_id_0, :]
filtered_sim = f_fun_sim[:, f_id_0]

final_sim = np.zeros((292, 292))
temp_map = {f_id_0[i]: i for i in range(filtered_sim.shape[0])}
for i in range(292):
    if i in f_id_0:
        for j in range(292):
            if j in f_id_0:
                final_sim[i, j] = filtered_sim[temp_map[i], temp_map[j]]
# np.save('./graph2mda/MDAD/filtered_mic_fun_mdad.npy', final_sim)
# np.save('./graph2mda/MDAD/index_in_HMDAD.npy', f_id_0)

abio_sim = np.load('./graph2mda/aBiofilm/filtered_mic_fun_abio.npy')
mdad_sim = np.load('./graph2mda/MDAD/filtered_mic_fun_mdad.npy')
converge = np.zeros_like(abio_sim)
for i in range(292):
    for j in range(292):
        if abio_sim[i, j] != 0 and mdad_sim[i, j] != 0:
            converge[i, j] = (abio_sim[i, j] + mdad_sim[i, j]) / 2
        elif abio_sim[i, j] != 0 and mdad_sim[i, j] == 0:
            converge[i, j] = abio_sim[i, j]
        elif abio_sim[i, j] == 0 and mdad_sim[i, j
        ] != 0:
            converge[i, j] = mdad_sim[i, j]

microbe1 = pd.read_csv('./graph2mda/temp.csv', index_col=0)
microbe_names = microbe1['0'].to_numpy()
final = pd.DataFrame(converge)
final.columns = microbe_names
final.index = microbe_names
# final.to_csv('./graph2mda/microbe_fun_sim_mdad_abio.csv')