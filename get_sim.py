from utils import *


def get_3ss_GIP_dis_drug(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim3 = mic_fun_DSS(disease_sim1, A)
    microbe_sim4 = pd.read_csv('./graph2mda/microbe_fun_sim_mdad_abio.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim2, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    m2 = GIP_m_sim
    m3 = new_normalization(microbe_sim3)
    m4 = new_normalization(microbe_sim4)

    # Sm_1 = KNN_kernel(microbe_sim1, k1)
    Sm_2 = KNN_kernel(GIP_m_sim, k1)
    Sm_3 = KNN_kernel(microbe_sim3, k1)
    Sm_4 = KNN_kernel(microbe_sim4, k1)

    Pm = MiRNA_updating2(Sm_2, Sm_3, Sm_4, m2, m3, m4)
    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_3ss_GIP_dis(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim3 = mic_fun_DSS(disease_sim1, A)
    # microbe_sim4 = pd.read_csv('./graph2mda/microbe_fun_sim_mdad_abio.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim2, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    m2 = GIP_m_sim
    m3 = new_normalization(microbe_sim3)
    # m4 = new_normalization(microbe_sim4)

    # Sm_1 = KNN_kernel(microbe_sim1, k1)
    Sm_2 = KNN_kernel(GIP_m_sim, k1)
    Sm_3 = KNN_kernel(microbe_sim3, k1)
    # Sm_4 = KNN_kernel(microbe_sim4, k1)

    Pm = MiRNA_updating(Sm_2, Sm_3, m2, m3)
    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_3ss_GIP_drug(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    # microbe_sim3 = mic_fun_DSS(disease_sim1, A)
    microbe_sim4 = pd.read_csv('./graph2mda/microbe_fun_sim_mdad_abio.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim2, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    m2 = GIP_m_sim
    # m3 = new_normalization(microbe_sim3)
    m4 = new_normalization(microbe_sim4)

    # Sm_1 = KNN_kernel(microbe_sim1, k1)
    Sm_2 = KNN_kernel(GIP_m_sim, k1)
    # Sm_3 = KNN_kernel(microbe_sim3, k1)
    Sm_4 = KNN_kernel(microbe_sim4, k1)

    Pm = MiRNA_updating(Sm_2, Sm_4, m2, m4)
    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_3ss_GIP_blastn_dis(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim1 = pd.read_csv('./microbe_seq/blast_mean_sim.csv', index_col=[0]).to_numpy()
    microbe_sim3 = mic_fun_DSS(disease_sim1, A)
    # microbe_sim4 = pd.read_csv('./graph2mda/microbe_fun_sim_mdad_abio.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim1, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    m1 = new_normalization(microbe_sim1)
    # m2 = new_normalization(GIP_m_sim)
    m2 = GIP_m_sim
    m3 = new_normalization(microbe_sim3)
    # m4 = new_normalization(microbe_sim4)

    Sm_1 = KNN_kernel(microbe_sim1, k1)
    Sm_2 = KNN_kernel(GIP_m_sim, k1)
    Sm_3 = KNN_kernel(microbe_sim3, k1)
    # Sm_4 = KNN_kernel(microbe_sim4, k1)

    Pm = MiRNA_updating2(Sm_1, Sm_2, Sm_3, m1, m2, m3)
    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_3ss_GIP_blastn_drug(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim1 = pd.read_csv('./microbe_seq/blast_mean_sim.csv', index_col=[0]).to_numpy()
    # microbe_sim3 = mic_fun_DSS(disease_sim1, A)
    microbe_sim4 = pd.read_csv('./graph2mda/microbe_fun_sim_mdad_abio.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim1, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    m1 = new_normalization(microbe_sim1)
    # m2 = new_normalization(GIP_m_sim)
    m2 = GIP_m_sim
    # m3 = new_normalization(microbe_sim3)
    m4 = new_normalization(microbe_sim4)

    Sm_1 = KNN_kernel(microbe_sim1, k1)
    Sm_2 = KNN_kernel(GIP_m_sim, k1)
    # Sm_3 = KNN_kernel(microbe_sim3, k1)
    Sm_4 = KNN_kernel(microbe_sim4, k1)

    Pm = MiRNA_updating2(Sm_1, Sm_2, Sm_4, m1, m2, m4)
    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_3ss_dis_drug(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim3 = mic_fun_DSS(disease_sim1, A)
    microbe_sim4 = pd.read_csv('./graph2mda/microbe_fun_sim_mdad_abio.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim2, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    # m2 = GIP_m_sim
    m3 = new_normalization(microbe_sim3)
    m4 = new_normalization(microbe_sim4)

    # Sm_1 = KNN_kernel(microbe_sim1, k1)
    # Sm_2 = KNN_kernel(GIP_m_sim, k1)
    Sm_3 = KNN_kernel(microbe_sim3, k1)
    Sm_4 = KNN_kernel(microbe_sim4, k1)

    Pm = MiRNA_updating(Sm_3, Sm_4, m3, m4)
    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_3ms_GIP_symptom(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim3 = mic_fun_DSS(disease_sim1, A)
    microbe_sim4 = pd.read_csv('./graph2mda/microbe_fun_sim_mdad_abio.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    # d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    # Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim2, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    # Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd = disease_updating(Sd_2, Sd_3, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    m2 = GIP_m_sim
    m3 = new_normalization(microbe_sim3)
    m4 = new_normalization(microbe_sim4)

    Sm_2 = KNN_kernel(GIP_m_sim, k1)
    Sm_3 = KNN_kernel(microbe_sim3, k1)
    Sm_4 = KNN_kernel(microbe_sim4, k1)

    Pm = MiRNA_updating2(Sm_2, Sm_3, Sm_4, m2, m3, m4)
    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_3ms_GIP_DSS(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    # disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim3 = mic_fun_DSS(disease_sim1, A)
    microbe_sim4 = pd.read_csv('./graph2mda/microbe_fun_sim_mdad_abio.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    # d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    # Sd_2 = KNN_kernel(disease_sim2, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    # Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd = disease_updating(Sd_1, Sd_3, d1, d3)
    Pd_final = (Pd + Pd.T) / 2

    m2 = GIP_m_sim
    m3 = new_normalization(microbe_sim3)
    m4 = new_normalization(microbe_sim4)

    Sm_2 = KNN_kernel(GIP_m_sim, k1)
    Sm_3 = KNN_kernel(microbe_sim3, k1)
    Sm_4 = KNN_kernel(microbe_sim4, k1)

    Pm = MiRNA_updating2(Sm_2, Sm_3, Sm_4, m2, m3, m4)
    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_3ms_DSS_symptom(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim3 = mic_fun_DSS(disease_sim1, A)
    microbe_sim4 = pd.read_csv('./graph2mda/microbe_fun_sim_mdad_abio.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    # d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim2, k2)
    # Sd_3 = KNN_kernel(GIP_d_sim, k2)
    # Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd = disease_updating(Sd_1, Sd_2, d1, d2)
    Pd_final = (Pd + Pd.T) / 2

    m2 = GIP_m_sim
    m3 = new_normalization(microbe_sim3)
    m4 = new_normalization(microbe_sim4)

    Sm_2 = KNN_kernel(GIP_m_sim, k1)
    Sm_3 = KNN_kernel(microbe_sim3, k1)
    Sm_4 = KNN_kernel(microbe_sim4, k1)

    Pm = MiRNA_updating2(Sm_2, Sm_3, Sm_4, m2, m3, m4)
    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_sim_3ss_1ms_disbiome(A, k1, k2):
    disease_sim1 = mesh_sim = pd.read_csv('./Disbiome/mesh/mesh_sim.csv', index_col=0).to_numpy()
    disease_sim2 = symptom_sim = pd.read_csv('./Disbiome/symptom/final_symptom_disease_bio.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim2, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    return GIP_m_sim, Pd_final


def get_sim_3ss_1ms_phendb(A, k1, k2):
    disease_sim1 = mesh_sim = pd.read_csv('./phendb/mesh/final_mesh_sim.csv', index_col=0).to_numpy()
    disease_sim2 = symptom_sim = pd.read_csv('./phendb/symptom/final_symptom_disease_phendb.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim2, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    return GIP_m_sim, Pd_final


def get_sim_3ss_1ms_peryton(A, k1, k2):
    disease_sim1 = mesh_sim = pd.read_csv('./peryton/DSS.csv', index_col=0).to_numpy()
    disease_sim2 = symptom_sim = pd.read_csv('./peryton/symptom_disease_peryton.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim2, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2
    # np.save('d_sim_final.npy', Pd_final)
    # np.save('d_sim_final.txt', Pd_final)
    # return Pm_final, Pd_final

    return GIP_m_sim, Pd_final






