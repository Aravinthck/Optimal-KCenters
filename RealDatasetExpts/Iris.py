
import os
import sys
path = os.path.dirname(os.getcwd())
sys.path.insert(0, path)

from Codes.model import *
from Codes.utils import *

from scipy import stats
import pickle

RESULT_DIR = "../../DatasetsResult"
DATA_DIR = "../../Datasets"
DATAINFO_DIR = "../../DatasetsInfo"

import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets


iris = datasets.load_iris()
X = iris.data  
y = iris.target
df = pd.DataFrame(np.c_[X,y], columns = ['SepalL','SepalW','PetalL', 'PetalW', 'y'])
df_data = pd.DataFrame(X, columns = ['X1','X2','X3', 'X4'])
K = 3

# KCenters - Gonzalez 


kc_Gon = UnsupervisedClustering(K =K)
kc_Gon.set_model(KCenters_Gon())

start = default_timer()
kc_Gon.fit(df_data )

end = default_timer()
   
metrics_list = kc_Gon.evaluate(y)
epsVal_kcgon = kc_Gon.epsVal

metrics_list = np.array(metrics_list) 

centers_kcgon = kc_Gon.cluster_centers

ari_kcgon = metrics_list[2]

print("\n\nResults for KCenter-Gon" )

print("\nARI result for KCenter-Gon: " , ari_kcgon)

purity_kcgon = metrics_list[0]
print("Purity result for KCenter-Gon: ", purity_kcgon)

nmi_kcgon = metrics_list[3]
print("NMI result for KCenter-Gon: ", nmi_kcgon)

print("EpsVal result for KCenter-Gon: ", epsVal_kcgon)

time_result_kcgon = end-start
print("Time taken for KCenter-Gon: ", time_result_kcgon, "\n")


# KCenters - Gonzalez Average

centers_list_gonavg = []

metrics_list = []
time_list=[]
epsVal_list = []

for j in range(10):

    kc_Gon = UnsupervisedClustering(K =K)
    kc_Gon.set_model(KCenters_Gon())

    start = default_timer()
    kc_Gon.fit(df_data )

    end = default_timer()

    time_list.append(end-start) 
    metrics_list.append(kc_Gon.evaluate(y))
    epsVal_list.append(kc_Gon.epsVal)
    centers_list_gonavg.append(kc_Gon.cluster_centers)

    

metrics_list = np.array(metrics_list) 

print("\n\nResults for KCenter-GonAvg" )


print('\nMean, sd, low_ci, high_ci')

ari = metrics_list[:,2]
ari_ci_low, ari_ci_high = stats.t.interval(0.95, len(ari)-1, loc=np.mean(ari), scale=stats.sem(ari))
ari_result_kcgonavg = [np.mean(ari), np.std(ari), ari_ci_low, ari_ci_high]
print("\nARI result for KCenter-GonAvg: " , ari_result_kcgonavg)

purity = metrics_list[:,0]
purity_ci_low, purity_ci_high = stats.t.interval(0.95, len(purity)-1, loc=np.mean(purity), scale=stats.sem(purity))
purity_result_kcgonavg = [np.mean(purity), np.std(purity), purity_ci_low, purity_ci_high]
print("Purity result for KCenter-GonAvg: ", purity_result_kcgonavg)

nmi = metrics_list[:,3]
nmi_ci_low, nmi_ci_high = stats.t.interval(0.95, len(nmi)-1, loc=np.mean(nmi), scale=stats.sem(nmi))
nmi_result_kcgonavg = [np.mean(nmi), np.std(nmi), nmi_ci_low, nmi_ci_high]
print("NMI result for KCenter-GonAvg: ", nmi_result_kcgonavg)


epsVal_ci_low, epsVal_ci_high = stats.t.interval(0.95, len(epsVal_list)-1, loc=np.mean(epsVal_list), scale=stats.sem(epsVal_list))
epsVal_result_kcgonavg = [np.mean(epsVal_list), np.std(epsVal_list), epsVal_ci_low, epsVal_ci_high]
print("EpsVal for KCenter-GonAvg: ", epsVal_result_kcgonavg)


time_result_kcgonavg = np.mean(time_list)
print("Average time taken for KCenter-GonAvg: ", time_result_kcgonavg, "\n")





# KCenters - HS 


kc_hs = UnsupervisedClustering(K =K)
kc_hs.set_model(KCenter_HS())

start = default_timer()
kc_hs.fit(df_data )

end = default_timer()
   
metrics_list = kc_hs.evaluate(y)
epsVal_kchs = kc_hs.epsVal
centers_kchs = kc_hs.cluster_centers


metrics_list = np.array(metrics_list) 

ari_kchs = metrics_list[2]
print("\n\nResults for KCenter-HS" )

print("\nARI result for KCenter-HS: " , ari_kchs)

purity_kchs = metrics_list[0]
print("Purity result for KCenter-HS: ", purity_kchs)

nmi_kchs = metrics_list[3]
print("NMI result for KCenter-HS: ", nmi_kchs)

print("EpsVal result for KCenter-HS: ", epsVal_kchs)

time_result_kchs = end-start
print("Time taken for KCenter-HS: ", time_result_kchs, "\n")



# KMedian

centers_list_km = []
metrics_list = []
time_list=[]
epsVal_list = []

for j in range(10):

    km = UnsupervisedClustering(K =K, max_iter=100)
    km.set_model(KMedian())

    start = default_timer()
    km.fit(df_data )

    end = default_timer()

    time_list.append(end-start) 
    metrics_list.append(km.evaluate(y))
    epsVal_list.append(km.epsVal)
    centers_list_km.append(km.cluster_centers)

    
print("\n\nResults for KMedian" )


metrics_list = np.array(metrics_list) 

print('\n Mean, sd, low_ci, high_ci')

ari = metrics_list[:,2]
ari_ci_low, ari_ci_high = stats.t.interval(0.95, len(ari)-1, loc=np.mean(ari), scale=stats.sem(ari))
ari_result_km = [np.mean(ari), np.std(ari), ari_ci_low, ari_ci_high]
print("\nARI result for KMedian: " , ari_result_km)

purity = metrics_list[:,0]
purity_ci_low, purity_ci_high = stats.t.interval(0.95, len(purity)-1, loc=np.mean(purity), scale=stats.sem(purity))
purity_result_km = [np.mean(purity), np.std(purity), purity_ci_low, purity_ci_high]
print("Purity result for KMedian: ", purity_result_km)

nmi = metrics_list[:,3]
nmi_ci_low, nmi_ci_high = stats.t.interval(0.95, len(nmi)-1, loc=np.mean(nmi), scale=stats.sem(nmi))
nmi_result_km = [np.mean(nmi), np.std(nmi), nmi_ci_low, nmi_ci_high]
print("NMI result for KMedian: ", nmi_result_km)

epsVal_ci_low, epsVal_ci_high = stats.t.interval(0.95, len(epsVal_list)-1, loc=np.mean(epsVal_list), scale=stats.sem(epsVal_list))
epsVal_result_km = [np.mean(epsVal_list), np.std(epsVal_list), epsVal_ci_low, epsVal_ci_high]
print("EpsVal for KMedian: ", epsVal_result_km)

time_result_km = np.mean(time_list)
print("Average time taken for KMedian: ", time_result_km, "\n")


# KMedian with Kmeanplus 

metrics_list = []
time_list=[]
epsVal_list = []
centers_list_kmplus = []
for j in range(10):

    kmplus = UnsupervisedClustering(K =K, max_iter=100)
    kmplus.set_model(KMedian_Plus())

    start = default_timer()
    kmplus.fit(df_data )

    end = default_timer()

    time_list.append(end-start) 
    metrics_list.append(kmplus.evaluate(y))
    epsVal_list.append(kmplus.epsVal)
    centers_list_kmplus.append(kmplus.cluster_centers)
    

metrics_list = np.array(metrics_list) 
print("\n\nResults for KMedianPlus" )

print('\nMean, sd, low_ci, high_ci')

ari = metrics_list[:,2]
ari_ci_low, ari_ci_high = stats.t.interval(0.95, len(ari)-1, loc=np.mean(ari), scale=stats.sem(ari))
ari_result_kmplus = [np.mean(ari), np.std(ari), ari_ci_low, ari_ci_high]
print("\nARI result for KMedianPlus: " , ari_result_kmplus)

purity = metrics_list[:,0]
purity_ci_low, purity_ci_high = stats.t.interval(0.95, len(purity)-1, loc=np.mean(purity), scale=stats.sem(purity))
purity_result_kmplus = [np.mean(purity), np.std(purity), purity_ci_low, purity_ci_high]
print("Purity result for KMedianPlus: ", purity_result_kmplus)

nmi = metrics_list[:,3]
nmi_ci_low, nmi_ci_high = stats.t.interval(0.95, len(nmi)-1, loc=np.mean(nmi), scale=stats.sem(nmi))
nmi_result_kmplus = [np.mean(nmi), np.std(nmi), nmi_ci_low, nmi_ci_high]
print("NMI result for KMedianPlus: ", nmi_result_kmplus)

epsVal_ci_low, epsVal_ci_high = stats.t.interval(0.95, len(epsVal_list)-1, loc=np.mean(epsVal_list), scale=stats.sem(epsVal_list))
epsVal_result_kmplus = [np.mean(epsVal_list), np.std(epsVal_list), epsVal_ci_low, epsVal_ci_high]
print("EpsVal for KMedianPlus: ", epsVal_result_kmplus)

time_result_kmplus = np.mean(time_list)
print("Average time taken for KMedianPlus: ", time_result_kmplus, "\n")


# k-MinMax 


mmOpt = UnsupervisedClustering(K =3 , max_iter= 10, random_state=0)
mmOpt.set_model(MinMax(initConstrCnt=5, optimalGap=0, tol=0.005 ))

start = default_timer()

mmOpt.fit(df_data )

end = default_timer()

metric_result = mmOpt.evaluate(y)
epsVal_mmopt = mmOpt.epsVal
optgap_mmopt = mmOpt.model_optGap
centers_mmopt = mmOpt.cluster_centers
optgap_mmopt = mmOpt.model_optGap
optVal_mmopt = mmOpt.model_optVal



print("\nResults for MM-Opt" )

ari_result_mmopt= metric_result[2]

print("\nARI result for MM-Opt: " , ari_result_mmopt)

purity_result_mmopt = metric_result[0]
print("Purity result for MM-Opt: ", purity_result_mmopt)

nmi_result_mmopt = metric_result[3]
print("NMI result for MM-Opt: ", nmi_result_mmopt)

print("EpsVal result for MM-Opt: ", epsVal_mmopt)

time_result_mmopt = end-start
print('time taken: ', time_result_mmopt)



# Save the final results 


with open(os.path.join(RESULT_DIR, 'Iris.pkl'), 'wb') as outp:

    pickle.dump(ari_kcgon, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(purity_kcgon, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(nmi_kcgon, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(epsVal_kcgon, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(centers_kcgon, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(time_result_kcgon, outp, pickle.HIGHEST_PROTOCOL)

    pickle.dump(ari_result_kcgonavg, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(purity_result_kcgonavg, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(nmi_result_kcgonavg, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(epsVal_result_kcgonavg, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(centers_list_gonavg, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(time_result_kcgonavg, outp, pickle.HIGHEST_PROTOCOL)


    pickle.dump(ari_kchs, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(purity_kchs, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(nmi_kchs, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(epsVal_kchs, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(centers_kchs, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(time_result_kchs, outp, pickle.HIGHEST_PROTOCOL)


    pickle.dump(ari_result_km, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(purity_result_km, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(nmi_result_km, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(epsVal_result_km, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(centers_list_km, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(time_result_km, outp, pickle.HIGHEST_PROTOCOL)


    pickle.dump(ari_result_kmplus, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(purity_result_kmplus, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(nmi_result_kmplus, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(epsVal_result_kmplus, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(centers_list_kmplus, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(time_result_kmplus, outp, pickle.HIGHEST_PROTOCOL)

    pickle.dump(ari_result_mmopt, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(purity_result_mmopt, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(nmi_result_mmopt, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(epsVal_mmopt, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(centers_mmopt, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(time_result_mmopt, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(optgap_mmopt, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(optVal_mmopt, outp, pickle.HIGHEST_PROTOCOL)




n, d = X.shape

with open(os.path.join(DATAINFO_DIR, 'Iris.pkl'), 'wb') as outp:

    pickle.dump(n, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(d, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(K, outp, pickle.HIGHEST_PROTOCOL)






