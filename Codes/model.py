from Codes.utils import *
from Codes.optimizer import *
from sklearn import metrics

class UnsupervisedClustering():


    def __init__(self, K ,
                max_iter = 10,
                indp_runs = 1,
                random_state=None
            ):
        """
        A class that collects all the components of different clustering methods.

        Parameters
        ----------

        K :             (int)       Number of clusters  
        max_iter:       (int)       Maximum number of iterations 
                                    MinMax clustering: iterations in the constraint generation process 
                                    K-means and K-means++: total number of iterations
        random_state:   (int)       Random state for initialization 


        Attributes
        ----------
        
        data:                   (pandas.DataFrame)                      Data after fitting with cluster labels               
        labels:                 (ndarray) of shape (n_samples,)         labels for the all the observations in the data
        cluster_centers:        (ndarray) of shape (K,n_dimensions)     Cluster centers found by the model
        binary_assign:          (ndarray) of shape (n_samples,K)        Indicator variables c_ij takes 1 if point 'i' is in cluster 'j'
        constr_gen_pts:         (list)                                  List of points for which we generated constraints   
        model_inertia:          (float)                                 SSE with the final labels
        model_optVal:           (float)                                 optimal value obtained from the MILP solver
        model_optGap:           (float)                                 optimality gap from the MILP solver at the final iteration


        The values of the below variables were extracted for the purpose of analyzing the MinMax clustering, specifically to provide the 
        constraint generation example plot and such results may not be necessary for other purposes 

        df_data_list:           (list) of pandas.DataFrame              A list of dataframes across multiple iterations 
        center_list:            (list) of ndarray                       list of centers across the iterations of constraint generation
        initContrs:             (list)                                  list of initial points for which we generate constraints and warm-start the MILP 
        trueMaxPts_list:        (list)                                  list of points at each iteration that are at the maximum distance from the centers 
        addPts_list:            (list) of list                          list of points for which we generate constraints at each iteration
        outliers:               (list)                                  list of points that are tagged as outliers


        """

        self.K = K
        self.max_iter = max_iter

        if random_state is not None:
            self.random_state=random_state  
        else: 
            self.random_state =  np.random.randint(0, 2**16-1)

        self.indp_runs = indp_runs
        self.model = None      # model object


    def set_model(self, model):
        """

        Parameters
        ----------

        model :         {"K_Means","K_MeansPlus","MinMax"} 
                        Defines the type of clustering       
        
        """
        self.model = model


    def fit(self, data):
        """
        Parameters
        ----------

        data :         (pandas.DataFrame) of shape (n_samples, n_dimensions)
        
        
        Returns
        -------
        self :          object 
                        after clustering the data                

        """  
        # print(data)    

        self.data = data  
        self.data, self.labels, self.cluster_centers, self.epsVal, self.binary_assign, self.constr_gen_pts, self.model_inertia, self.model_optVal, self.model_optGap, self.df_data_list, self.center_list, self.initContrs, self.trueMaxPts_list, self.addPts_list, self.outliers = self.model.optimize(self.data,self.K, max_iter = self.max_iter, random_state = self.random_state, indp_runs = self.indp_runs)
        # self.data, self.labels, self.cluster_centers = self.model.optimize(self.data,self.K, self.max_iter, self.random_state)
  
        return self

    
    def evaluate(self, true_labels, metric = 'all'):
        """
        Parameters
        ----------

        true_labels:        (ndarray) of (n_samples,) with the ground truth labels 
        metric:             {"all", "purity", "RI", "ARI", "NMI", "AMI", "homogeneity_score"} default is "all"
                            to evaluate the clustering using external clustering criteria
        
        Returns
        -------
        clusterEval :       (list)          list of evaluation metrics based on the input                

        """  

        clusterEval = []

        if metric == 'all' or metric == 'purity':
            contingency_matrix = metrics.cluster.contingency_matrix(true_labels, self.labels)
            clusterEval.extend([np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)])
        if metric == 'all' or metric == 'RI':
            clusterEval.extend([metrics.cluster.rand_score(true_labels, self.labels)])
        if metric == 'all' or metric == 'ARI':
            clusterEval.extend([metrics.cluster.adjusted_rand_score(true_labels, self.labels)])
        if metric == 'all' or metric == 'NMI':
            clusterEval.extend([metrics.cluster.normalized_mutual_info_score(true_labels, self.labels)])
        if metric == 'all' or metric == 'AMI':
            clusterEval.extend([metrics.cluster.adjusted_mutual_info_score(true_labels, self.labels)])
        if metric == 'all' or metric == 'homogeneity_score':
            clusterEval.extend([metrics.cluster.homogeneity_score(true_labels, self.labels)])
        
        return clusterEval
            
                

