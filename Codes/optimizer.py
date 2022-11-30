
from abc import ABC, abstractmethod
from Codes.utils import *
from Codes.baselineHelper import *


class LossOptimization(ABC):


    
    @abstractmethod
    # def optimize(self, data, K ,max_iter, random_state):
    def optimize(self, data, K ,*args, **kwargs):


        raise NotImplementedError



class MinMax(LossOptimization):


    def __init__(self, WarmStart = True,
                outliersCnt = 0,
                initConstrCnt = 10,
                time = 1,
                optimalGap = 0.05,
                tol = 0.005,
                use_bigM = False,
                use_errorBd = False,
                bigM = 10000,
                initConstrIntRatio = 2,
                outputFlag = False):

        """
        A class that performs kc-Opt algorithm

        Parameters
        ----------

        WarmStart :             (bool)      If we want to warm start the centers for the MILP at first iteration 
        outliersCnt:            (int)       Number of outliers (l)
        initConstrCnt:          (int)       Number of points (per cluster) for which we initialize the constraints set
        time:                   (int)       Time in minutes for which we want the MILP to run per every constraint generation iteration
        optimalGap:             (float)     Value between [0,1] to set the maximum optimality gap we expect from the MILP solver
        tol:                    (float)     An acceptable tolerance between the optimal value from MILP solver and true value of the max
                                            distance from any point to its cluster center for the purpose of exiting the iterations 
        use_bigM:               (bool)      default: "False" 
                                            If we want to use big M method instead of indicator constraints. big M method is not to be preferred
                                            over the use of indicator constraints.
        use_errorBd:            (bool)      default: "False"
                                            Adds additional variables to the MILP that are not necessary for our MILP formulation. All our experiments 
                                            are run with this in default value of False. 
        bigM:                   (float)     big M value if use_bigM is "True"
        initConstrIntRatio:     (int)       Ratio that decides the number of internal points (initConstrCnt/initConstrIntRatio)
                                            per cluster for which we initialize the constraints set

        outputFlag:             (bool)      default: "False"
                                            if True, returns the progress output from the MILP solver



        
        """


        
        self.time = time
        
        self.WarmStart = WarmStart
        self.outliersCnt = outliersCnt
        self.initConstrCnt = initConstrCnt
        self.initConstrIntRatio = initConstrIntRatio
        self.optimalGap = optimalGap
        self.tol = tol
        self.use_bigM = use_bigM
        self.use_errorBd = use_errorBd
        self.bigM = bigM
        self.outputFlag = outputFlag


    def optimize(self, data, K, max_iter, random_state,*args, **kwargs):

        """
        A function that performs the constraint generation methodology for MinMax clustering     

        """

        data = data.copy()
        X = data.to_numpy()
        n,f = X.shape
        
        self.randState = random_state
        self.max_iter = max_iter
        print("MinMax model", self.outliersCnt)

        print("# of outliers: ", self.outliersCnt)

        # centers list across iterations 

        center_list = []

        # outliers list

        trueOutliers_list = []
        outliers_list = []

        counter = 0

        # list of points that were added during the constraint generation iterations 

        addPts_list = [] 

        maxPts_list = []                # Gurobi returned max points
        trueMaxPts_list = []            # True max points excluding outliers

        # list of data with point to cluster assignment information

        df_data_list = []


        # a small epsilon value for ensuring values of centers in their first dimension are increasing 
        eps = 0.001          


        # set of points for which we generate constraints 
        cg_pts = set() 

        # Obtain the initial set of points for which we generate constraints before the first iteration of the MILP 
        kmeans_centers, assignCluster, binaryAssignVar,initConstrsEdge, initConstrsNear = initConstraints(data, K, addConstrs = self.initConstrCnt, randState = self.randState, ratio = self.initConstrIntRatio)

        # Initial set of constraints 
        initContrs =  initConstrsEdge.union(initConstrsNear)

        cg_pts = set(initContrs)
        n_cg_pts =  len(cg_pts)

        # ensuring initial number of points for which we generate constraints is larger than the number of outliers 
        if n_cg_pts < self.outliersCnt:
            print("Add more than %0.0f intial points" %(self.outliersCnt))

        features = range(f)
        clusters = range(K) 


        # Defining the initial model in Gurobi for the first iteration in the constraint generation optimization methodology 
        m = gp.Model()

        # Error variable / objective value 
        error = m.addVar(vtype=gp.GRB.CONTINUOUS,lb = 0, name="E")

        # Centers of the MinMax clustering
        z = m.addVars(clusters, features, vtype=gp.GRB.CONTINUOUS, lb = np.min(X), ub = np.max(X), name="z_kj")
        
        
        # the below three variables need to added on the go during the iterative constraint generations 

        # variable to capture the absolute value of the difference of center and points along all dimensions to eventually give the L1 loss in error variable
        e = m.addVars(cg_pts, clusters, features, vtype=gp.GRB.CONTINUOUS, lb = 0, name="e_ikj")

        if self.use_errorBd:
            absId = m.addVars(cg_pts, clusters, features, vtype=gp.GRB.BINARY, name="a_ikj")

        # Binary indicator variables  
        c = m.addVars(cg_pts, clusters, vtype=gp.GRB.BINARY, name="c_ik")



        
        m.modelSense = gp.GRB.MINIMIZE

        # Adding constraints to the initial Gurobi model

        # Constraints to ensure one point is assigned to only one cluster

        if self.outliersCnt > 0:

            # modified constraints to exclude exactly outliersCnt (l) number of points from being assigned to any cluster

            m.addConstrs((gp.quicksum(c[pt,k] for k in clusters) <= 1 for pt in cg_pts ),"pt_in_cluster")
            totpts = m.addConstr(gp.quicksum(c[pt,k] for k in clusters for pt in cg_pts ) ==  n_cg_pts - self.outliersCnt )
        else:
            m.addConstrs((gp.quicksum(c[pt,k] for k in clusters) == 1 for pt in cg_pts ),"pt_in_cluster")


        # Symmetry breaking constraints

        for k in range(K-1):
            m.addConstr( z[k,0] + eps <= z[k+1,0])

        m.addConstrs( ( e[pt,k,j] >= X[pt,j] - z[k,j] for j in features for k in clusters for pt in initContrs) , "Residuals")
        m.addConstrs( ( e[pt,k,j] >= -(X[pt,j] - z[k,j]) for j in features for k in clusters for pt in initContrs) , "Residuals")
        
        if self.use_errorBd:
            # print("adding errbd const")
        
            m.addConstrs( ( e[pt,k,j] <= -(X[pt,j] - z[k,j]) + self.bigM*(1 - absId[pt,k,j])   for j in features for k in clusters for pt in initContrs) , "Residuals")
            m.addConstrs( ( e[pt,k,j] <= (X[pt,j] - z[k,j]) + self.bigM*absId[pt,k,j]   for j in features for k in clusters for pt in initContrs) , "Residuals")

        if self.use_bigM:
            m.addConstrs( error >= gp.quicksum(e[pt,k,j] for j in features ) - self.bigM*(1 - c[pt,k]) for k in clusters for pt in initContrs)   
        else: 
            m.addConstrs( ( (c[pt,k] == 1) >> (error >= gp.quicksum(e[pt,k,j] for j in features)) for k in clusters for pt in initContrs), "abs_obj_fn1")


        # Warm start 
        if self.WarmStart:
            # print('Warm Starting')
            for k in clusters:
                for pt in cg_pts:
                    c[pt,k].start = binaryAssignVar[pt,k]
                for j in features:
                    z[k,j].start = kmeans_centers[k,j]

        m.setObjective(error)

        m.setParam('TimeLimit', self.time*60) 

        m.setParam('MIPGap', self.optimalGap) 
        m.setParam('OutputFlag', self.outputFlag)

        # Constraint generation iterations 

        while counter<self.max_iter:

            # MILP solve with Gurobi
            m.optimize()

            # Data extraction from Gurobi

            # print('\nCOST: %g' % (m.objVal))

            eik = np.zeros((n,K))
            for k in clusters:
                for pt in cg_pts:
                    eij = 0
                    for j in features:
                        eij+=e[pt,k,j].X
                        
                    eik[pt,k] = eij
            center =np.zeros((K,f))

            for k in clusters:
                for j in features:
                    center[k,j] = z[k,j].X

            vals_error = error.X
            center_list.append(center)

            # print('Centers\n', center)

            outliers = []
            cik = np.zeros((n,K))
            for pt in cg_pts:
                for k in clusters:

                    cik[pt,k] = c[pt,k].X

                
                if sum(cik[pt,:]) == 0:
                    outliers.append(pt)

            # Gurobi Maximum distance points 

            distMSys,_ = getDistAssignMat(X,center,Cik=cik)
            
            maxPts,_,_,_= getConstraintPts(distMSys,K)

            maxPts_list.append(maxPts)
            
            # print('Gurobi Outliers: ', outliers)
            outliers_list.append(outliers)

            data['cluster'] = getClusterAssign(cik, outliers)

            print('E: ',vals_error)

            # Explicitly check whether the maximum distance from all points to their closest center is same as that from Gurobi
            # If not, identify the most violated constraints and add them to the constraint set and re-solve the MILP

            # Get Distance matrix

            distM , trueCik =   getDistAssignMat(X,center) 


            # get outliers and points to add as constraints 

            ptConstrs, trueOutIndx, maxError, distMRes = getConstraintPts(distM,K,self.outliersCnt)
            
            # print('MaxError: ',maxError)

            trueOutliers_list.append(trueOutIndx)

            trueMaxPts_list.append(ptConstrs)

            # print('Adding max pts (with or without outliers) ',ptConstrs)

            # print('Adding outliers if any ', trueOutIndx)

            allAddPts = list(np.ravel(ptConstrs))
            allAddPts.extend(list(np.ravel(trueOutIndx)))
            

            # capture true labels for all points by assigning points to their closest center

            data['trueCluster'] = labels = getClusterAssign(trueCik, trueOutIndx)

            df_data_list.append(data.copy())

            # Exit here to increase maximum required iteration 
            if counter == self.max_iter-1:
                
                if any(maxError > vals_error + self.tol):

                    print('Optimal solution not reached but current cost: ', m.objVal)
                    print('Increase # of iterations')
                else:
                    print('Optimal solution reached with Cost: ', m.objVal)

                
                optval = m.objVal
                optgap = m.MIPGap
                # addPts_list.append(list(ptConstrs))

                if self.outliersCnt > 0:
                    data['trueCluster'] = labels = getClusterAssign(trueCik)


                inertia_ = getSSE(X, center, labels, trueOutIndx)

                epsVal = getOptimalValue(X, center, self.outliersCnt)


                break

            # check if we need to add any more constraints 
            if any(maxError > vals_error + self.tol):

                addptConstrs = [pt for pt in allAddPts if pt not in cg_pts]

                if not addptConstrs:
                    addptConstrs = getMoreConstraintPts(distMRes, K, cg_pts,n,trueCik)
                
                addPts_list.append(addptConstrs)

                # print("Generate constraints after repetitions are removed:" , addptConstrs)

                for pt in addptConstrs:
                    if pt not in cg_pts:
                        # print('Adding constr for: ', pt)

                        # add variables for the new points


                        for k in clusters:
                            c.update({(pt, k ): m.addVar(vtype=gp.GRB.BINARY, name= 'c_ik['+str(pt)+','+str(k)+']') })

                            for j in features:
                                e.update({(pt, k ,j): m.addVar(vtype=gp.GRB.CONTINUOUS, lb = 0, name= 'e_ikj['+str(pt)+','+str(k)+','+str(j)+']') })

                                if self.use_errorBd:
                                    # print("adding errbd variables")
                                    absId.update({(pt, k ,j): m.addVar(vtype=gp.GRB.BINARY, name= 'a_ikj['+str(pt)+','+str(k)+','+str(j)+']') })
    
                        # print('added vars for ', pt)

                        # add constraints for the new points

                        if self.outliersCnt > 0:
                            m.addConstr((gp.quicksum(c[pt,k] for k in clusters) <= 1  ))
                        else:
                            m.addConstr((gp.quicksum(c[pt,k] for k in clusters) == 1 ))


                        m.addConstrs( ( e[pt,k,j] >= X[pt,j] - z[k,j] for j in features for k in clusters ) )
                        m.addConstrs( ( e[pt,k,j] >= -(X[pt,j] - z[k,j]) for j in features for k in clusters ) )

                        if self.use_errorBd:
                            # print("adding errbd const")

                            m.addConstrs( ( e[pt,k,j] <= -(X[pt,j] - z[k,j]) + self.bigM*(1 - absId[pt,k,j])   for j in features for k in clusters ) )
                            m.addConstrs( ( e[pt,k,j] <= (X[pt,j] - z[k,j]) + self.bigM*absId[pt,k,j]   for j in features for k in clusters ) )

                        if self.use_bigM:
                            m.addConstrs( error >= gp.quicksum(e[pt,k,j] for j in features ) - self.bigM*(1 - c[pt,k]) for k in clusters)   
                        else: 
                            m.addConstrs( ( (c[pt,k] == 1) >> (error >= gp.quicksum(e[pt,k,j] for j in features)) for k in clusters ))

                    
                    cg_pts.add(pt)

                if self.outliersCnt > 0 :
                        
                    # delete the old constraint

                    m.remove(totpts)

                    # add the new constraint under the same name
                    n_cg_pts = len(cg_pts)
                    # print("adding totpts const")
            
                    totpts = m.addConstr(gp.quicksum(c[pt,k] for k in clusters for pt in cg_pts ) ==  n_cg_pts - self.outliersCnt )

                m.update()
                    
            else:
                print('Optimal solution reached with Cost: ', m.objVal)
                optval = m.objVal
                optgap = m.MIPGap
                addPts_list.append(list(ptConstrs))

                if self.outliersCnt > 0:
                    data['trueCluster'] = labels = getClusterAssign(trueCik)

                inertia_ = getSSE(X, center, labels, trueOutIndx)
                epsVal = getOptimalValue(X, center, self.outliersCnt)

                break

            counter+=1


        print('\n\n# of constraints added: ',len(cg_pts))

        return data, labels, center, epsVal, trueCik, cg_pts, inertia_, optval, optgap, df_data_list, center_list, initContrs, trueMaxPts_list, addPts_list, trueOutIndx





class K_Means(LossOptimization):


    def __init__(self, 
                 n_init = 1
                 ):
        """
        A class that performs KMeans clustering

        Parameters
        ----------

        n_init:    (int)        Default: 1 
                                Number of time the k-means algorithm will be run with different
                                centroid seeds in scikit-learn. Default is at one since we explicitly 
                                run the K-means algorithm for required number of independent runs to get final output.
        
        """

        self.n_init = n_init





    def optimize(self, data, K, max_iter, random_state,*args, **kwargs):
 
        """
        A function that calls KMeans clustering from scikit-learn
        
        """
        data = data.copy()
        X = data.to_numpy()
        kmeans = KMeans(n_clusters=K, init = 'random',  max_iter = max_iter, random_state=random_state, n_init=self.n_init).fit(data)
        best_centers = kmeans.cluster_centers_
        dictCluster = { j:i for i,j in enumerate(best_centers[:, 0].argsort()) }

        best_centers = best_centers[best_centers[:, 0].argsort()]

        assignCluster = [dictCluster.get(x) for x in kmeans.labels_  ]
        data['trueCluster'] = assignCluster
        
        epsVal = getOptimalValue(X, best_centers)

            
        return data, assignCluster , best_centers,epsVal, 0,0,kmeans.inertia_,0,0,0,0,0,0,0,0




class K_MeansPlus(LossOptimization):


    def __init__(self, 
                 n_init = 1
                 ):
        """
        A class that performs KMeans++ clustering

        Parameters
        ----------

        n_init:    (int)        Default: 1 
                                Number of time the k-means++ algorithm will be run with different
                                centroid seeds in scikit-learn. 
        
        """

        self.n_init = n_init





    def optimize(self, data, K, max_iter, random_state,*args, **kwargs):
 
        """
        A function that calls KMeans++ clustering from scikit-learn
        
        """
        data = data.copy()
        X = data.to_numpy()
        kmeans = KMeans(n_clusters=K, init = 'k-means++',  max_iter = max_iter, random_state=random_state, n_init=self.n_init).fit(data)
        best_centers = kmeans.cluster_centers_
        # print(best_centers)
        dictCluster = { j:i for i,j in enumerate(best_centers[:, 0].argsort()) }

        best_centers = best_centers[best_centers[:, 0].argsort()]
        assignCluster = [dictCluster.get(x) for x in kmeans.labels_  ]
        data['trueCluster'] = assignCluster

        epsVal = getOptimalValue(X, best_centers)
            
        return data, assignCluster , best_centers,epsVal, 0,0,kmeans.inertia_,0,0,0,0,0,0,0,0




class KCenters_Gon(LossOptimization):
    def __init__(self, 
                 distance_metric = "manhattan"
                 ):
        """
             ss
        """

        self.distance_metric = distance_metric 


    def optimize(self, data, K, random_state, *args, **kwargs):
        # what to output:
            # K centroids 
            # labels for each data point, c(i)

        # print(random_state)
        data = data.copy()
        inertia = 0

        D = data.shape[1] # dimension of each pt
        N = data.shape[0] # num of pts 

        data = data.copy()
        X = np.array(data)

        #### STEP 1 - randomly inititizalize the 1st center to one of the data points 
        rng = np.random.default_rng(seed=random_state) # add a seed
        centers = rng.choice(X, 1, replace=False)
        #print(centers)

        z = np.zeros((N,1))

        # if self.visualize_steps:
        #     plots(centers, data, z)

        i = 0
        while centers.shape[0] < K:
            # Running while loop until K centers have been chosen

            #### STEP 2 -
            # choose the next center to be the data point that has the maximum distance to an already 
            # chosen cluster centroid closest to it

            p_dists = pairwise_distances(centers, X, metric=self.distance_metric)
            # column number is the data point index
            # row number is the center's index (out of current set of centers)


            #closest_pair_dist_ind = np.argmin(p_dists, axis=0)
            closest_pair_distances = np.min(p_dists, axis=0) # preserve the closest distances bt data pts and (their resp closest) centers only 
            max_closest_pair_dist_ind = np.argmax(closest_pair_distances) # index of data point (i) w max distance to its closest center
            new_center = X[max_closest_pair_dist_ind, :]

            #### STEP 3 - add new center to list of centers 
            centers = np.vstack((centers, new_center))


  
            i=i+1

        new_p_dists = pairwise_distances(centers, X, metric=self.distance_metric)
        assignments = np.argmin(new_p_dists, axis=0)
        z = assignments 

        
        ## best centers for the clusters
        best_centers = centers
        z = z.flatten()
        z = z.astype(int)


        # inertia is the sse:
        inertia = getSSE(X, best_centers, (np.array(z)+1), trueOutIndx = [])


        # print(best_centers)
        dictCluster = { j:i for i,j in enumerate(best_centers[:, 0].argsort()) }
        best_centers = best_centers[best_centers[:, 0].argsort()] 
        assignCluster = [dictCluster.get(x) for x in z ] 


        data['trueCluster'] = assignCluster # pd dataframe


        epsVal = getOptimalValue(X, best_centers)
        return data, assignCluster , best_centers,epsVal, 0,0, inertia , 0 ,0,0,0,0,0,0,0



class KCenters_GonPlus(LossOptimization):

    def __init__(self, 
                 distance_metric = "manhattan"
                 ):
        """
           k  
        """

        self.distance_metric = distance_metric 


    def optimize(self, data, K, indp_runs, *args, **kwargs):
        data = data.copy()
        
        print(indp_runs)
        epsVal_list = []
        best_centers_list = []
        randomSeed_list = []
        for i in range(indp_runs):
            # print(i)
            randseed = np.random.randint(0, 2**16-1)
            randomSeed_list.append(randseed)
            dataTmp, assignCluster , best_centers,epsVal, _,_, inertia , _ ,_,_,_,_,_,_,_ = KCenters_Gon.optimize(self,data,K, randseed)
            # print(epsVal)
            epsVal_list.append(epsVal)
            # print(best_centers)
            best_centers_list.append(best_centers)

            # dataTmp, assignCluster , best_centers,epsVal, _,_, inertia , _ ,_,_,_,_,_,_,_ = data, 0 , 0,0, 0,0, 0 , 0 ,0,0,0,0,0,0,0
        bestEpsVal = np.argmin(epsVal_list)

        # print(np.argmin(epsVal_list))
        # print(best_centers_list[bestEpsVal][0])
        # print(randomSeed_list[bestEpsVal])

        
        return KCenters_Gon.optimize(self,data,K, randomSeed_list[bestEpsVal])


class KCenter_HS(LossOptimization):

    # def __init__(self, k =):
        
    def optimize(self, data, K, *args, **kwargs):
        
        data = data.copy()

        X = np.array(data)
        max_shape = min(30000, X.shape[0])
        X_subset = X[np.random.permutation(X.shape[0])[:max_shape]]


        X_subset.shape

        dist_matrix = pairwise_distances(X_subset, metric='manhattan')
        dist_matrix /= dist_matrix.max()


        hs1 = ClusteringProblem(k=K,dist_matrix=dist_matrix)
        hs1.find_R_f()
        best_centers = X_subset[hs1.hs_S]
        best_centers = best_centers[best_centers[:, 0].argsort()] 
        _, cik = getDistAssignMat(X, best_centers) 
        labels = getClusterAssign(cik)

        data['trueCluster'] = labels 

        # inertia is the sse:
        inertia = getSSE(X, best_centers, labels , trueOutIndx = [])

        # print(best_centers)

        epsVal = getOptimalValue(X, best_centers)

        return data, labels , best_centers,epsVal, 0,0, inertia , 0 ,0,0,0,0,0,0,0




class KCenters_HSPlus(LossOptimization):


    def optimize(self, data, K, indp_runs, *args, **kwargs):
        data = data.copy()
        
        print(indp_runs)
        data_list = []
        labels_list = []
        inertia_list = []
        epsVal_list = []
        best_centers_list = []
        randomSeed_list = []


        for i in range(indp_runs):
            # print(i)
            randseed = np.random.randint(0, 2**16-1)
            randomSeed_list.append(randseed)
            dataTmp, labels , best_centers,epsVal, _,_, inertia , _ ,_,_,_,_,_,_,_ = KCenter_HS.optimize(self,data,K)
            # print(epsVal)
            epsVal_list.append(epsVal)
            data_list.append(dataTmp)
            labels_list.append(labels)
            inertia_list.append(inertia)
            best_centers_list.append(best_centers)

        bestEpsVal = np.argmin(epsVal_list)

        # print(np.argmin(epsVal_list))
        # print(best_centers_list[bestEpsVal][0])
        # print(randomSeed_list[bestEpsVal])

        
        return data_list[bestEpsVal], labels_list[bestEpsVal] , best_centers_list[bestEpsVal],epsVal_list[bestEpsVal], 0,0, inertia_list[bestEpsVal] , 0 ,0,0,0,0,0,0,0


class KMedian(LossOptimization):
    # implement using pyclustering library 


    def optimize(self, data, K, max_iter, *args, **kwargs):
        X = np.array(data)
        data = data.copy()
        
        data1 = ((np.array(data))).tolist()
        
        # random initialization
        initial_medians = random_center_initializer(data1, K).initialize()

        #rng = np.random.default_rng(seed=random_state) 
        #initial_medians = rng.choice(data, K, replace=False)
        #initial_medians = initial_medians.tolist()


        # Create instance of K-Medians algorithm.
        kmedians_instance = kmedians(data1, initial_medians, metric = distance_metric(type_metric.MANHATTAN), itermax = max_iter)
        # Run cluster analysis and obtain results.
        kmedians_instance.process()

        clusters = kmedians_instance.get_clusters()

        medians = kmedians_instance.get_medians()

        # get labels for each data point (based on closest center/median) 
        closest_clusters = kmedians_instance.predict(data1)


        best_centers = np.array(medians)
        dictCluster = { j:i for i,j in enumerate(best_centers[:, 0].argsort()) }
        best_centers = best_centers[best_centers[:, 0].argsort()]

        assignCluster = [dictCluster.get(x) for x in closest_clusters ]
        data['trueCluster'] = assignCluster


        # add in code to compute the epsilon (cost/objective) value ###############
        epsVal = getOptimalValue(X, best_centers)

        return data, assignCluster , best_centers,epsVal,0,0,  0   ,0,0,0,0,0,0,0,0





class KMedian_Plus(LossOptimization):



    def optimize(self, data, K, max_iter,  *args, **kwargs):
        # K-medians algorithm (using the pyclustering library) with k-means++ initializatoin
        # first get a list of initial centers from self.data??? using the km++ initialization
        # then pass that initial set of centers into the pyclustering function for kmedian
        data = data.copy()

        data1 = ((np.array(data))).tolist()
        X = np.array(data)


        # Calculate initial centers using K-Means++ method. 
        N = np.array(data).shape[0]
        n_candidates = N

        # assume that all N data points can be chosen as centers (n_candidates parameter) 
        # init_medians = kmeans_plusplus_initializer(data1, K, n_candidates).initialize()
        init_kmeansplus, _ =  kmeans_plusplus(X,K,n_local_trials=1)
        init_medians = []
        for i in range(K):
            init_medians.append(init_kmeansplus[i])

        # call the pycluster function for kmedians 

        # Create instance of K-Medians algorithm.
        kmedians_instance = kmedians(data1, init_medians,metric = distance_metric(type_metric.MANHATTAN), itermax = max_iter)
        # Run cluster analysis and obtain results.
        kmedians_instance.process()
    
        clusters = kmedians_instance.get_clusters()

        medians = kmedians_instance.get_medians()

        # get cluster assignments (for each datum)
        closest_clusters = kmedians_instance.predict(data1)


        
        best_centers = medians
        best_centers = np.array(best_centers)

        dictCluster = { j:i for i,j in enumerate(best_centers[:, 0].argsort()) }

        best_centers = best_centers[best_centers[:, 0].argsort()]

        assignCluster = [dictCluster.get(x) for x in closest_clusters]
        data['trueCluster'] = assignCluster


        epsVal = getOptimalValue(X, best_centers)
        
        
        # add in code to compute the epsilon (cost/objective) value ###############

            
        return data, assignCluster , best_centers,epsVal,0,0, 0 ,0,0,0,0,0,0,0,0

    

class KZcenters_charikar(LossOptimization): 
    '''
    ## 
    '''       
    def __init__(self, 
                n_init = 1,
                distance_metric = "manhattan",
                l = 0,
                n_outliers=0,
                ):
        """
        A SGD optimizer (optionally with nesterov/momentum).
        :param C:                   
        :param kernel:              
        """
        self.n_init = n_init
        self.distance_metric = distance_metric ##
        self.n_outliers = n_outliers

        

    def optimize(self, data, K, 
                    sample_weight=None, guessed_opt=None,
                    n_outliers=0, delta=0.05,
                    dist_oracle=None, return_opt=True ,
                    densest_ball_radius=1, removed_ball_radius=3, *args, **kwargs): 
        """         
        
        Implementation of the algorithm proposed in Moses Charikar's SODA'01 paper:
        Moses Charikar, Samir Khuller, David M. Mount, and Giri Narasimhan.
        Algorithms for facility location problems with outliers. SODA'2001
        
        :param X: array of shape=(n_samples, n_features) -- data points 
        :param sample_weight: array of shape=(n_samples,) -- weight associated with each data pt


        :param guessed_opt: float, a guess for the optimal radius ##### initial guess for the value of G (optional variable though)
        
        
        :param n_clusters = K: int, number of cluster centers

        :param n_outliers: int, number of desired outliers.

        :param delta: float, the ratio of the geometric sequence that used to guess the opt.
        
        :param dist_oracle: An DistQueryOracle object

        :param return_opt: bool, if True then return the resulted clusters along with the radius
        
        :param densest_ball_radius: int, default 2, or 1 
            find the densest ball of radius densest_ball_radius * OPT

        :param removed_ball_radius: int, default 4, or 3
            remove the ball of radius removed_ball_radius * OPT
        
        :return results: list of (array, int)
                List of (ball center, #total weights in the ball)
        """
        
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        data = data.copy()
        
        X = np.array(data)

        if dist_oracle is None:
            dist_oracle = DistQueryOracle(tree_algorithm='auto', dist_metric=self.distance_metric) # create instance of class
        if not dist_oracle.is_fitted:
            dist_oracle.fit(X, sample_weight) # fitting is based on the sample weights 
            # the weights are all 1 so its a uniform distribution
        
        n_distinct_points= X.shape[0]
        

        if sample_weight is None:
            #sample_weight = np.ones(n_distinct_points) # equal weighting for all points anyway
            sample_weight = np.ones(X.shape[0]) # set weights for all pts to be 1 (if weights are not passed on)


        #print(sample_weight)
        n_samples = sum(sample_weight) # number of data points in total (since weights are all 1s)
        
    
        if n_distinct_points <= K: # handle edge case
            warnings.warn("Number of total distinct data points is smaller than required number of clusters.")
            return [(c, w) for c, w in zip(X, sample_weight)]

        #####################################################

        # estimate the upperbound and lowerbound of opt (r) for the data set (internal criterion)
        
        # minimum of the furthest pair distances between randomly chosen points 
        _, ub = dist_oracle.estimate_diameter(n_estimation=10)

        lb = np.inf
        for _ in range(10): 
            # 1 point is chosen randomly (set as the center) to find kneighbor data pts (in X) to that center
            
            # maximum of the closest pair distances between randomly chosen 'center' and k neighbors 
            lb_tmp, _ = dist_oracle.kneighbors(X[np.random.choice(n_distinct_points, 1, p=sample_weight / n_samples)].reshape(1, -1), k=2)
            lb_tmp = np.max(lb_tmp) # furthest of the 2 neighbors is chosen

            lb = min(lb, lb_tmp)

        if guessed_opt is not None: # if a guess 'guessed_opt' was provided 
            guessed_opt = min(guessed_opt, ub)
        
        if guessed_opt is None:
            guessed_opt = (lb + ub) / 2 # value of r (guess of OPT) from avg of lower and upper bounds 


        #####################################################

        results = []
        facility_idxs = np.arange(n_distinct_points) # indices of the n data points 
        n_facilities = len(facility_idxs)
        n_facilities_thresh = 1e4 


        while ub > (1 + delta) * lb: # keep on re-running until ub and lb(1+delta) converge to each other (crossing ub and lb)
            # in each iteration, BINARY SEARCH runs until ub and lb converge and we find 
            # the estimate of opt 

            removed = set()
            results = []

            to_be_updated = facility_idxs # the entire set of points (initialially) are to be explored 

            for i in range(K): 
                if len(removed) == n_distinct_points:
                    break # if radius is too large, all pts become within 3r of the first center
                            # hence we break out of for loop 


                # When the number of available facilities is huge, use the dense_ball_ method that
                # has caching and early-returning
                if n_facilities > n_facilities_thresh:
                    p, covered_pts = dist_oracle.dense_ball_(densest_ball_radius * guessed_opt, # corresponds to G in alg 3
                                                            changed=to_be_updated,
                                                            facility_idxs=facility_idxs,
                                                            except_for=removed, # dont look at points that are covered (within 3G of prev selected center)
                                                            minimum_density=np.inf)
                else:
                    p, covered_pts = dist_oracle.densest_ball(densest_ball_radius * guessed_opt, # corresponds to G in alg 3
                                                            removed)


                # pts in ball of radius (corresponding to 3r) around newly chosen center p 
                to_be_removed = dist_oracle.ball(p.reshape(1, -1), removed_ball_radius * guessed_opt)[0]
                # does this return the list of indices of pts within 3G of the newly chosen center? i think so 

                to_be_removed = set(to_be_removed).difference(removed) # only care about new points that have just been covered
                to_be_removed = np.array(list(to_be_removed))
                removed.update(to_be_removed)

            
                w_p = sum(sample_weight[to_be_removed]) # number of points just covered (within 3G) by new p (to be removed for next iteration)
                
                results.append((p, w_p)) 
                # p is the new center
                # w_p is sum of weights of pts within 3r of p (the new center) // no. of pts 
                

                ############################

                # after removing ball(p, aL), only the densest ball whose centers resides 
                # in ball(p, (a+b)L) \ ball(p, aL) is affected.  
                # -- we only need to consider the points that lie in the disc r in the set (a, b)
                if n_facilities > n_facilities_thresh:
                    to_be_updated = set(dist_oracle.ball(p.reshape(1, -1),
                                                        (densest_ball_radius + removed_ball_radius) * guessed_opt)[0]). \
                                                        intersection(facility_idxs)
                    
                    to_be_updated.difference_update(removed)  # sets 



            n_covered = sum(wp for _,wp in results) 
            

            # in each iteration n_covered grows until the final set of outliers remain at the end, at most z
 

            #############################################################

            # Binary search part of the heuristic to find r, 'guessed_opt'
            if n_covered >= n_samples - self.n_outliers: # n_outliers (expected no of outliers) is by default = 0
                # have actually covered more points than expected 
                # so make the G estimates smaller (1st half of lb-up interval)
                ub = guessed_opt
                guessed_opt = (lb + guessed_opt) / 2

            else:
                # havent actually covered enough points (not as much as expected)
                # so enlarge magnitudes of G estimate (2nd half of the lb-ub interval)
                lb = guessed_opt
                guessed_opt = (guessed_opt + ub) / 2



        ############################################################################

        # if the program finishes before finding k'<k centers, we use the FarthestNeighbor
        # method to produce the remained k-k' centers 
    
        if len(results) < K: # if we didn't get all K centers yet 
            centers = [c for c, _ in results]
            
            # find the distance from each data pt in X to its closest center (and find min dists -- closest centers)
            _, dists_to_centers = pairwise_distances_argmin_min(X, np.atleast_2d(centers), self.distance_metric)
            

            for i in range(0, K - len(results)): #loop 'centers remaining' number of times 
                next_idx = np.argmax(dists_to_centers, dtype=object,) # next center is the data pt furthest from its closest center 
                centers.append(X[next_idx])
                # TODO: here the new center's weight is set to its own weight, this might be problematic(?)

                results.append((X[next_idx], sample_weight[next_idx]))
                _, next_dist = pairwise_distances_argmin_min(X, np.atleast_2d(centers[-1]), self.distance_metric)
                dists_to_centers = np.minimum(dists_to_centers, next_dist)



        ######## Assignment step for each point 
        # Assign every point (including outliers) to its closest center 
    
        # set of centers 
        centers = np.copy(results)
        centers = (np.array(centers.tolist(), dtype=object))[:,0]
        centers = np.array([np.array(centers[i]) for i in range((centers.shape)[0])], dtype=object)

        p_dists = pairwise_distances(centers, X, metric=self.distance_metric)
        z = np.argmin(p_dists, axis=0)



        best_centers = centers
        
        dictCluster = { j:i for i,j in enumerate(best_centers[:, 0].argsort()) }
        best_centers = best_centers[best_centers[:, 0].argsort()] 
        
        assignCluster = [dictCluster.get(x) for x in z ] 
        
        labels = z


        # inertia is the sse:
        inertia = getSSE(X, best_centers, labels , trueOutIndx = [])
        
        # epsVal = getOptimalValue(X, best_centers)

        epsVal = getOptimalValue(X, best_centers, self.n_outliers)



        return X, labels , best_centers, epsVal, 0,0, inertia , 0 ,0,0,0,0,0,0,0
        #return results, guessed_opt, z, best_centers


