
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import *
from scipy.spatial import distance_matrix
from typing import *
from Codes.utils import *

verbose = False

class ClusteringProblem:
    def __init__(
        self,
        k: int,
        dist_matrix: np.array,
    ):
        """
        Returns an object to which stochastic constraints can be added to represent the clustering problem we care about.
        Args:
            k (int): The number of clusters.
            dist_matrix (np.array): Distance matrix corresponding to distances between each of the points in original_points (this doesn't have to be the Euclidean metric and will be used for k-center objective calculation).
        """
        self._k = k
        self._dist_matrix = dist_matrix
        self._similarity_sets = defaultdict(set)
        self._R_f = None
        self._R_f_index = None
        self._hs_phi = None
        self._sorted_distances = np.unique(self.dist_matrix)
        self._R_j = None
        self._R_j_array = None
        self._hs_S = None

    def construct_similarity_sets(self):
        assert self._R_f is not None, "R_f not found yet"
        if self._R_j_array is None:
            R_j_array = np.random.uniform(0, 2*self.R_f, self.num_points)
            self._R_j_array = R_j_array
        else:
            R_j_array = self._R_j_array
        for point in range(self.num_points):
            indices = set((self.dist_matrix[point] <= R_j_array[point]).nonzero()[0])
            self._similarity_sets[point].update(indices)
        max_R_j = np.max(R_j_array)
        index = np.searchsorted(self._sorted_distances, max_R_j)
        if index < len(self._sorted_distances) - 1:
            index += 1
        self._R_f_index = index

    def find_R_f(self):
        self._R_f, self._R_f_index, self._hs_phi,self._hs_S = find_R_f(self, hs=True)

    def set_R_f(self, R_f, R_f_index, hs_phi):
        self._R_f, self._R_f_index, self._hs_phi,self._hs_S= R_f, R_f_index, hs_phi, hs_S

    @property
    def R_j_array(self):
        return self._R_j_array

    @R_j_array.setter
    def R_j_array(self, arr):
        self._R_j_array = arr

    @property
    def hs_phi(self):
        return self._hs_phi

    @property
    def num_points(self):
        return self._dist_matrix.shape[0]

    def permutation_of_points(self, specified_set=None, list_fmt=False):
        if not list_fmt:
            if specified_set is None:
                return np.random.permutation(self.num_points)
            else:
                return np.random.permutation(list(specified_set))
        else:
            if specified_set is None:
                return random.sample(range(self.num_points), self.num_points)
            else:
                return random.sample(list(specified_set), len(specified_set))

    @property
    def dist_matrix(self):
        return self._dist_matrix

    @property
    def k(self):
        return self._k

    @property
    def similarity_sets(self):
        return self._similarity_sets
    @property
    def hs_S(self):
        return self._hs_S
    @property
    def R_f(self):
        return self._R_f

    @property
    def R_f_index(self):
        return self._R_f_index

    @property
    def sorted_distances(self):
        return self._sorted_distances

def hochbaum_shmoys_filtering(clustering_problem: ClusteringProblem, R: float) -> bool:
    """Runs Hochbaum-Shmoys algorithm
    Args:
        clustering_problem (ClusteringProblem): the problem instance
        R (float): the radius guess to be used for Hochbaum-Shmoys
    Returns:
        bool: returns True if a feasible clustering can be found using the provided R
    """
    k = clustering_problem.k
    dist_matrix = clustering_problem.dist_matrix
    U = clustering_problem.permutation_of_points()  # pick in an arbitrary order
    set_U = set(U)
    S = set()  # selected centers
    max_radius = None
    while len(set_U):
        c = random.sample(set_U, 1)[0]
        set_U.remove(c)
        S.add(c)
        G_c = set()
        for j in set_U:
            if dist_matrix[j, c] <= 2 * R:
                if max_radius is None:
                    max_radius = dist_matrix[j, c]
                else:
                    max_radius = max(max_radius, dist_matrix[j, c])
                G_c.add(j)
        set_U -= G_c
    if len(S) > k:
        return False
    else:
        return True, S





def find_R_f(clustering_problem,  hs=True) -> float:
    sorted_distances = clustering_problem.sorted_distances
    if hs:
        func = hochbaum_shmoys_filtering


    l = 0 if hs else clustering_problem.R_f_index
    h = len(sorted_distances) - 1
    i = 0

    best_result = None

    while l < h:
        m = (l + h) // 2
        i += 1
        if verbose:
            print(f"Iteration {i} of binary search for main_problem: {not hs}")
            print(f"Current value of m is {m}")
        if hs:
            result = func(clustering_problem, sorted_distances[m])
            # print(result)


        if (not hs and result is not None) or (result and hs):
            h = m
            best_result = result
            # print(best_result)
        else:
            l = m + 1

    r_f = sorted_distances[l]

    if hs:
        hs_phi = dict()
        hs_S = list(best_result[-1])
        for point in range(clustering_problem.num_points):
            # hs_phi[point] = hs_S[clustering_problem.dist_matrix[point, hs_S].argmin()]
            hs_phi[point] = clustering_problem.dist_matrix[point, hs_S].argmin()


        return r_f, l, hs_phi, hs_S  # minimum radius that's feasible







##############################################################

# k,z-charikar_centers helper class 


class DistQueryOracle(object):

    def __init__(self, tree_algorithm='auto', leaf_size=60,
                 metric='minkowski', precompute_distances='auto',
                 random_state=None, debugging=False, dist_metric="euclidean"):
        """
        :param tree_algorithm: string in {'auto', 'kd_tree', 'ball_tree', 'brute', 'lsh'}
            determines the
        :param leaf_size: int, default 40
            leaf size passed to BallTree or KDTree
        :param metric: string, default 'minkowski'
            the distance metric to use in the tree
        :param precompute_distances: {'auto', True, False}
            Precompute distances (faster but takes more memory).
            'auto': do not precompute distances if n_samples^2  > 1.2 million.
            This corresponds to about 10MB overhead per job using double precision.
            True: always precompute distances
            False: never precompute distances
        :param random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.
        :param debugging: bool, default False, whether to print debugging and User warning info
        """
        self.tree_algorithm = tree_algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        # TODO: add pre-computed distance matrix
        self.precompute_distances = precompute_distances
        self.random_state_ = random_state
        self.debugging_ = debugging
        self.dist_metric = dist_metric


        self.nn_tree_ = None
        self.ball_oracle_ = None
        self.knn_oracle_ = None
        self.fitted_data_ = None
        self.data_weight_ = None
        self.diam_ = None
        self.n_samples_, self.n_features_ = None, None
        self.new_data = True

        # variables used for supporting fast densest-ball query and removing
        self.n_facilities_ = None
        self.facility_idxs_ = None
        self.sorted_distance_cache_ = None
        self.sorted_dist_idxs_ = None
        self.unweightted_ = None
        self.ball_weight_cache_ = None
        self.ball_size_cache_ = None
        self.ball_radius_cache_ = None
        self.ball_cache_ = None
        self.cache_inconsistent_ = None

    @property
    def is_fitted(self):
        return self.ball_oracle_ is not None

    def fit(self, X, sample_weight=None):
        """
        :param X: array of shape=(n_samples, n_features),
            Data set to be processed
        :param sample_weight: array of shape=(n_samples,),
            weight on each data instance. "None" means uniform weights.
        :return:
        """
        if self.tree_algorithm == 'auto':
            if X.shape[1] < 20:
                self.nn_tree_ = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
                self.ball_oracle_ = lambda cs, r: self.nn_tree_.query_radius(cs, r, return_distance=False)
                self.knn_oracle_ = lambda cs, k, rd: self.nn_tree_.query(cs, k, return_distance=rd)
            elif X.shape[0] < 40:
                self.sorted_distance_cache_, self.sorted_dist_idxs_ = self.precompute_distances_matrix_(X)
                self.ball_oracle_ = lambda cs, r: _brute_force_ball(X, cs, r, dist_metric = self.dist_metric)
                self.knn_oracle_ = lambda cs, k, rd: _brute_force_knn(X, cs, k, return_distance=rd, dist_metric=self.dist_metric)
                self.brute_force_ball_within_dataset_oracle_ = lambda cidx, r, co: \
                    _brute_force_ball_within_dataset(X, cidx, r,
                                                     sorted_distances=self.sorted_distance_cache_,
                                                     sorted_idxs=self.sorted_dist_idxs_, count_only=co, dist_metric=self.dist_metric)
            else:
                self.nn_tree_ = BallTree(X, leaf_size=self.leaf_size, metric=self.metric)
                self.ball_oracle_ = lambda cs, r: self.nn_tree_.query_radius(cs, r, return_distance=False)
                self.knn_oracle_ = lambda cs, k, rd: self.nn_tree_.query(cs, k, return_distance=rd)

        elif self.tree_algorithm == 'kd_tree':
            self.nn_tree_ = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
            self.ball_oracle_ = lambda cs, r: self.nn_tree_.query_radius(cs, r, return_distance=False)
            self.knn_oracle_ = lambda cs, k, rd: self.nn_tree_.query(cs, k, return_distance=rd)

        elif self.tree_algorithm == 'ball_tree':
            self.nn_tree_ = BallTree(X, leaf_size=self.leaf_size, metric=self.metric)
            self.ball_oracle_ = lambda cs, r: self.nn_tree_.query_radius(cs, r, return_distance=False)
            self.knn_oracle_ = lambda cs, k, rd: self.nn_tree_.query(cs, k, return_distance=rd)

        elif self.tree_algorithm == 'brute':
            self.sorted_distance_cache_, self.sorted_dist_idxs_ = self.precompute_distances_matrix_(X)
            self.ball_oracle_ = lambda cs, r: _brute_force_ball(X, cs, r, dist_metric=self.dist_metric)
            self.knn_oracle_ = lambda cs, k, rd: _brute_force_knn(X, cs, k, return_distance=rd, dist_metric=self.dist_metric)
            self.brute_force_ball_within_dataset_oracle_ = lambda cidx, r, co: \
                _brute_force_ball_within_dataset(X, cidx, r,
                                                 sorted_distances=self.sorted_distance_cache_,
                                                 sorted_idxs=self.sorted_dist_idxs_, count_only=co,
                                                 dist_metric = self.dist_metric)

        else:
            raise ValueError("tree_algorithm \"{}\" not properly specified".
                             format(self.tree_algorithm))

        self.fitted_data_ = X
        self.n_samples_, self.n_features_ = X.shape

        if sample_weight is not None:
            self.data_weight_ = sample_weight
        else:
            self.data_weight_ = np.ones(X.shape[0], dtype=np.int)
            self.unweightted_ = True

        return self

    def precompute_distances_matrix_(self, X):
        if self.precompute_distances is True or \
                (self.precompute_distances == 'auto' and 8 * (X.shape[0] ** 2) < 20e6):
            distance_cache = pairwise_distances(X)
            distance_argsort = np.argsort(distance_cache, axis=1)
            return distance_cache[np.arange(distance_cache.shape[0])[:, None], distance_argsort],\
                   distance_argsort
        else:
            return None, None

    def ball(self, centers, radius):
        """Query the data points in X that are within distance `radius` to `centers`
        :param centers: array of shape=(n_points, n_features), queried centers
        :param radius: float, radius of the ball
        :return: an array of array,
            indices for each center in centers
        """
        if self.ball_oracle_ is None:
            raise NotFittedError("Tree hasn't been fitted yet\n")

        centers = check_array(centers, ensure_2d=True)
        return self.ball_oracle_(centers, radius)



    def densest_ball(self, radius, except_for=None):
        """Find the ball of given radius that covers the most data points
        :param radius: float, radius of the ball
        :param except_for: iterable or set,
            indices of points that should not be considered
        :return (densest_center, densest_ball): (array of shape=(n_features,), array of shape=(n_covered,)
            the center of the densest ball as well as the index of points the ball covers
        """
        if except_for is None or len(except_for) == 0:
            except_for = None

        if self.ball_oracle_ is None:
            raise NotFittedError("Tree hasn't been fitted yet\n")

        if except_for is not None and len(except_for) == self.n_samples_:
            return None, None

        if except_for is None:
            densest_idx, _ = max(
                ((i, self.data_weight_[self.ball_oracle_(x.reshape(1, -1), radius)[0]].sum())
                 for i, x in enumerate(self.fitted_data_)),
                key=lambda a: a[1]
            )
            densest_center = self.fitted_data_[densest_idx]
            densest_ball = self.ball_oracle_(densest_center.reshape(1, -1), radius)[0]
        else:
            densest_idx, _ = max(
                ((i, self.data_weight_[list(set(self.ball_oracle_(x.reshape(1, -1), radius)[0]).
                                            difference(except_for))].sum())
                 for i, x in enumerate(self.fitted_data_) if i not in except_for),
                key=lambda a: a[1]
            )
            densest_center = self.fitted_data_[densest_idx]
            densest_ball = np.array(list(set(self.ball_oracle_(densest_center.reshape(1, -1), radius)[0]).
                                         difference(except_for)))
        # assert len(densest_ball) > 0
        return densest_center, densest_ball

    def init_all_densest_ball_faster_but_dirty(self, radius):
        self.ball_radius_cache_ = radius
        balls = self.ball_oracle_(self.fitted_data_, radius)

    def dense_ball_(self, radius, except_for, changed, facility_idxs=None, return_idx=False, minimum_density=None):
        """
        When the radius is fixed and need to do a series of query, then this function
        will cache previous calculated balls for fast retrieving.
        Warning: This function implementation is coupled with the one that invokes it. Shouldn't
        be called by other functions except for KZCenter.fit.
        :param radius:
        :param except_for: iterable or set,
            indices of points that should not be considered
        :param changed: iterable or set,
            indices in cache that need to be updated
        :param return_idx: bool,
            whether to return the index of the densest ball center within the data set
        :param minimum_density: float,
            minimum density requirement for early returning
        :return :
            (densest_center, densest_ball, center_idx):
                (array of shape=(n_features,), array of shape=(n_covered,), int)
            (densest_center, densest_ball): (array of shape=(n_features,), array of shape=(n_covered,)
                the center of the densest ball as well as the index of points the ball covers:
        """
        # They said that if I can run fast enough, sadness wouldn't catch me up.
        if facility_idxs is None:
            self.facility_idxs_ = np.arange(self.fitted_data_.shape[0])
        else:
            self.facility_idxs_ = facility_idxs

        if self.ball_radius_cache_ != radius:
            # new search begins, should refresh all caches
            self.ball_radius_cache_ = radius
            self.ball_weight_cache_ = [None] * self.n_samples_
            self.ball_size_cache_ = [None] * self.n_samples_
            self.ball_cache_ = [None] * self.n_samples_
            self.cache_inconsistent_ = False

        if self.cache_inconsistent_ and self.debugging_:
            warnings.warn("Cache is inconsistent, may get outdated result\n", UserWarning)

        if self.ball_weight_cache_ is None:
            self.ball_weight_cache_ = [None] * self.n_samples_

        if except_for is None:
            except_for = {}
        if changed is None:
            changed = range(self.n_samples_)
        if len(except_for) == self.n_samples_:
            return (None, None, None) if return_idx else (None, None)

        for i in changed:
            x = self.fitted_data_[i]
            ball_i = set(self.ball_oracle_(x.reshape(1, -1), radius)[0])

            # update ball cache
            if len(except_for) / len(ball_i) > 10:
                ball_i.difference_update(except_for.intersection(ball_i))
            else:
                ball_i.difference_update(except_for)

            self.ball_size_cache_[i] = len(ball_i)
            self.ball_weight_cache_[i] = self.data_weight_[list(ball_i)].sum()
            # if a ball covers all points, then it must be the densest one
            if self.ball_size_cache_[i] >= min(self.n_samples_ - len(except_for), minimum_density):
                self.cache_inconsistent_ = True
                ball_i = np.array(list(ball_i))
                return (self.fitted_data_[i], ball_i, i) if return_idx \
                    else (self.fitted_data_[i], ball_i)

        dense_idx = None
        dense_ball_weight = 0
        dense_ball = None

        ball_i = None
        remained = set(self.facility_idxs_).difference(except_for)
        if len(remained) == 0:
            return (None, None, None) if return_idx else (None, None)

        for i in remained:
            # because ball_cache can become inconsistent due to early returning
            if self.ball_size_cache_[i] is None:
                ball_i = set(self.ball_oracle_(self.fitted_data_[i].reshape(1, -1), radius)[0])
                ball_i.difference_update(except_for)
                self.ball_size_cache_[i] = len(ball_i)
                self.ball_weight_cache_[i] = self.data_weight_[list(ball_i)].sum()

            # if a ball covers all points, then it must be the densest one
            # this serves as an early return, but
            if self.ball_size_cache_[i] >= min(self.n_samples_ - len(except_for), minimum_density):
                self.cache_inconsistent_ = True
                if not ball_i:
                    ball_i = set(self.ball_oracle_(self.fitted_data_[i].reshape(1, -1), radius)[0])
                    ball_i.difference_update(except_for)
                ball_i = np.array(list(ball_i))
                return (self.fitted_data_[i], ball_i, i) if return_idx \
                    else (self.fitted_data_[i], ball_i)

            if dense_ball_weight < self.ball_weight_cache_[i]:
                dense_ball_weight = self.ball_weight_cache_[i]
                dense_idx = i
        dense_ball = set(self.ball_oracle_(self.fitted_data_[dense_idx].reshape(1, -1), radius)[0])
        dense_ball.difference_update(except_for)
        dense_ball = np.array(list(dense_ball))

        return (self.fitted_data_[dense_idx], dense_ball, dense_idx) if return_idx \
            else (self.fitted_data_[dense_idx], dense_ball)





    def densest_ball_faster_but_dirty(self, radius, except_for, changed, within_idx=None, return_idx=False):
        """
        When the radius is fixed and need to do a series of query, then this function
        will cache previous calculated balls for fast retrieving.
        Warning: This function implementation is coupled with the one that invokes it. Shouldn't
        be called by other functions except for KZCenter.fit.
        :param radius:
        :param except_for: iterable or set,
            indices of points that should not be considered
        :param changed: iterable or set,
            indices in cache that need to be updated
        :param return_idx: bool,
            whether to return the index of the densest ball center within the data set
        :return :
            (densest_center, densest_ball, center_idx):
                (array of shape=(n_features,), array of shape=(n_covered,), int)
            (densest_center, densest_ball): (array of shape=(n_features,), array of shape=(n_covered,)
                the center of the densest ball as well as the index of points the ball covers:
        """
        # TODO: what is the actual complexity of radius_query for BallTree and KDTree?
        # They said that if I can run fast enough, sadness wouldn't catch me up.

        if self.ball_radius_cache_ != radius:
            # new search begins, should refresh all caches
            self.ball_radius_cache_ = radius
            self.ball_weight_cache_ = [None] * self.n_samples_
            self.ball_size_cache_ = [None] * self.n_samples_
            # self.ball_cache_ = [None] * self.n_samples_
            self.ball_cache_ = np.ones(self.n_samples_, dtype=object) * -1
            self.cache_inconsistent_ = False

        if self.cache_inconsistent_:
            warnings.warn("Cache is inconsistent, may get outdated result\n", UserWarning)

        if self.ball_weight_cache_ is None:
            self.ball_weight_cache_ = [None] * self.n_samples_

        if except_for is None:
            except_for = {}
        if changed is None:
            changed = range(self.n_samples_)
        if len(except_for) == self.n_samples_:
            return (None, None, None) if return_idx else (None, None)

        # the intersection between changed and except_for should be empty
        # changed = np.array(list(changed))
        # x = np.atleast_2d(self.fitted_data_[changed])
        # not_cached_yet = np.where(self.ball_cache_[changed] == -1)[0]
        # if len(not_cached_yet) > 0:
        #     balls = self.ball_oracle_(x[not_cached_yet], radius)
        # for i in not_cached_yet:
        #     self.ball_cache_[changed[i]] = set(balls[i])

        approx = 1.0
        for i in changed:
            if i in except_for:
                continue

            x = self.fitted_data_[i]
            if self.ball_cache_[i] == -1:
                self.ball_cache_[i] = set(self.ball_oracle_(x.reshape(1, -1), radius)[0])

            # update ball cache
            if len(except_for) / len(self.ball_cache_[i]) > 10:
                self.ball_cache_[i].difference_update(except_for.intersection(self.ball_cache_[i]))
            else:
                self.ball_cache_[i].difference_update(except_for)

            self.ball_size_cache_[i] = len(self.ball_cache_[i])
            self.ball_weight_cache_[i] = self.data_weight_[list(self.ball_cache_[i])].sum()
            # if a ball covers all points, then it must be the densest one
            if self.ball_size_cache_[i] >= approx * (self.n_samples_ - len(except_for)):
                self.cache_inconsistent_ = True
                return (self.fitted_data_[i], self.ball_cache_[i], i) if return_idx \
                    else (self.fitted_data_[i], self.ball_cache_[i])

        densest_idx = None
        densest_ball_weight = 0

        for i in range(self.n_samples_):
            if i in except_for:
                continue

            # because ball_cache can become inconsistent due to early returning
            if self.ball_cache_[i] == -1:
                self.ball_cache_[i] = set(self.ball_oracle_(x.reshape(1, -1), radius)[0])
                self.ball_cache_[i].difference_update(except_for)
                self.ball_size_cache_[i] = len(self.ball_cache_[i])
                self.ball_weight_cache_[i] = self.data_weight_[list(self.ball_cache_[i])].sum()

            # if a ball covers all points, then it must be the densest one
            # this serves as an early return, but
            if self.ball_size_cache_[i] >= approx * (self.n_samples_ - len(except_for)):
                self.cache_inconsistent_ = True
                return (self.fitted_data_[i], self.ball_cache_[i], i) if return_idx \
                    else (self.fitted_data_[i], self.ball_cache_[i])

            if densest_ball_weight < self.ball_weight_cache_[i]:
                densest_ball_weight = self.ball_weight_cache_[i]
                densest_idx = i

        return (self.fitted_data_[densest_idx], self.ball_cache_[densest_idx], densest_idx) if return_idx \
            else (self.fitted_data_[densest_idx], self.ball_cache_[densest_idx])

    def estimate_diameter(self, n_estimation=1):
        """
        Pick an arbitrary point in the data set, suppose d is the largest distance between this
        point and any other points, then the diameter must be in [d, 2d]
        Time complexity: O(n_samples * n_estimations * n_features)
        :param n_estimation: number of sampled estimation points
        :return: (lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise NotFittedError("Tree hasn't been fitted yet\n")

        if self.diam_ is not None:
            return self.diam_

        self.diam_ = estimate_diameter(X=self.fitted_data_,
                                       n_estimation=n_estimation,
                                       metric=self.metric,
                                       dist_metric=self.dist_metric)

        return self.diam_

    def farthest_neighbor(self, x, return_distance=True):
        """Find the farthest point away from x
        Time complexity: O(n_samples * n_features)
        :param x:
        :param return_distance:
        :return:
        """
        if not self.is_fitted:
            raise NotFittedError("Tree hasn't been fitted yet\n")
        return farthest_neighbor(x, X=self.fitted_data_, return_distance=return_distance, dist_metric=self.dist_metric)

    def kneighbors(self, centers, k=1, return_distance=True):
        """return the k nearest neighbors to centers
        Time complexity: depends on the tree alg used
            brute - O(n_samples * n_features)
            ball_tree - O(n_features * \log n_samples)
            kd_tree - O(n_features * \log n_samples) for small n_features and O(n_samples * n_features)
                      for large n_features
            lsh - o(n_samples * n_features)
        :param centers: array of shape=(n_queries, n_features)
        :param k: queried number of nearest neighbors
        :param return_distance: default True, whether to return the distance of the neighbors to its nearest centers
        :return: idxs or (dists, idxs)
            idx - the indices of the nearest neighbor in the fitted data set
            dists - the corresponding distance of nearest neighbors
        """
        if not self.is_fitted:
            raise NotFittedError("Tree hasn't been fitted yet\n")

        centers = check_array(centers, ensure_2d=True)
        return self.knn_oracle_(centers, k, return_distance)
