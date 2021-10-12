"""
name:   testEM2
author:  Alaa Maalouf

researchers (last-name alphabetical-order):
    Dan Feldman
    Harry Lang
    Alaa Maalouf
    Daniela Rus
"""

import numpy as np
from scipy.linalg import null_space
import copy
import time
import random

# import getdim # file in directory to load BERT
LAMBDA = 1
Z = 2
# NUM_INIT_FOR_EM = 1
STEPS = 20
M_ESTIMATOR_FUNCS = {
    "lp": (lambda x: np.abs(x) ** Z / Z),
    "huber": (
        lambda x: x ** 2 / 2
        if np.abs(x) <= LAMBDA
        else LAMBDA * (np.abs(x) - LAMBDA / 2)
    ),
    "cauchy": (lambda x: LAMBDA ** 2 / 2 * np.log(1 + x ** 2 / LAMBDA ** 2)),
    "geman_McClure": (lambda x: x ** 2 / (2 * (1 + x ** 2))),
    "welsch": (
        lambda x: LAMBDA ** 2 / 2 * (1 - np.exp(-(x ** 2) / LAMBDA ** 2))
    ),
    "tukey": (
        lambda x: LAMBDA ** 2 / 6 * (1 - (1 - x ** 2 / LAMBDA ** 2) ** 3)
        if np.abs(x) <= LAMBDA
        else LAMBDA ** 2 / 6
    ),
}
global OBJECTIVE_LOSS
OBJECTIVE_LOSS = M_ESTIMATOR_FUNCS["lp"]


def computeDistanceToSubspace(point, X):
    """
    This function is responsible for computing the distance between a point and a J dimensional affine subspace.

    :param point: A numpy array representing a .
    :param X: A numpy matrix representing a basis for a J dimensional subspace.
    :param v: A numpy array representing the translation of the subspace from the origin.
    :return: The distance between the point and the subspace which is spanned by X and translated from the origin by v.
    """
    if point.ndim > 1:
        return np.linalg.norm(np.dot(point, null_space(X)), ord=2, axis=1)
    return np.linalg.norm(np.dot(point, null_space(X)))


def computeDistanceToSubspaceviaNullSpace(point, null_space):
    """
    This function is responsible for computing the distance between a point and a J dimensional affine subspace.

    :param point: A numpy array representing a .
    :param X: A numpy matrix representing a basis for a J dimensional subspace.
    :param v: A numpy array representing the translation of the subspace from the origin.
    :return: The distance between the point and the subspace which is spanned by X and translated from the origin by v.
    """
    if point.ndim > 1:
        return np.linalg.norm(np.dot(point, null_space), ord=2, axis=1)
    return np.linalg.norm(np.dot(point, null_space))


def computeCost(P, w, X, show_indices=False):
    """
    This function represents our cost function which is a generalization of k-means where the means are now J-flats.

    :param P: A weighed set, namely, a PointSet object.
    :param X: A numpy matrix of J x d which defines the basis of the subspace which we would like to compute the
              distance to.
    :param v: A numpy array of d entries which defines the translation of the J-dimensional subspace spanned by the
              rows of X.
    :return: The sum of weighted distances of each point to the affine J dimensional flat which is denoted by (X,v)
    """
    global OBJECTIVE_LOSS
    if X.ndim == 2:
        dist_per_point = OBJECTIVE_LOSS(
            computeDistanceToSubspaceviaNullSpace(P, null_space(X))
        )
        cost_per_point = np.multiply(w, dist_per_point)
    else:
        temp_cost_per_point = np.empty((P.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            temp_cost_per_point[:, i] = np.multiply(
                w,
                OBJECTIVE_LOSS(
                    computeDistanceToSubspaceviaNullSpace(
                        P, null_space(X[i, :, :])
                    )
                ),
            )

        cost_per_point = np.min(temp_cost_per_point, 1)
        indices = np.argmin(temp_cost_per_point, 1)
    if not show_indices:
        return np.sum(cost_per_point), cost_per_point
    else:
        return np.sum(cost_per_point), cost_per_point, indices


def computeSuboptimalSubspace(P, w, J):
    """
    This function computes a suboptimal subspace in case of having the generalized K-means objective function.

    :param P: A weighted set, namely, an object of PointSet.
    :return: A tuple of a basis of J dimensional spanning subspace, namely, X and a translation vector denoted by v.
    """

    start_time = time.time()

    _, _, V = np.linalg.svd(
        P, full_matrices=False
    )  # computing the spanning subspace
    return V[:J, :], time.time() - start_time


def EMLikeAlg(P, w, j, k, steps, NUM_INIT_FOR_EM=10):
    """
    The function at hand, is an EM-like algorithm which is heuristic in nature. It finds a suboptimal solution for the
    (K,J)-projective clustering problem with respect to a user chosen

    :param P: A weighted set, namely, a PointSet object
    :param j: An integer denoting the desired dimension of each flat (affine subspace)
    :param k: An integer denoting the number of j-flats
    :param steps: An integer denoting the max number of EM steps
    :return: A list of k j-flats which locally optimize the cost function
    """

    start_time = time.time()
    np.random.seed(random.seed())
    n, d = P.shape
    min_Vs = None
    optimal_cost = np.inf
    # print ("started")
    for iter in range(NUM_INIT_FOR_EM):  # run EM for 10 random initializations
        Vs = np.empty((k, j, d))
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        idxs = np.array_split(idxs, k)  # ;print(idxs)
        for i in range(k):  # initialize k random orthogonal matrices
            Vs[i, :, :], _ = computeSuboptimalSubspace(
                P[idxs[i], :], w[idxs[i]], j
            )

        for i in range(
            steps
        ):  # find best k j-flats which can attain local optimum
            dists = np.empty(
                (n, k)
            )  # distance of point to each one of the k j-flats
            for l in range(k):
                _, dists[:, l] = computeCost(P, w, Vs[l, :, :])

            cluster_indices = np.argmin(
                dists, 1
            )  # determine for each point, the closest flat to it
            unique_idxs = np.unique(
                cluster_indices
            )  # attain the number of clusters

            for (
                idx
            ) in (
                unique_idxs
            ):  # recompute better flats with respect to the updated cluster matching
                Vs[idx, :, :], _ = computeSuboptimalSubspace(
                    P[np.where(cluster_indices == idx)[0], :],
                    w[np.where(cluster_indices == idx)[0]],
                    j,
                )

        current_cost = computeCost(P, w, Vs)[0]
        if current_cost < optimal_cost:
            min_Vs = copy.deepcopy(Vs)
            optimal_cost = current_cost
        print(
            "finished iteration number {} with cost {}".format(
                iter, optimal_cost
            )
        )
    return min_Vs, time.time() - start_time


"""

"""


def main():
    pass


if __name__ == "__main__":
    main()
