"""
name:    factor.py
author:  Harry Lang

researchers (last-name alphabetical-order):
    Dan Feldman
    Harry Lang
    Alaa Malouf
    Daniela Rus

usage:
    import factor
    U,V = factor.factorize(A, j, k)

unit tests:
    python3 factor.py

input:
    A       n x d matrix
    j       integer >= 1
    k       integer >= 1

output:
    U       n x jk matrix
    V       jk x d matrix

description:
    The product UV should be a smaller approximation of A.
    The rows of A are partitioned into k groups by treating
    its n rows as n points in R^d, then finding a (k,j) projective
    clustering and assigning each row to a nearest
    j-flat from the clustering.  This creates submatrices
    (A_1, ..., A_k).  Each A_i is factored as U_i * V_i
    in a j-rank approximation.  These j-rank approximations
    are stitched together to form the output pair (U,V).
"""

import numpy as np
from .testEM2 import computeCost, computeDistanceToSubspace, EMLikeAlg

"""
The main function of this file.
Factors a matrix into two components using
the process described at the top of this file.

@param A
    n x d matrix (NumPy array)
@param j
    integer >= 1, the dimension of each subspace in
    the projective clustering used to partition A and
    also the rank of each low-rank approximation used
    for each element of the partition
@param k
    integer >= 1, the number of subspaces used in the
    projective clustering
@return
    U, V where U is n x jk and V is jk x d
    so that UV approximates A
    TODO also return list from [0, ..., n-1]
    where list[i] is a list of columsn that are 0 in row i.
"""


def factorize(A, j, k):
    partition, listU, listV = raw(A, j, k)
    return stitch(partition, listU, listV)


"""
Combines k-matrices into one large block matrix (not a true block matrix unless
you permute the rows correctly).
"""


def stitch(partition, listU, listV):
    U = _stitchU(partition, listU)
    V = _stitchV(listV)
    return U, V


"""
Factorizes into k-matrices based on (k,j) projective clustering

@return
    3-tuple of partition, list of left-hand-side matrices,
    list of right-hand-side matrices
"""


def raw(A, j, k, steps=10, NUM_INIT_FOR_EM=10):

    # returns (k, j, d) tensor to represent the k flats
    # by j basis vectors in R^d
    flats = _getProjectiveClustering(
        A, j, k, steps=steps, NUM_INIT_FOR_EM=NUM_INIT_FOR_EM
    )

    # partition[i] == z means row i of A belongs to flat z
    # where 0 <= i < n and 0 <= z < k
    print("Timing partition...")
    import time

    start = time.time()
    partition = list(_partitionToClosestFlat_new(A, flats))
    end = time.time()
    duration = end - start
    print("Partition complete:", duration, "s")

    listU = []
    listV = []
    n = A.shape[0]
    for z in range(k):
        indices_z = [row for row in range(n) if partition[row] == z]
        A_z = A[indices_z, :]
        U_z, V_z = lowRank(A_z, j)
        listU.append(U_z)
        listV.append(V_z)

    return partition, listU, listV


"""
Generates the column indices for each row that must be 0 in the
stitched matrix U from stitch(partition, listU, listV).

@param dict
    list mapping the row number to the partition index in [0, ..., k-1]
@param j
    dimension of each subspace in the projective clustering
@param k
    number of subpaces in the projective clustering
@return
    list mapping the row number to a list of column indices that must be zero
"""


def getZeros(partition, j, k):

    # generate the k lists of zeroes
    zeros = []
    for i in range(k):
        list_i = list(range(i * j)) + list(range((i + 1) * j, k * j))
        zeros.append(list_i)

    # assign each row to its zero columns
    rows = []
    for i in range(len(partition)):
        rows.append(zeros[partition[i]])

    return rows


"""
Unit Test for _getZeros()
"""


def _testGetZeros():
    partition = [0, 2, 1, 1]  # n = 4, k = 3
    actual = getZeros(partition, 2, 3)  # j = 2
    expected = [[2, 3, 4, 5], [0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 4, 5]]
    if not expected == actual:
        print("Expected:", expected)
        print("Actual:", actual)
        raise Exception("Failed Test: getZeros()")
    else:
        print("Success: Test getZeros()")


"""
Partitions the rows of an n x d matrix according
to a nearest subspace from a list of subspaces in
R^d.

@param A
    n x d matrix, whose rows will be partitioned.
    Represents n points in R^d.
@param flats
    (k,j,d)-tensor representing the k subspaces to
    determine the partition. Each subspace is defined
    by j basis vectors in R^d.
@return partition
    a list taking values in [1, ..., k] where the
    i-th element denotes the index of the subspace
    that the i-th row of A was partitioned to.
"""


def _partitionToClosestFlat_new(A, flats):
    dists = np.empty((A.shape[0], flats.shape[0]))
    for l in range(flats.shape[0]):
        _, dists[:, l] = computeCost(A, np.ones(A.shape[0]), flats[l, :, :])

    cluster_indices = np.argmin(
        dists, 1
    )  # determine for each point, the closest flat to it
    return cluster_indices


def _partitionToClosestFlat(A, flats):

    # check compatible dimensions of R^d
    if A.shape[1] != flats.shape[2]:
        raise ValueError("Points and subspaces must belong to same space R^d")

    partition = []
    n = A.shape[0]
    k = flats.shape[0]
    for i in range(n):
        point = A[i, :]  # find closest subspace to this point
        winner = 0  # initialize winning flat index
        winningDistance = np.inf  # initialize winning cost
        for flat in range(k):
            distance = computeDistanceToSubspace(point, flats[flat])
            if distance < winningDistance:
                winningDistance = distance
                winner = flat
        partition.append(
            winner
        )  # partition[i] = index of nearest subspace to row i
    return partition


"""
High-performance implementation of _partitionToClosestFlat().

@date 18 September 2020
"""


def _partitionToClosestFlat_native(A, flats):

    k = flats.shape[0]
    j = flats.shape[1]
    d = flats.shape[2]

    # check compatible dimensions of R^d
    if A.shape[1] != d:
        raise ValueError("Points and subspaces must belong to same space R^d")

    # check all bases are orthonormal
    for z in range(k):
        basis = flats[z]
        near_zero = basis.dot(basis.T) - np.identity(j)
        # possibly not exactly 0 due to floating-point errors
        max_entry = np.max(np.abs(near_zero))
        if not max_entry < d * 1e-10:
            raise ValueError("Basis is non-orthonormal")

    partition = []
    n = A.shape[0]
    for i in range(n):
        point = A[i, :]  # find closest subspace to this point
        winner = 0  # initialize winning flat index

        # Observe that the minimum distance is the same
        # as the maximal projection.  To find the closest
        # flat, we instead find the flat that yields the
        # projection of greatest magnitude
        maxCost = 0  # initialize winning cost

        for z in range(k):  # for each flat
            basis = flats[z]  # j basis vectors in R^d
            sum_of_squares = 0
            for b in range(j):
                projection = basis[b].dot(point)
                sum_of_squares += projection * projection
            if sum_of_squares > maxCost:
                winner = z
                maxCost = sum_of_squares
        partition.append(
            winner
        )  # partition[i] = index of nearest subspace to row i
    return partition


"""
Unit test for _partitionToClosestFlat().

Example with n = 3, d = 3, k = 2, j = 2
"""


def _testPartitionToClosestFlat():

    flats = np.array(
        [
            [[1, 0, 0], [0, 1, 0]],  # flat 0  # orthogonal to [0,0,1]
            [[0.8, 0, 0.6], [0.6, 0, -0.8]],  # flat 1
        ]
    )  # orthogonal to [0,1,0]

    # distance to flat:        #   0  |  1
    A = np.array(
        [[1, 2, 4], [0, -1, 2], [1, 8, -3]]  #   4  |  2  #   2  |  1
    )  #   3  |  8

    actual = list(_partitionToClosestFlat_new(A, flats))
    # new =  list(_partitionToClosestFlat_new(A, flats))
    expected = [1, 1, 0]
    print(actual)

    if not (expected == actual):
        print("Expected:", expected)
        print("Actual:", actual)

        raise Exception("Failed Test: _partitionToClosestFlat()")
    else:
        print("Success: Test _partitionToClosestFlat()")


"""
Computes the low-rank approximation of a matrix.

This computes a (possibly non-unique) minimizer M' of
the Frobenius norm of (M - M') for input M where
M' has rank at most r.  Uniqueness is determined by uniqueness
of the greatest r singular values of M.  The computation
is by truncating the SVD of M.

@param M
    n x d matrix, the matrix to be approximated
@param r
    integer >= 1, the rank of the approximation
@return
    U,V where U.shape==(n,r) and V.shape==(r,d)
    defining an r-rank approximation of M
"""


def lowRank(M, r):

    U, D, Vt = np.linalg.svd(M)

    # truncate to:
    #   left-most r columns of U
    #   first r values of D
    #   top-most r rows of Vt
    U_trunc = U[:, :r]
    D_trunc = np.diag(D[:r])  # also convert from vector to matrix
    Vt_trunc = Vt[:r, :]

    # arbitrary choice to combine D with either side
    return U_trunc.dot(D_trunc), Vt_trunc


"""
Stitches together U_1, ..., U_k into a single
n x r matrix U.  The column space, of dimension r,
is the direct-sum space of the column spaces of
U_1, ..., U_k.

@param partition
    list of length n containing values in [0, ..., k-1].
    partition[i] is the component of the partition that
    row i of the input matrix belongs to.
@param listU
    list of the left-hand-side matrices of the low-rank
    approximation of the submatrices built by partitioning
    the rows of the input matrix
@return U
    a global reconstruction that can be used to as the
    left-hand-side matrix of the decomposition of the input
    matrix
"""


def _stitchU(partition, listU):

    # n: rows of original matrix == size of list partitions
    n = len(partition)

    # r: middle space between R^n and R^d, of dimension r where
    # r is the sum of all column-spaces of each U_z in listU
    r = 0
    for U_z in listU:
        r += U_z.shape[1]

    U = np.zeros((n, r))  # final U is mostly zeros

    # counter[z] stores current row of listU[z]
    counters = [0] * len(listU)

    for row in range(n):
        index_component = partition[row]
        index_row = counters[index_component]  # row of U_z to use
        counters[index_component] += 1

        # insert row of U_z in column-space of R^r starting at col_start
        col_start = 0
        for u in listU[:index_component]:
            col_start += u.shape[1]
        component = listU[index_component]
        col_end = col_start + component.shape[1]
        U[row, col_start:col_end] = component[index_row]
    return U


"""
Unit test for _stitchU()

Example with n = 4, k = 2
"""


def _testStitchU():
    partition = [0, 1, 1, 0]
    U1 = np.array([[1, 2], [3, 4]])
    U2 = np.array([[21, 22], [23, 24]])
    expected = np.array(
        [[1, 2, 0, 0], [0, 0, 21, 22], [0, 0, 23, 24], [3, 4, 0, 0]]
    )

    actual = _stitchU(partition, [U1, U2])

    if not np.array_equal(expected, actual):
        print("Expected:", expected)
        print("Actual:", actual)
        raise Exception("Failed Test: _stitchU()")
    else:
        print("Success: Test _stitchU()")


"""
Takes the k matrices V_1, ..., V_k each of dimension j x d
from the k low-rank approximations, and stitches them together
into one large jk x d matrix.

@param listV
    [V_1, ..., V_k]
@return:
    jk x d matrix V that can be used as the right-hand-side
    matrix of the decomposition of the input matrix A ~ U.dot(V)
"""


def _stitchV(listV):
    return np.concatenate(listV)


"""
Randomized algorithm to compute projective clustering.
Each of the j-dimensional subspaces pass through origin.

@param P
    n x d matrix
@param j
    integer >= 1, the dimension of each flat
@param k
    the number of flats
@param verbose
    if False, disables printing within the clustering subroutine
@return:
    (k, j, d)-tensor where the first index identifies
    the flat, the second index identifies the basis
    vector, and the third index identifies the coordinate.
"""


def _getProjectiveClustering(
    P, j, k, verbose=True, steps=15, NUM_INIT_FOR_EM=10
):
    n = P.shape[0]
    w = np.ones(n)  # unit weights
    # steps = 15 # number of EM steps

    if not verbose:
        import os
        import sys

        sys.stdout = open(os.devnull, "w")  # disable printing
    flats, runtime = EMLikeAlg(
        P, w, j, k, steps=steps, NUM_INIT_FOR_EM=NUM_INIT_FOR_EM
    )
    if not verbose:
        sys.stdout = sys.__stdout__  # re-enable printing

    return flats


"""
Unit test for factorize() to check that
it produces matrices of the correct dimension.
"""


def _testFactorizeDim():

    n = 45
    d = 18
    j = 3
    k = 5

    np.random.seed(15937)  # seed so A is deterministic
    A = np.random.rand(n, d)

    U, V = factorize(A, j, k)

    expected = A.shape
    actual = U.dot(V).shape
    if not expected == actual:
        print("Expected:", expected)
        print("Actual:", actual)
        raise Exception("Failed Test: _factorize_dim()")
    else:
        print("Success: Test _factorize_dim()")


# run unit tests
if __name__ == "__main__":
    try:
        _testPartitionToClosestFlat()
        _testStitchU()
        _testFactorizeDim()
        _testGetZeros()
    except:
        print("TESTS FAILED\n")
        raise
    else:
        print("ALL TESTS PASSED")
