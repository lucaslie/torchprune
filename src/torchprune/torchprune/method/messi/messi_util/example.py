import numpy as np
import factor

#%%

n = 32
d = 64
A = np.random.random((n, d))

NUM_INIT_FOR_EM = 20  ## how many times to try differecnt seeds
steps = 25  # EM algorhtm steps
j = 10
k = 3

partition, listU, listV = factor.raw(
    A, j=j, k=k, steps=steps, NUM_INIT_FOR_EM=NUM_INIT_FOR_EM
)
zero_idxes = factor.getZeros(partition, j, k)
U_ranked, V_ranked = factor.stitch(partition, listU, listV)
## U_ranked has n rows and kj columns, each row has at most j nonz zero entires (at least j(k-1) zerors)
## V_ranked has kj rows and d columns
## See appendix in the MESSI paper.
print("=================summary:=================")
print(
    "non zero intries in U_ranked = {}, j*n = {} ".format(
        np.count_nonzero(U_ranked), j * n
    )
)
num_of_params = (
    np.count_nonzero(U_ranked) + V_ranked.shape[0] * V_ranked.shape[1]
)
print("number of prams = {}".format(num_of_params))
print("error = {} ".format(np.linalg.norm(U_ranked.dot(V_ranked) - A, 2) ** 2))

######################SVD TIME##############################3
rank = (
    int(num_of_params / (n + d)) + 1
)  #### EVEN WHEN SVD HAS MORE PARAMETERS WE WIN
U, D, V = np.linalg.svd(A)
svd_u = np.zeros((U.shape[0], rank))
svd_u[:rank, :rank] = np.diag(D[:rank])

svd_v = np.zeros((rank, V.shape[0]))
svd_v[:rank, :rank] = np.diag(D[:rank])

u_new = np.dot(U, np.sqrt(svd_u))
v_new = np.dot(np.sqrt(svd_v), V)
print("=================SVD summary:=================")
num_of_params_svd = (
    u_new.shape[0] * u_new.shape[1] + v_new.shape[0] * v_new.shape[1]
)
print("svd number of prams = {}".format(num_of_params_svd))
print("svd error = {} ".format(np.linalg.norm(u_new.dot(v_new) - A, 2) ** 2))
