import numpy as np
import scipy.sparse as scsp



def Laplace_2D(n):
	N = n**2
	h = 1/(n-1)
	hfac = -1/h**2
	print(f"-1/h^2 = {hfac}")
	row = np.hstack([  np.arange(N), np.arange(N-1), np.arange(1,N), np.arange(N-n), np.arange(n, N)  ])
	col = np.hstack([  np.arange(N), np.arange(1,N), np.arange(N-1), np.arange(n, N), np.arange(N-n)  ])
	#print(row)
	#print(col)
	nnz = row.shape[0]
	data = np.zeros(nnz)
	for k in range(nnz):
		i = row[k]
		j = col[k]
		if i==j:
			data[k] = -2*hfac*2
			#print(f"(A cronecker I + I cronecker A ){i},{j}")
		if i==j+1 or j==i+1:
			data[k] = hfac
			#print(f"(I cronecker A ){i},{j}")
		if i==j+n or j==i+n:
			#print(f"(A cronecker I ){i},{j}")
			data[k] = hfac
	A = scsp.coo_matrix((data, (row, col)), shape=(N,N))
	return A
