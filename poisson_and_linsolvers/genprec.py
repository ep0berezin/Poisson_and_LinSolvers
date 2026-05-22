import numpy as np
import scipy.sparse as scsp

def pathes(row, col, printout = False): #generate row and col for COO equivalent to A^2.  In other words, for a given graph with [*row elem*]->[*col elem*] adds pathes of length 2.
	row = list(row)
	col = list(col)
	for k in range(len(col)):
		if row[k]!=col[k]:
			for p in range(len(col)):
				if  row[p]!=col[p] and col[k] == row[p] and row[k] != col[p] :
					not_duplicate = True
					for m in range(len(col)):
						if row[m] == row[k] and col[m] == col[p] : 
							not_duplicate = False
							break
					if not_duplicate :
						row.append(row[k])
						col.append(col[p])
						if printout : print(f"Added path {row[k]} -> {col[p]} at step {k}")
	
	return row, col
	
def ilu(p, A, printout= False):
	row = A.row
	col = A.col
	for _ in range(p):
		row, col = pathes(row, col, printout) #generate pattern by recursively constructing A^p 's rows and cols.
	row = np.array(row)
	col = np.array(col)
	A = scsp.coo_matrix((   np.hstack([ A.data, np.zeros(len(row) - len(A.data)) ]), (row, col)), shape=(A.shape[0],A.shape[1])) #add zeros to matrix in positions of a new sparsity pattern
	A = A.tocsr() #convert to CSR to use indexing and row-wise axpy
	edges = set(zip(row,col)) #make set of edges for fast pattern check
	for j in range(A.shape[1]):
		for i in reversed(range(j+1,A.shape[0])):
			if (i,j) in edges :
				gamma = - A[i,j]/A[j,j]
				if printout : print(f"({i},{j}) gamma = - {A[i,j]}/{A[j,j]} =  {gamma}")
				if printout : print(A)
				if printout : print("-->")
				A[i,j:] += A[j,j:]*gamma  #CSR allows to use slices
				A[i,j] = -1*gamma
				if printout : print(A)
	L = scsp.tril(A) #cut L with main diagonal to avoid sparse structure modificiation -- to just overwrite diagonal elements.
	L.setdiag(1.) #efficient setdiag using CSR format.
	L = L.tocsr()
	U = scsp.triu(A).tocsr()
	return L, U

