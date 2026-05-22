import genmatrix as gm
import genprec as gp
import numpy as np
import solvers as slv
import scipy.sparse as scsp
import matplotlib.pyplot as plt
import argparse

np.set_printoptions(precision=16, suppress=True, linewidth=1000, threshold=10000)

def plot_residuals(iterdata):
	fig, ax = plt.subplots(1,1,figsize = (6,6))
	ax.plot( iterdata.values["iteration"], iterdata.values["residual"]/ iterdata.values["residual"][0] )
	ax.set_xlabel(fr"iteration")
	ax.set_ylabel(fr"$\|r\|_2$")
	plt.show()
	
def plot_residuals_relations(iterdata):
	fig, ax = plt.subplots(1,1,figsize = (6,6))
	r = np.array(iterdata.values["residual"])
	
	ratios = r[1:] / r[:-1]
	iterations = iterdata.values["iteration"][1:]
	
	ax.plot(iterations, ratios)
	ax.set_xlabel("iteration")
	ax.set_ylabel(r"$||r_{i+1}|| / ||r_i||$")
	plt.show()

def apply_Minv(L, U, v):
	z = scsp.linalg.spsolve_triangular(L, v, lower=True, unit_diagonal=True)
	z = scsp.linalg.spsolve_triangular(U, z, lower=False)
	return z

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-ms","--meshsize", help="specify mesh size", default=3)
	parser.add_argument("-it", "--maxiter", help="specify max iter", default=100)
	parser.add_argument("-sl", "--solver", help="specify solver")
	parser.add_argument("-pc", "--preconditioner", help="specity preconditioner")
	parser.add_argument("-p", "--iluorder", help="specify ilu order")
	clargs = parser.parse_args()
	n = int(clargs.meshsize)
	N = n**2
	maxiter = int(clargs.maxiter)
	A = gm.Laplace_2D(n) #A in coo format
	Acsr = A.tocsr()
	
	if clargs.preconditioner == "ilu" :
		p=int(clargs.iluorder) #ILU(p) order
		L, U = gp.ilu(p, A, True)
		matvec = lambda x: Acsr @ apply_Minv(L, U, x) #preconditioned matvec
		matvecadj = lambda x: Acsr.T @ apply_Minv(L, U, x) #preconditioned matvec
	else :
		matvec = lambda x: Acsr @ x  #regular matvec
		matvecadj = lambda x: Acsr.T @ x # regular matvec adjoint
	f = np.ones(N)

	if clargs.solver == "gmres" :
		print("GMRES")
		sol, iterdata = slv.GMRES(matvec, f, np.zeros(N), maxiter)
	if clargs.solver == "cg" :
		print("CG")
		sol, iterdata = slv.CG(matvec, f, np.zeros(N), maxiter)
	if clargs.solver == "bicg" :
		print("BiCG")
		sol, iterdata = slv.BiCG(matvec, matvecadj, f, np.zeros(N), maxiter)
	if clargs.preconditioner == "ilu" : 
		x = apply_Minv(L, U, sol)
		
	plot_residuals(iterdata)




