import numpy as np
import scipy as sc

class iterations_data:
	def __init__(self, valtypes):
		self.values = { key : [] for key in valtypes }
		self.elapsed = 0.
	def saveval(self, val, valtype, printout = True):
		if printout : print(f"{valtype}: {val}")
		self.values[valtype].append(val)
		return val


def GMRES(mv, b, x_0, m, eps_zero=1e-15, printout=True): #GMRES without restarts
	iterdata_vals_types = ["iteration", "residual"]
	iterdata = iterations_data(iterdata_vals_types)
	save_residual = lambda val : iterdata.saveval(val, "residual", printout)
	save_iteration = lambda iteration : iterdata.saveval(iteration, "iteration", printout)
	last_step = False
	H = np.zeros((m+1,m))
	V = np.zeros((x_0.shape[0],m+1))
	r = b - mv(x_0)
	save_iteration(0)
	save_residual(np.linalg.norm(r))
	beta = np.linalg.norm(r)
	V[:,0] = r/beta
	for j in range(m): #Arnoldi procedure to build Krylov subspace basis
		save_iteration(j+1)
		w = mv(V[:,j]) #next Krylov vector
		for i in range(j+1):
			h = np.dot(w,V[:,i]) #compute projection
			H[i,j] = h #save h to Hessenberg matrix
			w -= h*V[:,i] #substract projection
		H[j+1,j] = np.linalg.norm(w)
		print(f"h_{{j+1,j}} = {H[j+1,j]}")
		if H[j+1,j] < eps_zero: #if triggered - dim(K(b,A)) = j+1
			#GoTo is not needed since we compute x at all iterations. So solution for nonzero columns of Hessenberg already constructed.
			print(f"h_{{j+1,j}} = 0. Full orthonormal basis constructed.")
			last_step = True
		else:
			V[:,j+1] = w/H[j+1,j]
		e1 = np.zeros(j+2)
		e1[0]=1.
		#y = np.linalg.pinv(H[:j+2,:j+1])@(beta*e1)
		Q, R = np.linalg.qr(H[:j+2,:j+1])
		#y =  np.linalg.inv(R)@Q.T@e1*beta #inv inefficient
		y = sc.linalg.solve_triangular(R, Q.T@e1*beta, check_finite=False)
		x = x_0 + V[:,:j+1]@y
		r_m = b - mv(x)
		save_residual( np.linalg.norm(r_m) )
		if last_step : return x, iterdata
	return x, iterdata

def CG(mv, b, x_0, m, eps=1e-15, printout=True): #Conjugate gradient
	iterdata_vals_types = ["iteration", "residual"]
	iterdata = iterations_data(iterdata_vals_types)
	save_residual = lambda val : iterdata.saveval(val, "residual", printout)
	save_iteration = lambda iteration : iterdata.saveval(iteration, "iteration", printout)
	r = b - mv(x_0)
	save_iteration(0)
	save_residual(np.linalg.norm(r))
	p = r
	x = x_0
	for j in range(m):
		matvec = mv(p)
		alpha = np.dot(r,r) / np.dot(matvec,p)
		x += alpha*p
		r_new = r - alpha*matvec
		beta = np.dot(r_new,r_new)/np.dot(r,r)
		p_new = r_new + beta*p
		
		p = p_new
		r = r_new
		save_iteration(j)
		save_residual(np.linalg.norm(r))
		if np.linalg.norm(r) < eps: 
			return x, iterdata
		
	return x, iterdata

def BiCG(mv, mvT, b, x_0, m, eps=1e-15, printout=True):
	iterdata_vals_types = ["iteration", "residual"]
	iterdata = iterations_data(iterdata_vals_types)
	save_residual = lambda val : iterdata.saveval(val, "residual", printout)
	save_iteration = lambda iteration : iterdata.saveval(iteration, "iteration", printout)
	
	r = b - mv(x_0)
	rs = r
	p = r
	ps = rs

	save_iteration(0)
	save_residual(np.linalg.norm(r))
	
	x = x_0
	for j in range(m):
		matvec = mv(p)
		matvecT = mvT(ps)
		alpha = np.dot(r,rs)/np.dot(matvec, ps)
		x += alpha*p
		r_new = r - alpha*matvec
		rs_new = rs - alpha*matvecT
		beta = np.dot(r_new, rs_new)/np.dot(r,rs)
		p_new = r_new + beta*p
		ps_new = rs_new + beta*ps
		
		p = p_new
		ps = ps_new
		r = r_new
		rs = rs_new
		
		save_iteration(j)
		save_residual(np.linalg.norm(r))
		if np.linalg.norm(r) < eps:
			return x, iterdata
		
	return x, iterdata
