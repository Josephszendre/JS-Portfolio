# iterative_solvers.py
"""Volume 1A: Iterative Solvers.
Joseph Szendre
321 Blue
10.29.16
"""
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
import time
#import pdb
# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant nxn matrix.

    Inputs:
        n (int): the dimension of the system.
        num_entries (int): the number of nonzero values. Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): An nxn strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in xrange(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in xrange(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A


# Problems 1 and 2
def jacobi_method(A, b, tol=1e-8, maxiters=100, plot=False):
    """Calculate the solution to the system Ax = b voa the Jacobi Method.

    Inputs:
        A ((n,n) ndarray): A square matrix.
        b ((n,) ndarray): A vector of length n.
        tol (float, opt): the convergence tolerance.
        maxiters (int, opt): the maximum number of iterations to perform.
        plot (bool, opt): if True, plot the convergence rate of the algorithm.
            (this is for Problem 2).

    Returns:r
        x ((n,) ndarray): the solution to system Ax = b.
    """
    """
    Pseudocode
    Ensure A, b are in decimal form
    
    
    
    """
    
    
    """
    e:0...n inclusive
    """
    A=np.array(A)*1.0
    b=np.array(b)*1.0    
    m,n=A.shape
    e=[]
    xk=np.zeros((m,))
    
    def iter(xi):
        xj=np.zeros((m,))
        for i in xrange(m):
            xj[i]=(b[i]-(np.dot(A[i],xi)-A[i,i]*xi[i]))/A[i,i]
        return xj

            
    for i in xrange(1,maxiters+1):
        e+=[la.norm(np.dot(A,xk)-b,ord=np.inf)]
        xk=iter(xk)
        if (la.norm(np.dot(A,xk)-b,ord=np.inf)<tol) or (i==maxiters):
            e+=[la.norm(np.dot(A,xk)-b,ord=np.inf)]
            break
            
    if plot==False:
        return xk
    else:
        #How many iterations happened
        iters=len(e) #1..len(e)
        dom=np.arange(0,iters)
        
        plt.semilogy(dom,e,'b.-',basey=10,lw=2, ms=2)
        plt.xlabel("Iteration #")
        plt.ylabel("Absolute Error of Approximation")
        #plt.legend(loc="upper left")
        plt.title("Convergence of Jacobi Method", fontsize=18)
        plt.show()
        return xk
        

        
        
    

# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiters=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Inputs:
        A ((n,n) ndarray): A square matrix.
        b ((n,) ndarray): A vector of length n.
        tol (float, opt): the convergence tolerance.
        maxiters (int, opt): the maximum number of iterations to perform.
        plot (bool, opt): if True, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): the solution to system Ax = b.
    """
    A=np.array(A)*1.0
    b=np.array(b)*1.0    
    m,n=A.shape
    e=[]
    xk=np.zeros((m,))
    
    def iter(xi):
        xj=np.zeros((m,))
        for i in xrange(m):
            xj[i]=(b[i]-(np.dot(A[i],xi)-A[i,i]*xi[i]))/A[i,i]
            xi[i]=xj[i]
        return xj

    if plot==True:    
        for i in xrange(1,maxiters+1):
            e+=[la.norm(np.dot(A,xk)-b,ord=np.inf)]
            #print i-1,e[i-1],xk
            xk=iter(xk)
            if (la.norm(np.dot(A,xk)-b,ord=np.inf)<tol) or (i==maxiters):
                e+=[la.norm(np.dot(A,xk)-b,ord=np.inf)]
                break
            #How many iterations happened
            iters=len(e) #1..len(e)
            dom=np.arange(0,iters)
        
            plt.semilogy(dom,e,'b.-',basey=10,lw=2, ms=2)
            plt.xlabel("Iteration #")
            plt.ylabel("Absolute Error of Approximation")
            #plt.legend(loc="upper left")
            plt.title("Convergence of Gauss-Seidel Method", fontsize=18)
            plt.show()
            return xk
           
    else:
        for i in xrange(1,maxiters+1):
            xk=iter(xk)
            if (la.norm(np.dot(A,xk)-b,ord=np.inf)<tol) or (i==maxiters):
                return xk
        

# Problem 4
def prob4():
    """For a 5000 parameter system, compare the runtimes of the Gauss-Seidel
    method and la.solve(). Print an explanation of why Gauss-Seidel is so much
    faster.
    """
    print "Lab pdf is wrong here"
    t=0.0
    t1=[]
    t2=[]
    for i in xrange(5,12):
        A=diag_dom(2**i,2**(i+1))
        b=np.random.random((2**i,))
        
        t=time.time()
        gauss_seidel(A,b,maxiters=1000)
        t1+=[time.time()-t]
        
        t=time.time()
        la.solve(A,b)
        t2+=[time.time()-t]

    dom=2**np.arange(5,12)    
    plt.loglog(dom,t1,'b.-',basey=2,basex=2,lw=2, ms=2, label="Gauss-Seidel Approximation Time")
    plt.loglog(dom,t2,'r.-',basey=2,basex=2,lw=2, ms=2, label="la.solve(A,b)")
    plt.xlabel("Matrix size nxn")
    plt.ylabel("Time to Find Approximation")
    plt.legend(loc="upper left")
    plt.title("Temporal Complexity of Gauss-Seidel vs la.solve(A,b)", fontsize=18)
    plt.show()    

# Problem 5
def sparse_gauss_seidel(A, b, tol=1e-8, maxiters=29):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Inputs:
        A ((n,n) csr_matrix): An nxn sparse CSR matrix.
        b ((n,) ndarray): A vector of length n.
        tol (float, opt): the convergence tolerance.
        maxiters (int, opt): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): the solution to system Ax = b.
    """
     

    def iter(xi):
        xj=np.zeros((m,))
        for i in xrange(m): 
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            aii=A[i,i]
            xj[i]=(b[i]-(np.dot(A.data[rowstart:rowend], xi[A.indices[rowstart:rowend]])-aii*xi[i]))/(aii)
            xi[i]=xj[i]
        return xj
            
        #Aix = np.dot(A.data[rowstart:rowend], x[A.indices[rowstart:rowend]])

    m=len(b)
    xk=np.zeros((m,))
    for i in xrange(0,maxiters):
        xk=iter(xk)
        if (la.norm(A.dot(xk)-b,ord=np.inf)<tol) or (i==maxiters-1):
            return xk

# Problem 6
def sparse_sor(A, b, omega, tol=1e-8, maxiters=29):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Inputs:
        A ((n,n) csr_matrix): An nxn sparse matrix.
        b ((n,) ndarray): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float, opt): the convergence tolerance.
        maxiters (int, opt): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): the solution to system Ax = b.
    """
    def iter(xi):
        xj=np.zeros((m,))
        for i in xrange(m): 
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            aii=A[i,i]
            xj[i]=xi[i]+omega*(b[i]-np.dot(A.data[rowstart:rowend], xi[A.indices[rowstart:rowend]]))/(aii)
            xi[i]=xj[i]
        return xj
            
        #Aix = np.dot(A.data[rowstart:rowend], x[A.indices[rowstart:rowend]])

    m=len(b)
    xk=np.zeros((m,))
    for i in xrange(0,maxiters):
        xk=iter(xk)
        if (la.norm(A.dot(xk)-b,ord=np.inf)<tol) or (i==maxiters-1):
            return xk

# Problem 7
def finite_difference(n):
    """Return the A and b described in the finite difference problem that
    solves Laplace's equation.
    """
    B=sparse.diags([1,-4,1],[-1,0,1],shape=(n,n))
    A=sparse.block_diag([B for i in xrange(0,n)])
    A.setdiag(1,k=-n)
    A.setdiag(1,k=n)    
    b=[-100]+[0 for i in xrange(1,n-1)]+[-100]
    b=b*n
    return A,b


# Problem 8
def compare_omega():
    """Time sparse_sor() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiters = 1000 using the A and b generated by finite_difference()
    with n = 20. Plot the times as a function of omega.
    """
    t=0.0
    t1=[]
    W=[1,1.05]+[1.0+i*1.0/10.0 for i in xrange(1,10)]+[1.95]
    A,b=finite_difference(20)
    for w in W:        
        t=time.time()
        sparse_sor(csr_matrix(A),b,w,tol=1e-2,maxiters=1000)
        t1+=[time.time()-t]
        
    plt.plot(W,t1,'r.-',lw=2, ms=2)
    plt.xlabel("omega")
    plt.ylabel("Time needed to reach tol=1e-2")
    plt.title("SOR Convergence Times with Omega", fontsize=18)
    plt.show()    


# Problem 9
def hot_plate(n,c=None):
    """Use finite_difference() to generate the system Au = b, then solve the
    system using SciPy's sparse system solver, scipy.sparse.linalg.spsolve().
    Visualize the solution using a heatmap using np.meshgrid() and
    plt.pcolormesh() ("seismic" is a good color map in this case).
    """
    A,b=finite_difference(n)
    if c is None:
        u=linalg.spsolve(A.tocsr(),b)
        
    else:
        u=linalg.spsolve(A.tocsr(),c)
        
    
    x=np.linspace(1,n,n)
    y=np.linspace(1,n,n)
    X,Y=np.meshgrid(x,y)
    Z=u.reshape((n,n))
    
    
    
    
    # Plot the heat map of f over the 2-D domain.
    plt.pcolormesh(X, Y, Z, cmap="seismic")
    plt.colorbar()
    plt.xlim(1,n)
    plt.ylim(1,n)
    plt.show()
    
if __name__=="__main__":
    A=csr_matrix(diag_dom(500))
    b=np.random.random(500)
    #A,b=finite_difference(4)
    #print A.toarray(),"\n",b
    x=np.linspace(1,300,300)
    y=np.linspace(1,300,300)
    X,Y=np.meshgrid(x,y)
    Z=np.sin(X)+5*np.tan(Y)
    hot_plate(2500)
    #print np.log2(la.norm(A.dot(sparse_sor(A,b,1.1,tol=1e-30,maxiters=10))-b,ord=np.inf))