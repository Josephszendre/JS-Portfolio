# simplex.py
"""Volume 2B: Simplex.
<Name>
<Class>
<Date>

Problems 1-6 give instructions on how to build the SimplexSolver class.
The grader will test your class by solving various linear optimization
problems and will only call the constructor and the solve() methods directly.
Write good docstrings for each of your class methods and comment your code.

prob7() will also be tested directly.
"""

import numpy as np
# Problems 1-6

class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        maximize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """

    def __init__(self, c, A, b):
        """

        Parameters:
            c (1xn ndarray): The coefficients of the linear objective function.
            A (mxn ndarray): The constraint coefficients matrix.
            b (1xm ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """

        if not self.isFeasible(b):
            raise ValueError("The problem:\nmaximize        c^Tx\nsubject to      Ax <= b\nx >= 0 is not feasible at the origin as required.")                 
        m,n=A.shape
        self.m,self.n=A.shape
        self.tableau=self.constructTableau(np.array(c),np.array(A),np.array(b))
        self.basic=set(np.arange(n+1,m+n+1))
        self.original=set(xrange(1,m+n+1))        

        
             
    def constructTableau(self,c,A,b):
        #top row
        r=np.zeros((self.m+self.n+2,))
        r[1:c.flatten().shape[0]+1]=-c.flatten()
        r[r.shape[0]-1]=1.
        print r
        #remaining part
        b=np.hstack((np.reshape(b,(A.shape[0],1)),A,np.eye(A.shape[0]),np.zeros((A.shape[0],1))))
        R=np.vstack((r,b))
        print R
        return R
       

    def pivot(self):
        m=self.m
        n=self.n
        i,j=self.blandWayOfPickingPivotColRow()

 
        print "pivoting on ",i,j
        """
        pivots with variable i entering, variable j leaving        
        1) divide row 3 by the negative of its third entry 
           and divide rows 1,2 by their 3rd entry
        """
								#1 Pivot row gets divided by pivot entry
        self.tableau[i,:]/=self.tableau[i,j]

        #2 Pivot row gets added to eliminate 0s on the pivot column
        a=set(np.arange(self.m+1))-set([i])


        for k in a:
            self.tableau[k,:]+=self.tableau[i,:]*(-1.*self.tableau[k,j])
        print "after pivot on",i,j
        print self.tableau								
        




        
    
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The maximum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        ct=0
        self.time=0
        while (not self.isOptimal() and ct!=100):
            self.pivot()
            ct+=1
            
            
            
        print "found optimal set"
        a1={}    
        a=set()
        for i,x in enumerate(self.tableau[0,1:self.m+self.n+1]):
            if np.abs(x)<1e-13:
                a.add(i)
                a1[i]=self.tableau[np.argmax(self.tableau[1:,i+1].flatten())+1,0]
                
        b1={}
        b=set(np.arange(0,self.m+self.n))-a
        for i in b:
            b1[i]=0.
        
            
        
        return self.tableau[0,0],a1,b1

            
    def blandWayOfPickingPivotColRow(self):
        # counter variables i,j,k
        firstcol=0
        for i,x in enumerate(self.tableau[0,1:self.m+self.n+1]):
            if firstcol==0 and x<-1e-4:
                firstcol=i+1
        j=firstcol
        
        #check for unboundedness
        firstrow=0
        for i,x in enumerate(self.tableau[1:,j]):
            if x>0:
                firstrow=i+1
          
        if (firstrow==0):
            raise ValueError("The problem is unbounded yo.")
        
        least=1e14*1.
        i=0

        for k,x in enumerate(self.tableau[1:,j]):
            if x>0.and self.tableau[k+1,0]!=0. and self.tableau[k+1,0]/x<least:
                least=self.tableau[k+1,0]/x
                i=k+1
        if i==0:
            raise ValueError("row picked should not be 0")
        
        return i,j
                
                
             
        
 

        
    def isOptimal(self):
        firstcol=0
        for i,x in enumerate(self.tableau[0,1:self.m+self.n+1]):
            print i,x
            if firstcol==0 and x<-1e-13:
                print "not optimal"
                return False

        return True
        
        
   
    def isFeasible(self,b):
        """
        checks to see if the problem is feasible
        at the origin
        """
        for i in b.flatten():
            if i<0.:
                return False
        return True
        
            
        
        
        
    """
    Can assume that 0 is feasible.
   
    Checks: If at any time c_i is all negative, we have reached an optimal value

    How to pivot
    
    
    """
    

# Problem 7
def prob7(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        The minimizer of the problem (as an array).
    """
    A=np.load(filename)
    d= np.vstack((A['A'],np.eye(4)))
    print A['p']
    b=np.reshape(np.hstack((A['m'],A['d'])),(-1,1))
    c = SimplexSolver(A['p'],d,b)
    return A,c.solve()
    

if __name__ == "__main__":
    sol=prob7()
    print sol
    
    """
    a=np.array([[  0.,   5.,   2.,  3.],
       [  1.,  10.,  -2.,  -4.],
       [  1.,  12.,  -3.,  -5.],
       [  1.,  13.,  -6.,  -3.]])

    d=np.array([[-1,1],[1,3],[3,4]])[:,::-1]
    d=np.random.random((8,6))
    c = SimplexSolver(np.array([8,6,4,5,6,7,7]),d,np.array([2,4,3,2,1,6,5,7]))
    print c.solve()
    """

    
    