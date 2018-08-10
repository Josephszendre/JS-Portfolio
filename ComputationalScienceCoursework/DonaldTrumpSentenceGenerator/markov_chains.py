# markov_chains.py
"""Volume II: Markov Chains.
<Name>
<Class>
<Date>
"""

import numpy as np
import numpy.random
import numpy.linalg as la



# Problem 1
def random_markov(n):
    """Create and return a transition matrix for a random Markov chain with
    'n' states. This should be stored as an nxn NumPy array.
    """
    a=np.random.random((n,n))
    b=a.T/a.sum(axis=1)
    return b


# Problem 2
def forecast(days):
    """Forecast tomorrow's weather given that today is hot."""
    transition = np.array([[0.7, 0.6], [0.3, 0.4]])
    a=[[1.],[0.]]
    f=[]
    for i in range(0,days):
         a=np.dot(transition,a)
         print(a)
         f+=[np.random.binomial(1, transition[1, 0])]

    return f


# Problem 3
def four_state_forecast(days):
    """Run a simulation for the weather over the specified number of days,
    with mild as the starting state, using the four-state Markov chain.
    Return a list containing the day-by-day results, not including the
    starting day.

    Examples:
        >>> four_state_forecast(3)
        [0, 1, 3]
        >>> four_state_forecast(5)
        [2, 1, 2, 1, 1]
    """
    transition=np.array([[0.5, 0.3, 0.1, 0.],
    [.3, 0.3, 0.3, 0.3],
    [0.2, 0.3, 0.4, 0.5],
    [0., 0.1, 0.2, 0.2]])
    transition=list(transition.T)
    x=1
    f=[]
    for i in xrange(0,days):
         x=np.random.multinomial(1,transition[x]).nonzero()[0][0]
         f+=[x]

    return f
    
    

# Problem 4
def steady_state(A, tol=1e-12, N=40):
    """Compute the steady state of the transition matrix A.

    Inputs:
        A ((n,n) ndarray): A column-stochastic transition matrix.
        tol (float): The convergence tolerance.
        N (int): The maximum number of iterations to compute.

    Raises:
        ValueError: if the iteration does not converge within N steps.

    Returns:
        x ((n,) ndarray): The steady state distribution vector of A.
    """
    x2=np.random.random((2,))
    #x1=np.zeros((2,))
    for i in range(0,N):
        x1=x2.copy()
        x2=np.dot(A,x1)
        if la.norm(x2-x1,ord=1)<tol:
            return x2          
    raise ValueError("The sequence did not converge in N steps.")
        


# Problems 5 and 6
class SentenceGenerator(object):
    """Markov chain creator for simulating bad English.

    Attributes:
        (what attributes do you need to keep track of?)

    Example:
        >>> yoda = SentenceGenerator("Yoda.txt")
        >>> print yoda.babble()
        The dark side of loss is a path as one with you.
    """

    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        
        uniq_words=set()
        
        thefile=open(filename,"r")
        read=thefile.read()
        sentences=read.split("\n")
        words=[]
        for s in sentences:
            a=s.split(" ")
            words+=[a]
            for w in a:
                uniq_words.add(w)  
                
        
        ct=len(uniq_words)        
        
        uniq_words=list(uniq_words)
        states=["$tart"]+uniq_words + ["$top"]
        tran=np.zeros((len(uniq_words)+2,len(uniq_words)+2))

        for j in range(len(words)):
            for k in range(len(words[j])):
                word=words[j][k]
                
                if k==len(words[j])-1:
                    tran[ct+1,uniq_words.index(word)+1]+=1
                elif k==0:
                    tran[uniq_words.index(word)+1,0]+=1
                    next=words[j][k+1]
                    tran[uniq_words.index(next)+1,uniq_words.index(word)+1]+=1

                else:
                    next=words[j][k+1]
                    tran[uniq_words.index(next)+1,uniq_words.index(word)+1]+=1

        
        tran[len(uniq_words)+1,len(uniq_words)+1]+=1
        a=tran.sum(axis=0)
        self.tran=(tran/a)


        self.uniq_words=uniq_words    
        thefile.close()
    

    def babble(self):
        """Begin at the start sate and use the strategy from
        four_state_forecast() to transition through the Markov chain.
        Keep track of the path through the chain and the corresponding words.
        When the stop state is reached, stop transitioning and terminate the
        sentence. Return the resulting sentence as a single string.
        """
        go=True
        a=0
        words=[]
        i=0

        while go:
            a=np.random.multinomial(1,self.tran[:,a]).nonzero()[0][0]
            if a==len(self.uniq_words)+1:
                #end reached
                	
                return " ".join(words)
            else:
                words+=[self.uniq_words[a-1]] 
            i+=1
            if i>1000:
                go=False

            

if __name__=="__main__":
    S=SentenceGenerator("DTSentences.txt")
    for j in range(0,100):
        print (S.babble())
