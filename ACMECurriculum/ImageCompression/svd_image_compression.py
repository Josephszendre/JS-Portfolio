# svd_image_compression.py
"""Volume 1A: SVD and Image Compression.
<Name>
<Class>
<Date>
"""

from scipy import linalg as la
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import linalg as LAsp

# Problem 1
def truncated_svd(A,k=None):
    """Computes the truncated SVD of A. If r is None or equals the number
        of nonzero singular values, it is the compact SVD.
    Parameters:
        A: the matrix
        k: the number of singular values to use
    Returns:
        U - the matrix U in the SVD
        s - the diagonals of Sigma in the SVD
        Vh - the matrix V^H in the SVD
    """
    
    
    
    AHA=np.conj(A).T.dot(A)
    evals,evecs=la.eig(AHA)
    order=np.argsort(evals)

    evals=evals[order][::-1].copy()
    evecs=evecs.T[order][::-1].copy()
    m,n=AHA.shape
    
    tol=1e-12
    Vh=[]
    for i in xrange(0,m):
		      if np.abs(evals[i])>=tol:
	         		Vh+=[evecs[i]]
    
    Vh=np.array(Vh)
    s=np.sqrt(evals[:Vh.shape[0]])
    U=[]
    for i in xrange(0,len(s)):
        U+=[(1./s[i])*A.dot(Vh[i])]
    U=np.array(U).T
    
    return U,s,Vh
    
		
		

# Problem 2
def visualize_svd():
    """Plot each transformation associated with the SVD of A."""
    A=np.array([[3,1],[1,3]])
    U,s,Vh=truncated_svd(A)
    
    twopi=np.linspace(0,2.*np.pi,360)
    one=np.reshape(np.linspace(0,1,100),(1,100))
    zeros=np.zeros((1,100))
    S=np.vstack((np.reshape(np.cos(twopi),(1,360)),np.reshape(np.sin(twopi),(1,360))))
    e1=np.vstack((zeros,one))
    e2=e1[::-1] 	
    
    s1S=Vh.dot(S)
    s1e1=Vh.dot(e1)
    s1e2=Vh.dot(e2)

    s2S=np.diag(s).dot(s1S)
    s2e1=np.diag(s).dot(s1e1)
    s2e2=np.diag(s).dot(s1e2)
    
    s3S=U.dot(s2S)
    s3e1=U.dot(s2e1)
    s3e2=U.dot(s2e2)
    
    
    
    

    
    
    plt.subplot(221)
    plt.plot(S[0],s3S[1],"b-.",lw=2)
    plt.plot(e1[0],s3e1[1],"g-.",lw=2)
    plt.plot(e2[0],s3e2[1],"r-.",lw=2)
    
    
    
    plt.subplot(222)
    plt.plot(s1S[0],s3S[1],"b-.",lw=2)
    plt.plot(s1e1[0],s3e1[1],"g-.",lw=2)
    plt.plot(s1e2[0],s3e2[1],"r-.",lw=2)
    
    
    plt.subplot(223)
    plt.plot(s2S[0],s3S[1],"b-.",lw=2)
    plt.plot(s2e1[0],s3e1[1],"g-.",lw=2)
    plt.plot(s2e2[0],s3e2[1],"r-.",lw=2)
    
    plt.subplot(224)   
    
    plt.plot(s3S[0],s3S[1],"b-.",lw=2)
    plt.plot(s3e1[0],s3e1[1],"g-.",lw=2)
    plt.plot(s3e2[0],s3e2[1],"r-.",lw=2)
     
    plt.show()

# Problem 3
def svd_approx(A, k):
    """Returns best rank k approximation to A with respect to the induced 2-norm.

    Inputs:
    A - np.ndarray of size mxn
    k - rank

    Return:
    Ahat - the best rank k approximation
    """
    U,s,Vh=la.svd(A,full_matrices=False)
    return U[:,:k].dot(np.diag(s[:k])).dot(Vh[:k,:]) 
    

# Problem 4
def lowest_rank_approx(A,e):
    """Returns the lowest rank approximation of A with error less than e
    with respect to the induced 2-norm.

    Inputs:
    A - np.ndarray of size mxn
    e - error

    Return:
    Ahat - the lowest rank approximation of A with error less than e.
    """
    
    
    U,s,Vh=la.svd(A,full_matrices=False)
    t=s.copy()
    t[t>e]=0
    i=t.nonzero()[0][0]
    
    return U[:,:i].dot(np.diag(s[:i])).dot(Vh[:i,:])
    
    
	
    
    

# Problem 5
def compress_image(filename,k):
    """Plot the original image found at 'filename' and the rank k approximation
    of the image found at 'filename.'

    filename - jpg image file path
    k - rank
    """
    img_color=plt.imread(filename)
    orig=img_color.copy()
    R=img_color[:,:,0]
    G=img_color[:,:,1]
    B=img_color[:,:,2]
    
    m,n=(R.shape[0],R.shape[1])
    u1,s1,vh1=la.svd(R,full_matrices=False)
    u2,s2,vh2=la.svd(G,full_matrices=False)
    u3,s3,vh3=la.svd(B,full_matrices=False)
    img_color[:,:,0]=u1[:,:k].dot(np.diag(s1[:k]).dot(vh1[:k,:]))
    img_color[:,:,1]=u2[:,:k].dot(np.diag(s2[:k]).dot(vh2[:k,:]))
    img_color[:,:,2]=u3[:,:k].dot(np.diag(s3[:k]).dot(vh3[:k,:]))
    plt.subplot(211)
    img_color[img_color>1]=1.
    img_color[img_color<0]=0.
    plt.imshow(img_color)
    plt.subplot(212)
    plt.imshow(orig)
    plt.show()
    print G
    
    
   
    
    
    
    
def getImage(filename='dream.png'):    
    '''
    # Helper function used to convert the image into the correct format.
    def getImage(filename='dream.png'):
    
    Reads an image and converts the image to a 2-D array of brightness
    values.

    Inputs:
        filename (str): filename of the image to be transformed.
    Returns:
        img_color (array): the image in array form
        img_brightness (array): the image array converted to an array of
            brightness values.
    '''
    img_color = plt.imread(filename)
    img_brightness = (img_color[:,:,0]+img_color[:,:,1]+img_color[:,:,2])/3.0

    return img_color,img_brightness
    
    

if __name__=="__main__":
    visualize_svd()
    

	#visualize_svd()
	#S=np.random.random((3,3))
    #a,b,c=truncated_svd(S)
	#print a.dot(np.diag(b)).dot(c)-S
	#np.reshape(np.arange(9),(3,3)))
    #a=np.arange(0,9)
    #b=a/100.+a[::-1]*1j
    #print b[np.argsort(b)]
    #c=[]
    #for i in xrange(0,100000):
    #    A=np.random.random((5,5))
    #    B=lowest_rank_approx(A,.2)
    #    c+=[la.norm(A-B,ord=2)]
    #print max(c)
    a,b=getImage()
    #a,b,c=truncated_svd(5)
    #print a.dot(b.dot(c))
    #print a

