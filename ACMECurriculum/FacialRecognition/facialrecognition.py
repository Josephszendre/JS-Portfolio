
# coding: utf-8

# In[ ]:

import numpy as np
from scipy import linalg as la
from os import walk
from scipy.ndimage import imread
from matplotlib import pyplot as plt

# In[ ]:

def plot(img, w=200, h=180):
    """Helper function for plotting flattened images."""
    plt.imshow(img.reshape((w,h)))
    plt.show()

# Problem 1
def getFaces(path="./faces94"):
    """Traverse the directory specified by 'path' and return an array containing one column vector per subdirectory.
    For the faces94 dataset, this gives an array with just one column for each face in the dataset.
    Each column corresponds to a flattened grayscale image.
    """
    # Traverse the directory and get one image per subdirectory.
    faces = []
    for (dirpath, dirnames, filenames) in walk(path):
        for f in filenames:
            if f[-3:]=="jpg":          # only get jpg images
                 # Load the image, convert it to grayscale, and flatten it into vector.
                faces.append(imread(dirpath+"/"+f).mean(axis=2).ravel())
                break
    # Put all face vectors column-wise into a matrix.
    return np.transpose(faces)



# Problems 2, 3, 4, 5
class FacialRec:
    """
    Attributes:
        F
        mu
        Fbar
        U
    """
    def __init__(self,path):
        self.initFaces(path)
        self.initMeanImage()
        self.initDifferences()
        self.initEigenfaces()
    def initFaces(self, path):
        self.faces=getFaces(path)
    def initMeanImage(self):
        self.mu=faces.mean(axis=1)
    def initDifferences(self):
        self.Fbar=self.faces-np.reshape(self.mu, (len(self.mu),1))
        plot(Fbar[:,0])
    def initEigenfaces(self):
        self.U,s,Vh=la.svd(Fbar, full_matrices=False)
        plot(U[:,0])
    def project(self, A, s=38):
        return U[:,:s].T.dot(A)
    def findNearest(self, image, s=38):
        img=image.flatten()
        g=np.reshape(U[:,:s].T.dot(img-self.mu),(s,1))
        F=U[:,:s].T.dot(Fbar)
        diff=(F-g).T
        idx=np.argmin([la.norm(diff[i],ord=2) for i in xrange(diff.shape[1])])
        return idx
    
        
        
        


# In[ ]:

# Problem 6
def findNearest(self, image, s=38):
    """Project Fbar, producing a matrix whose columns are f-hat"""
    # Fhat =
    """Shift 'image' by the mean and project, producing g-hat"""
    # ghat =
    """For both Fhat and ghat, use your project function from the previous problem.
    Return the index that minimizes ||fhat_i - ghat||_2."""
    
if __name__=="__main__":
    a=getFaces()
    print a.shape

