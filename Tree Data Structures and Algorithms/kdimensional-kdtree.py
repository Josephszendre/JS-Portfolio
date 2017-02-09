# kdtrees.py
"""Volume 2A: Data Structures 3 (K-d Trees).
<Name>
<Class>
<Date>
"""

import numpy as np
from trees import BST, BSTNode
from scipy.spatial import KDTree
import pdb
from sklearn import neighbors

# Problem 1
def metric(x, y):
    """Return the euclidean distance between the 1-D arrays 'x' and 'y'.

    Raises:
        ValueError: if 'x' and 'y' have different lengths.

    Example:
        >>> metric([1,2],[2,2])
        1.0
        >>> metric([1,2,1],[2,2])
        ValueError: Incompatible dimensions.
    """
    if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
        return np.sqrt(sum((x-y)**2))
    else:
        raise Exception("Must input a np.ndarray yo")


# Problem 2
def exhaustive_search(data_set, target):
    """Solve the nearest neighbor search problem exhaustively.
    Check the distances between 'target' and each point in 'data_set'.
    Use the Euclidean metric to calculate distances.

    Inputs:
        data_set ((m,k) ndarray): An array of m k-dimensional points.
        target ((k,) ndarray): A k-dimensional point to compare to 'dataset'.

    Returns:
        ((k,) ndarray) the member of 'data_set' that is nearest to 'target'.
        (float) The distance from the nearest neighbor to 'target'.
    """        
    least=10e9*1.0
    for d in data_set[::]:
        m=metric(d,target)
        if m<least:
            least=m
            n=d
        print m, d
    
    return n
            
        


# Problem 3: Write a KDTNode class.
class KDTNode(BSTNode):
    def __init__(self, ndimarray):
        BSTNode.__init__(self,np.array(ndimarray))
        self.axis=-1
        self.dim=ndimarray.ndim
    
    
        

# Problem 4: Finish implementing this class by overriding
#            the __init__(), insert(), and remove() methods.
class KDT(BST):
    """A k-dimensional binary search tree object.
    Used to solve the nearest neighbor problem efficiently.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other
            nodes in the tree, the root houses data as a NumPy array.
        k (int): the dimension of the tree (the 'k' of the k-d tree).
    """
    def __init__(self):
        BST.__init__(self)
        self.k=0
        
    def remove(self,*args):
        raise NotImplementedError("You cannot remove anything once it's been entered.")

            
    def find(self, data):
        """Return the node containing 'data'. If there is no such node
        in the tree, or if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            'data' is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.axis] < current.value[current.axis]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursion on the root of the tree.
        return _step(self.root)

    def insert(self, data):
        """Insert a new node containing 'data' at the appropriate location.
        Return the new node. This method should be similar to BST.insert().
        """
        def _plusaxis(axis, k):
            if axis<k:
                return axis+1
            elif axis==k:
                return 0
                
        d=np.array(data)
        
        if self.k==0:
            self.k=d.ndim
        elif self.k!=d.ndim:
            raise ValueError("The array must be of the same length as other lengths")
        
        
        if self.root is None:
            self.root=KDTNode(data)
        else:
            #If the node already exists then raise ValueError
            try:
                self.find(data)
                raise ValueError("The data point " + str(data) + " is already in the tree")
            except Exception:
                pass

            node=self.root
            found=False
            axis=0        
            while (not found):
                
                if node.value[axis]<data[axis]:
                    if node.right is None:
                        node.right=KDTNode(data)
                        node.right.prev=node
                        node.right.axis=_plusaxis(axis,self.k)
                        found=True
                    if node.right is not None:
                        node=node.right
                else: 
                    #node.value[axis] >= data.value[axis]
                    if node.left is None:
                        node.left=KDTNode(data)
                        node.left.prev=node
                        node.left.axis=_plusaxis(axis,self.k)
                        found=True
                    else:
                        node=node.left
                
                axis=_plusaxis(axis,self.k)



# Problem 5
def nearest_neighbor(data_set, target):
    """Use your KDT class to solve the nearest neighbor problem.

    Inputs:
        data_set ((m,k) ndarray): An array of m k-dimensional points.
        target ((k,) ndarray): A k-dimensional point to compare to 'dataset'.

    Returns:
        The point in the tree that is nearest to 'target' ((k,) ndarray).
        The distance from the nearest neighbor to 'target' (float).
    """

    def KDTsearch(current, neighbor, distance):
        """The actual nearest neighbor search algorithm.

        Inputs:
            current (KDTNode): the node to examine.
            neighbor (KDTNode): the current nearest neighbor.
            distance (float): the current minimum distance.

        Returns:
            neighbor (KDTNode): The new nearest neighbor in the tree.
            distance (float): the new minimum distance.
        """
        if current is None:
            return neighbor, distance
        
        idx=current.axis
        m=metric(current.value, target)
        if m<distance:
            neighbor=current
            distance=m
        if target[idx]<current.value[idx]:
            neighbor,distance=KDTsearch(current.left,neighbor, distance)
            if target[idx]+distance>=current.value[idx]:
                neighbor,distance=KDTsearch(current.right,neighbor,distance)
        else:
            neighbor,distance=KDTsearch(current.right,neighbor, distance)
            if target[idx]-distance<=current.value[idx]:
                neighbor,distance=KDTsearch(current.left,neighbor,distance)
        
        return neighbor,distance    
            
    K=KDT()
    for d in data_set:
        K.insert(d)
        
    return KDTsearch(K.root,K.root,metric(K.root.value,target))
    

# Problem 6
def postal_problem():
    """Use the neighbors module in sklearn to classify the Postal data set
    provided in 'PostalData.npz'. Classify the testpoints with 'n_neighbors'
    as 1, 4, or 10, and with 'weights' as 'uniform' or 'distance'. For each
    trial print a report indicating how the classifier performs in terms of
    percentage of correct classifications. Which combination gives the most
    correct classifications?

    Your function should print a report similar to the following:
    n_neighbors = 1, weights = 'distance':  0.903
    n_neighbors = 1, weights =  'uniform':  0.903       (...and so on.)
    """
    
    labels, points, testlabels, testpoints = np.load('PostalData.npz').items()
    
    def _test(w,n):
        nbrs=neighbors.KNeighborsClassifier(n_neighbors=n, weights=w,p=2)    
        nbrs.fit(points[1],labels[1])
        prediction=nbrs.predict(testpoints[1])
        return np.average(prediction==testlabels[1])
    

    for i in [1,4,10]:
        print "n_neighbors =", i, ", weights = 'distance': ", _test("distance",i)
        print "n_neighbors =", i, ", weights =  'uniform': ", _test("uniform",i)
    
    
if __name__=="__main__":
    #print metric(np.array([2,2,2]),np.array([1,1,1]))
    #print exhaustive_search(np.array([[1.0,1.0,1.0],[5.0,6.9,6.9],[2,3,4],[1,2,3]])*1.0,np.array([3.0,3.0,3.0]))
    #postal_problem()
    import time

    for i in xrange(0,1):
        a=time.time()
        data=np.random.random((10000,20))
        target=np.random.random(20)
        tree=KDTree(data)
        nearest,distance=nearest_neighbor(data,target)
        print nearest.value,"\n",distance, "time ==", time.time()-a
        
        tree=KDTree(data)
        min_distance,index=tree.query(target)
        print metric(data[index],target)-distance
