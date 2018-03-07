# trees.py
"""Volume 2A: Data Structures II (Trees).
<Name>
<Class>
<Date>
"""
from matplotlib import pyplot as plt
import random
import numpy as np
import pdb
import time

class SinglyLinkedListNode(object):
    """Simple singly linked list node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList(object):
    """A very simple singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None
    def append(self, data):
        """Add a Node containing 'data' to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

def iterative_search(linkedlist, data):
    """Search 'linkedlist' iteratively for a node containing 'data'.
    If there is no such node in the list, or if the list is empty,
    raise a ValueError.

    Inputs:
        linkedlist (SinglyLinkedList): a linked list.
        data: the data to search for in the list.

    Returns:
        The node in 'linkedlist' containing 'data'.
    """
    current=linkedlist.head
    while current is not None:
        if current.value == data:
            return current
        else:
            current = current.next

# Problem 1
def recursive_search(linkedlist, data):
    """Search 'linkedlist' recursively for a node containing 'data'.
    If there is no such node in the list, or if the list is empty,
    raise a ValueError.

    Inputs:
        linkedlist (SinglyLinkedList): a linked list object.
        data: the data to search for in the list.

    Returns:
        The node in 'linkedlist' containing 'data'.
    """
    def _step(current_node, data):
        if current_node.value == data:
            return current_node
        else:
            if current_node.next is None:
                raise ValueError("Node is not contained in the tree.")
            else:
                return _step(current_node.next, data)


    return _step(linkedlist.head,data)

class BSTNode(object):
    """A Node class for Binary Search Trees. Contains some data, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the data attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST(object):
    """Binary Search Tree data structure class.
    The 'root' attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing 'data'. If there is no such node
        in the tree, or if the tree is empty, raise a ValueError.
        """

 
        if self.root is None:
            raise ValueError("The tree is empty")
        else:
            found=False
            current=self.root
            target=data
                
            while not found:
                if current.value==target:
                    found=True
                elif current.left is None:
                    if current.right is None:
                        raise ValueError("Such a node doth exist not.")
                    else:
                        current=current.right
                elif current.right is None:
                    current=current.left
                elif current.left is not None and current.right is not None:
                    if target<current.value:
                        current=current.left
                    else:
                        current=current.right
                
        
        # Start the recursion on the root of the tree.
        return current

    # Problem 2
    def insert(self, data):
        """Insert a new node containing 'data' at the appropriate location.
        Do not allow for duplicates in the tree: if there is already a node
        containing 'data' in the tree, raise a ValueError.

        Example:
            >>> b = BST()       |   >>> b.insert(1)     |       (4)
            >>> b.insert(4)     |   >>> print(b)        |       / \
            >>> b.insert(3)     |   [4]                 |     (3) (6)
            >>> b.insert(6)     |   [3, 6]              |     /   / \
            >>> b.insert(5)     |   [1, 5, 7]           |   (1) (5) (7)
            >>> b.insert(7)     |   [8]                 |             \
            >>> b.insert(8)     |                       |             (8)
            """

        #Case 1: The tree is empty
        #Case 2: Otherwise, pass in root into _step
        #pdb.set_trace()
        if self.root is None:
            self.root=BSTNode(data)
        else:
            found=False
            currentnode=self.root
            while found is False:                
                if currentnode.value==data:
                    raise ValueError("The node already exists homeboy")
                elif currentnode.value<data:                    
                    if currentnode.right is None:
                        currentnode.right=BSTNode(data)
                        currentnode.right.prev=currentnode
                        currentnode.right.value=data
                        found=True
                    else:
                        currentnode=currentnode.right
                elif currentnode.value>data:
                    if currentnode.left is None:
                        currentnode.left=BSTNode(data)
                        currentnode.left.prev=currentnode
                        currentnode.left.value=data
                        found=True
                    else:
                        currentnode=currentnode.left
            
        
    # Problem 3
    def remove(self, data):
        """Remove the node containing 'data'. Consider several cases:
            1. The tree is empty
            2. The target is the root:
                a. The root is a leaf node, hence the only node in the tree
                b. The root has one child
                c. The root has two children
            3. The target is not the root:
                a. The target is a leaf node
                b. The target has one child
                c. The target has two children
            If the tree is empty, or if there is no node containing 'data',
            raise a ValueError.

        Examples:
            >>> print(b)        |   >>> b.remove(1)     |   [5]
            [4]                 |   >>> b.remove(7)     |   [3, 8]
            [3, 6]              |   >>> b.remove(6)     |
            [1, 5, 7]           |   >>> b.remove(4)     |
            [8]                 |   >>> print(b)        |
        """
        
        def _stepleft(the_node):
            if the_node.left is None:
                return the_node
            else:
                return _stepleft(the_node.left)
            
        
        node=self.find(data)
        
        
        
        """
        Pseudocode
        
        if (node is root)
          if (root has no children)
            set root=None
          else if (node has one child)
            attach child to parent
          else if (root has two parents)
            find right least successor
            set root value equal to least successor
            if least successor has right child
              atachright child to least successor parent on left
            else
              least successor parent.left = None
        else
        
        
        """
        
        
        if node==self.root:
            if node.left is not None and node.right is not None:
                least=_stepleft(node.right)
                node.value=least.value
                if node.right.value==least.value:
                    #no left successors, need only connect node and least.right if it exists
                    if least.right is not None:
                        #exist right successors
                        node.right=least.right
                        least.right.prev=node
                    else:
                        #no right successors
                        node.right=None
                        least.prev=None
                else:
                    #least is a left node with no left children (with possible right child)   
                    if least.right is None:
                        least.prev.left=None
                        least.prev=None
                    else:
                        least.prev.left=least.right
                        least.right.prev=least.prev
            elif node.left is None and node.right is not None:
                self.root=node.right
                node.right.prev=None
                node.right=None
                node.left=None
            elif node.left is not None and node.right is None:
                self.root=node.left
                node.left.prev=None
                node.left=None
            elif node.left is None and node.right is None:
                self.root=None       
        #No subnodes
        elif node.left is None and node.right is None:
            #pdb.set_trace()
            print node.value
            if node.prev.right is not None:
                
                if node.prev.right.value == node.value:
                    node.prev.right=None
                    #node.prev=None
            if node.prev.left is not None:
                if node.prev.left.value==node.value:
                    node.prev.left=None
                    #node.prev=None
        #Only right subnode
        elif node.left is None and node.right is not None:
            #Right subtree
            if node.prev.right is not None:
                if node.prev.right.value==node.value:
                    node.prev.right=node.right
                    node.right.prev=node.prev    
            #Left subtree
            if node.prev.left is not None:
                if node.prev.left.value==node.value:
                    node.prev.left=node.right
                    node.right.prev=node.prev
                    
        #Only left subnode                
        elif node.left is not None and node.right is None:
            #Right subtree
            if node.prev.right is not None:
                if node.prev.right.value==node.value:
                    node.prev.right=node.left
                    node.left.prev=node.prev    
            #Left subtree
            if node.prev.left is not None:
                if node.prev.left.value==node.value:
                    node.prev.left=node.left
                    node.left.prev=node.prev
        #Two subnodes (
        elif node.left is not None and node.right is not None:
            least=_stepleft(node.right)
            node.value=least.value
            if least.value == node.right.value:
                #Right subnode w/ possible children
                if least.right is not None:
                    least.right.prev=least.prev
                    node.right=least.right
                else:
                     node.right=None
              
            else:
                #Left subnode w/ possible children
																if least.right is None:
																    #No children
																				least.prev.left=None
																				least.prev=None
																else:
																    #right child node
																				least.prev.left=least.right
																				least.right.prev=least.prev
                        
                        
        else:
            raise Exception("Runtime Error")
        
        
        
    def check_tree(self,node):
        if node is not None:
            if node.left is not None:
                if node.left.prev is None:
                    raise Exception("Runtime error")
            if node.right is not None:
                if node.right.prev is None:
                    raise Exception("Runtime error")
        check_tree(node.left)
        check_tree(node.right)
     
    def __str__(self):
        """String representation: a hierarchical view of the BST.
        Do not modify this method, but use it often to test this class.
        (this method uses a depth-first search; can you explain how?)

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed out
                (2) (5)    [2, 5]       by depth levels. The edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """

        if self.root is None:
            return "[]"
        str_tree = [list() for i in xrange(_height(self.root) + 1)]
        visited = set()

        def _visit(current, depth):
            """Add the data contained in 'current' to its proper depth level
            list and mark as visited. Continue recusively until all nodes have
            been visited.
            """
            str_tree[depth].append(current.value)
            visited.add(current)
            if current.left and current.left not in visited:
                _visit(current.left, depth+1)
            if current.right and current.right not in visited:
                _visit(current.right, depth+1)

        _visit(self.root, 0)
        out = ""
        for level in str_tree:
            if level != list():
                out += str(level) + "\n"
            else:
                break
        return out



class AVL(BST):
    """AVL Binary Search Tree data structure class. Inherits from the BST
    class. Includes methods for rebalancing upon insertion. If your
    BST.insert() method works correctly, this class will work correctly.
    Do not modify.
    """
    def _checkBalance(self, n):
        return abs(_height(n.left) - _height(n.right)) >= 2

    def _rotateLeftLeft(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n == self.root:
            self.root = temp
        return temp

    def _rotateRightRight(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n == self.root:
            self.root = temp
        return temp

    def _rotateLeftRight(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotateLeftLeft(n)

    def _rotateRightLeft(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotateRightRight(n)

    def _rebalance(self,n):
        """Rebalance the subtree starting at the node 'n'."""
        if self._checkBalance(n):
            if _height(n.left) > _height(n.right):
                if _height(n.left.left) > _height(n.left.right):
                    n = self._rotateLeftLeft(n)
                else:
                    n = self._rotateLeftRight(n)
            else:
                if _height(n.right.right) > _height(n.right.left):
                    n = self._rotateRightRight(n)
                else:
                    n = self._rotateRightLeft(n)
        return n

    def insert(self, data):
        """Insert a node containing 'data' into the tree, then rebalance."""
        BST.insert(self, data)
        n = self.find(data)
        while n:
            n = self._rebalance(n)
            n = n.prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() has been disabled for this class.")

def _height(current):
    """Calculate the height of a given node by descending recursively until
    there are no further child nodes. Return the number of children in the
    longest chain down.

    This is a helper function for the AVL class and BST.__str__().
    Abandon hope all ye who modify this function.

                                node | height
    Example:  (c)                  a | 0
              / \                  b | 1
            (b) (f)                c | 3
            /   / \                d | 1
          (a) (d) (g)              e | 0
                \                  f | 2
                (e)                g | 0
    """
    if current is None:
        return -1
    return 1 + max(_height(current.right), _height(current.left))


# Problem 4
def prob4():
    """Compare the build and search speeds of the SinglyLinkedList, BST, and
    AVL classes. For search times, use iterative_search(), BST.find(), and
    AVL.find() to search for 5 random elements in each structure. Plot the
    number of elements 0in the structure versus the build and search times.
    Use log scales if appropriate.
    """
    afile=open("./english.txt", "r")
    lines=afile.readlines()
    afile.close
    t1=[]
    t2=[]
    t3=[]
    t4=[]
    t5=[]
    t6=[]
    for i in xrange(3,11):
        rand_idx=np.unique(np.random.randint(0,len(lines),(2**i)))
        rand_subset = np.array([lines[j] for j in rand_idx])
        #rand_subset = random.shuffle(rand_subset)
        rand_five_idx=np.unique(np.random.randint(0,len(rand_subset),(5)))
        rand_five=[rand_subset[k] for k in rand_five_idx]
        
        #Load linked list
        S=SinglyLinkedList()
        #time
        t=time.time()
        for l in rand_subset:
            S.append(l)
        #record time
        t1+=[time.time()-t]
        
        #iterative_search
        t=time.time()        
        for l in rand_five:            
            iterative_search(S,l)
        t2+=[time.time()-t]
        
    
        #Load BST
        B=BST()
        t=time.time()
        for l in rand_subset:    
            B.insert(l)
        t3+=[time.time()-t]
    
        #BST.find()
        t=time.time()        
        for l in rand_five:
            B.find(l)
        t4+=[time.time()-t]
        
    
    
        #Load AVL Tree
        A=AVL()
        t=time.time()
        for l in rand_subset:
            A.insert(l)
        t5+=[time.time()-t]
        
        #BST.find() on AVL
        t=time.time()
        for l in rand_five:
            A.find(l)
        t6+=[time.time()-t]
        
        
    
    dom=2**np.arange(3,11)
    
    #print np.log(t6)/np.log(2)
    print "t1", t1
    print "t2", t2
    print "t3", t3
    print "t4", t4
    print "t5", t5
    print "t6", t6
    
    
    
    
    plt.subplot(121)
    plt.ylim(0,0.2)
    plt.xlim(7,1200)
    plt.plot(dom,t1,"b-x",lw=2,label="Singly Linked List") #basex=2, basey=2, lw=2, label="Singly Linked List")
    plt.plot(dom,t3,"g-x",lw=2,label="Binary Search Tree") #basex=2, basey=2, lw=2, label="Singly Linked List")
    plt.plot(dom,t5,"r-x",lw=2,label="AVL Tree") #basex=2, basey=2, lw=2, label="Singly Linked List")
    plt.title("Build Times", fontsize=18)

    plt.subplot(122)
    plt.ylim(0,.0009)
    plt.xlim(7,2048)
    plt.plot(dom,t2,"b-x",lw=2,label="Singly Linked List") #basex=2, basey=2, lw=2, label="Singly Linked List")
    plt.plot(dom,t4,"g-x",lw=2,label="Binary Search Tree") #basex=2, basey=2, lw=2, label="Singly Linked List")
    plt.plot(dom,t6,"r-x",lw=2,label="AVL Tree") #basex=2, basey=2, lw=2, label="Singly Linked List")
    plt.title("Search Times", fontsize=18)
    
    """
    plt.subplot(122)
    plt.ylim(-2,2)
    plt.xlim(0,2*np.pi)
    plt.loglog(x,np.sin(2*x), "r--", basex=2,basey=2, lw=2)
    plt.title("Search Times", fontsize=18)
    """
    plt.show()    

if __name__ == "__main__":
    s=SinglyLinkedList()
    s.append("a")
    s.append("b")
    #s.append
    #prob4()
    
    
    

    B=AVL()
    
    a=np.unique(np.random.randint(0,1000,(20)))
    random.shuffle(a)
    for i in a:
        B.insert(i)
        print B
        print B.find(i)
    

    random.shuffle(a)
    for i in a:
        print "Remove", i
        #B.remove(i)
								        
        print B
    
    #print B
    
    #Works for removing left leaves on left side
    #Works for removing right leaves on right side
    #Works for removing left leaves on right side
    #Works for removing right leaves on left side
    
    #Does work for subtrees with no right leaf
    #Does not work for subtrees with no left leaf
    #Does not work for left subtrees with both leafs
