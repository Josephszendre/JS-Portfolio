# linked_lists.py
"""Volume II Lab 4: Data Structures 1 (Linked Lists)
<Name>
<Class>
<Date>
"""


# Problem 1
from matplotlib import pyplot as plt
import numpy as np

class Node(object):
    """A basic node class for storing data.
    The data type must be an int, str or float to work
    """
    def __init__(self, data):
        """Store 'data' in the 'value' attribute.
        data must be an int, float or str or a
        ValueError exception will be raised.
        """
        if isinstance(data,int) or isinstance(data,float) or isinstance(data,str):
            self.value = data
        else:
            raise TypeError("Uncomparable")
            
        


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the 'Node' class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store 'data' in the 'value' attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None
        self.prev = None


class LinkedList(object):
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the 'head' and 'tail' attributes by setting
        them to 'None', since the list is empty initially.
        """
        self.head = None
        self.tail = None
        self.leng = 0

    def append(self, data):
        """Append a new node containing 'data' to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.leng==0:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
            self.leng=1
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
            self.leng+=1

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing 'data'.
        If no such node exists, raise a ValueError.

        Examples:
            >>> l = LinkedList()
            >>> for i in [1,3,5,7,9]:
            ...     l.append(i)
            ...
            >>> node = l.find(5)
            >>> node.value
            5
            >>> l.find(10)
            ValueError: <message>
        """
        found=False
        current=self.head
        if self.head is None:
            raise ValueError("The List is empty yo.")
            
        while not found:
            if current.value == data:
                found = True
            else:
                if current.next is None:
                    raise ValueError("No node with " + str(data) + " object")            
                else:
                    current=current.next
        return current


    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in [1,3,5]:
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        return self.leng

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()   |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:   |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)    |   ...     l2.append(i)
            ...                     |   ...
            >>> print(l1)           |   >>> print(l2)
            [1, 3, 5]               |   ['a', 'b', 'c']
        """
        ret_str="["
        current=self.head
        complete=False
        buff=""
        if isinstance(self.head.value, str):
            buff="'"
        while not complete:
            ret_str+=buff+str(current.value)+buff
            
            if not isinstance(current.next, Node):
                ret_str+="]"
                complete=True
            else:
                ret_str+=", "
                current=current.next
        return ret_str
            
        

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing 'data'. Return nothing.

        Raises:
            ValueError: if the list is empty, or does not contain 'data'.

        Examples:
            >>> print(l1)       |   >>> print(l2)
            [1, 3, 5, 7, 9]     |   [2, 4, 6, 8]
            >>> l1.remove(5)    |   >>> l2.remove(10)
            >>> l1.remove(1)    |   ValueError: <message>
            >>> l1.remove(9)    |   >>> l3 = LinkedList()
            >>> print(l1)       |   >>> l3.remove(10)
            [3, 7]              |   ValueError: <message>
        """
        target = self.find(data)
        if target.value==self.head.value:
            if self.head.next is None:
                self.head=None
                self.tail=None
            else:
                self.head=self.head.next
                self.head.next.prev=None
        elif target.next is None:
            self.tail=target.prev
            self.tail.next=None
        else:
            #Not root and not tail
            target.prev.next=target.next
            target.next.prev=target.prev
        self.leng-=1

    # Problem 5
    def insert(self, data, place):
        """Insert a node containing 'data' immediately before the first node
        in the list containing 'place'. Return nothing.

        Raises:
            ValueError: if the list is empty, or does not contain 'place'.

        Examples:
            >>> print(l1)           |   >>> print(l1)
            [1, 3, 7]               |   [1, 3, 5, 7, 7]
            >>> l1.insert(7,7)      |   >>> l1.insert(3, 2)
            >>> print(l1)           |   ValueError: <message>
            [1, 3, 7, 7]            |
            >>> l1.insert(5,7)      |   >>> l2 = LinkedList()
            >>> print(l1)           |   >>> l2.insert(10,10)
            [1, 3, 5, 7, 7]         |   ValueError: <message>
        """
        print "insert, " + str(data) + " before " + str(place)
        neighbor=self.find(place)
        node=LinkedListNode(data)
        if not isinstance(neighbor.prev, Node):
            neighbor.prev=node
            self.head=node
            self.head.next=neighbor
        else:
            neighbor.prev.next=node
            node.next=neighbor
            node.prev=neighbor.prev
            neighbor.prev=node
        
        self.leng+=1


# Problem 6: Write a Deque class.
class Deque(LinkedList):
    def __init__(self):
        LinkedList.__init__(self)
    
    def popleft(self):
        if self.leng==0:
            return None
        elif self.leng==1:
            data=self.head.value
            self.head=None
            self.tail=None
            self.leng-=1
            return data
        else:
            data=self.head.value
            self.head.next.prev=None
            self.head=self.head.next
            self.leng-=1
            return data
    
    def pop(self):
        if self.leng==0:
                                    return None
        elif self.leng==1:
            data=self.head.value
            self.head=None
            self.tail=None
            self.leng=0
            return data
        else:
            data=self.tail.value
            self.tail.prev.next=None
            self.tail=self.tail.prev
            self.leng-=1
            return data
            
    
    def appendleft(self,data):
        head=self.head
        node=LinkedListNode(data)
        if self.leng!=0:
            head.prev=node
            node.next=head
            self.head=node
            self.leng+=1
        else:
            self.head=LinkedListNode(data)
            self.tail=self.head
            self.leng+=1

    def remove(*args, **kwargs):
        raise NotImplementedError("Use pop() or popleft() for removal")

    def insert(*args, **kwargs):
        raise NotImplementedError("Use pop() or popleft() for removal")

# Problem 7
def prob7(infile, outfile):
    """Reverse the file 'infile' by line and write the results to 'outfile'."""
    d=Deque()
    file=open(infile, "r")
    lines=file.readlines()
    for s in lines:
        d.append(s)

    print d
    
    newfile=open(outfile, "w")

    l=d.pop()

    while isinstance(l,str):
        newfile.write(l)
        l=d.pop()
    newfile.close()
    file.close()

    
if __name__ == "__main__":
    prob7('./asdf.txt', 'asdf2.txt')
    l = LinkedList()
    

    print ""
    a=["ASDF", "QWERTY", "ZXCV"]
    for i in a:
        print "l.append('" + i + "')"
        l.append(i)
    print "\nProblem 3: LinkedList.__str__() works with strings"
    print l, "\n"        

    print "Problem 2: find(self, node) works with strings"
    for i in a:
        print "l.find('" + i + "') is " + l.find(i).value
        print l.find(i).value
         
    a=np.unique(np.random.randint(0,1000,(100)))
    random.shuffle(a)
    
    print "Construct integer linked list from", a
    for i in a:
        print "l.append('" + i + "')"
        l.append(i)
        
    print "\nProblem 3: LinkedList.__str__() works with strings"
    print l, "\n"        

    print "Problem 2: find(self, node) works with strings"
    for i in a:
        print "l.find('" + i + "') is " + l.find(i).value
        print l.find(i).value
         
    print l
    
    
    
    s=LinkedList()
    s.append("asdf")
