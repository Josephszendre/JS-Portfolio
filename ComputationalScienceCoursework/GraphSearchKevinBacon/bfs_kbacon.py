# bfs_kbacon.py
"""Volume 2A: Breadth-First Search (Kevin Bacon).
<Name>
<Class>
<Date>
"""

from collections import deque
import pdb
import networkx as nx

# Problems 1-4: Implement the following class
class Graph(object):
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a list of the
    corresponding node's neighbors.

    Attributes:
        dictionary: the adjacency list of the graph.
    """

    def __init__(self, adjacency):
        """Store the adjacency dictionary as a class attribute."""
        self.dictionary = adjacency

    # Problem 1
    def __str__(self):
        """String representation: a sorted view of the adjacency dictionary.

        Example:
            >>> test = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
            >>> print(Graph(test))
            A: B
            B: A; C
            C: B
        """
        _str=""
        
        for s in sorted(self.dictionary.keys()):
            _str+= s+": "+"; ".join(self.dictionary[s]) + "\n"

        return _str

    # Problem 2
    def traverse(self, start):
        """Begin at 'start' and perform a breadth-first search until all
        nodes in the graph have been visited. Return a list of values,
        in the order that they were visited.

        Inputs:
            start: the node to start the search at.

        Returns:
            the list of visited nodes (in order of visitation).

        Raises:
            ValueError: if 'start' is not in the adjacency dictionary.

        Example:
            >>> test = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
            >>> Graph(test).traverse('B')
            ['B', 'A', 'C']
        """
        
        
        marked=set()
        visited=list()
        visit_queue=deque()

        #Visit first node
        visited.append(start)
        marked.add(start)
        current=start

        for s in self.dictionary[start]:
            visit_queue.append(s)
            marked.add(s)

        stop=len(visit_queue)==0
        while not stop: #len(visit_queue)!=0:
            current=visit_queue.popleft()

            visited.append(current)
            for s in self.dictionary[current]:
                if s not in marked:
                    visit_queue.append(s)
                    marked.add(s)

            stop=len(visit_queue)==0
            
        return visited


    # Problem 3 (Optional)
    def DFS(self, start):
        """Begin at 'start' and perform a depth-first search until all
        nodes in the graph have been visited. Return a list of values,
        in the order that they were visited. If 'start' is not in the
        adjacency dictionary, raise a ValueError.

        Inputs:
            start: the node to start the search at.

        Returns:
            the list of visited nodes (in order of visitation)
        """
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def shortest_path(self, start, target):
        """Begin at the node containing 'start' and perform a breadth-first
        search until the node containing 'target' is found. Return a list
        containg the shortest path from 'start' to 'target'. If either of
        the inputs are not in the adjacency graph, raise a ValueError.

        Inputs:
            start: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from start to target,
                including the endpoints.

        Example:
            >>> test = {'A':['B', 'F'], 'B':['A', 'C'], 'C':['B', 'D'],
            ...         'D':['C', 'E'], 'E':['D', 'F'], 'F':['A', 'E', 'G'],
            ...         'G':['A', 'F']}
            >>> Graph(test).shortest_path('A', 'G')
            ['A', 'F', 'G']
        """
        
        def _getpath(lst):
            """
            Build l backwards
            
            """

            l=[]
            l.append(target)
            curr=target  
            for i in xrange(len(lst)-1,-1	,-1):
                a=lst[i]

                if a[1]==start:
                    l.append(start)
                    return l[::-1]
                elif a[0]==curr:
                    l.append(a[1])
                    curr=a[1]
            raise ValueError("Incorrect Array Inputed")            
        
        marked=set()
        visited=list()
        visit_queue=deque()
        
        
        #Visit first node
        visited.append(start)
        marked.add(start)
        current=start
        
        if not (start in self.dictionary and target in self.dictionary):
            raise ValueError("Both start and target nodes must be contained in the graph")
        
        if target in self.dictionary[start]:
            return "['"+start+"', '"+target+"]" 
        
        for s in self.dictionary[start]:
            visit_queue.append(s+current)
            marked.add(s)
            
        prev=current
        stop=len(visit_queue)==0
        while not stop: #len(visit_queue)!=0:
            current=visit_queue.popleft()
            prev=current[1]
            current=current[0]
            
            visited.append(current+prev)
            for s in self.dictionary[current]:
                if s == target:
                    visited.append(s+current)
                    return _getpath(visited)
                    
                elif s not in marked:
                    visit_queue.append(s+current)
                    marked.add(s)

            stop=len(visit_queue)==0
        
        raise ValueError("The graph is not connected, start and target are in two disconnected trees")
        

# Problem 5: Write the following function
def convert_to_networkx(dictionary):
    """Convert 'dictionary' to a networkX object and return it."""
    nx_graph=nx.Graph()
    a=[]
    for l in dictionary.keys():
         for j in dictionary[l]:
              a.append((l,j))    
    
    nx_graph.add_edges_from(a)
    return nx_graph                 

# Helper function for problem 6
def parse(filename="movieData.txt"):
    """Generate an adjacency dictionary where each key is
    a movie and each value is a list of actors in the movie.
    """

    # open the file, read it in, and split the text by '\n'
    with open(filename, 'r') as movieFile:
        moviesList = movieFile.read().split('\n')
    graph = dict()

    # for each movie in the file,
    for movie in moviesList:
        # get movie name and list of actors
        names = movie.split('/')
        title = names[0]
        graph[title] = []
        # add the actors to the dictionary
        for actor in names[1:]:
            graph[title].append(actor)

    return graph


# Problems 6-8: Implement the following class
class BaconSolver(object):
    """Class for solving the Kevin Bacon problem."""

    # Problem 6
    def __init__(self, filename="movieData.txt"):
        """Initialize the networkX graph and with data from the specified
        file. Store the graph as a class attribute. Also store the collection
        of actors in the file as an attribute.
        """
        a=parse(filename)
        self.graph=convert_to_networkx(a)
        self.graph.add_node("asdfasdfasdfsadf")
        b=a.items()
        c=[]
        for (d,e) in b:
            for f in e:
                c.append(f)

        c=list(set(c))
        self.actors=c

        
    # Problem 6
    def path_to_bacon(self, start, target="Bacon, Kevin"):
        """Find the shortest path from 'start' to 'target'."""
        return nx.shortest_path(self.graph,start,target)

    # Problem 7
    def bacon_number(self, start, target="Bacon, Kevin"):
        """Return the Bacon number of 'start'."""
        return (1+len(nx.shortest_path(self.graph,start,target)))/2

        

    # Problem 7
    def average_bacon(self, target="Bacon, Kevin"):
        """Calculate the average Bacon number in the data set.
        Note that actors are not guaranteed to be connected to the target.

        Inputs:
            target (str): the node to search the graph for
        """
        l=[]
        n=0
        for i in self.actors:
            try:
                path=nx.shortest_path(self.graph,i,target)
                l+=[(len(path)+1)/2]
            except Exception:
                n+=1
        return sum(l)*1.0/(len(l)*1.0)
            

if __name__ == "__main__":
    #print G.traverse("A")
    test = {'A':['B', 'F','I'], 'B':['A', 'C'], 'C':['B', 'D'],
         'D':['C', 'E'], 'E':['D', 'F'], 'F':['A', 'E', 'G'],
         'G':['A', 'F','H'], 'H':['A'],'I':['H']}
    G=Graph(test)
    print G.shortest_path("A","H")
    print G.traverse("A")

    
    b=BaconSolver()
    print b.path_to_bacon("Jackson, Samuel L.")
    print b.bacon_number("Jackson, Samuel L.")
    print b.average_bacon()

# =========================== END OF FILE =============================== #
