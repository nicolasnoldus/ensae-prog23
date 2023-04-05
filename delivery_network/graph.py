import math
import random

class Graph:
    """
    A class representing graphs as adjacency lists and implementing various algorithms on the graphs. Graphs in the class are not oriented. 
    Attributes: 
    -----------
    nodes: NodeType
        A list of nodes. Nodes can be of any immutable type, e.g., integer, float, or string.
        We will usually use a list of integers 1, ..., n.
    graph: dict
        A dictionnary that contains the adjacency list of each node in the form
        graph[node] = [(neighbor1, p1, d1), (neighbor1, p1, d1), ...]
        where p1 is the minimal power on the edge (node, neighbor1) and d1 is the distance on the edge
    nb_nodes: int
        The number of nodes.
    nb_edges: int
        The number of edges. 
    """

    def __init__(self, nodes=[]):
        """
        Initializes the graph with a set of nodes, and no edges. 
        Parameters: 
        -----------
        nodes: list, optional
            A list of nodes. Default is empty.
        """
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
        self.list_of_neighbours = []
        self.list_of_edges = []
        self.max_power = 0

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output
    
    def add_edge(self, node1, node2, power_min, dist=1):
        """
        Adds an edge to the graph. Graphs are not oriented, hence an edge is added to the adjacency list of both end nodes. 
        Parameters: 
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        power_min: numeric (int or float)
            Minimum power on this edge
        dist: numeric (int or float), optional
            Distance between node1 and node2 on the edge. Default is 1.
        """
        if node1 not in self.graph:
            self.graph[node1] = []
            self.nb_nodes += 1
            self.nodes.append(node1)
        if node2 not in self.graph:
            self.graph[node2] = []
            self.nb_nodes += 1
            self.nodes.append(node2)

        self.graph[node1].append((node2, power_min, dist))
        self.graph[node2].append((node1, power_min, dist))
        self.nb_edges += 1
        self.list_of_edges.append((node1,node2,power_min))
    

    def get_path_with_power(self, src, dest, power):
        ancetres = self.bfs(src, dest, power) #ancetres est encore le dicitonnaire qui a comme clé un noeud et comme valeur 
        #celui par lequel on a pu parvenir à ce noeud.
        parcours = []
        #on crée une liste parcours qui nous donnera le parcours entre les deux noeuds choisis en respectant toujours la puissance 
        #exigée.
        a = dest #on renomme dest en a pour faciliter la suite.
        if a not in ancetres:
            return None
            #s'il n'y a pas de graphe connexe avec la puissance exigée, on retourne none.
        while a != src:
            #On met cette condition car on part de l'arrivée de notre bfs, donc on remonte point par point jusqu'à arriver à 
            #notre point de départ.
            parcours.append(a)
            a = ancetres[a]
            #on rajoute le sommet a à notre liste de noeuds parcourus et on définit "le nouveau" a comme étant son ancêtre pour
            #rebrousser chemin.
        parcours.append(src)
        #on doit ajouter l'origine a la main parce que notre boucle while s'arrête dès lors que a prend la valeur du noeud de départ
        #et ne le rajoute donc pas dans la liste.
        parcours.reverse()
        return parcours


    def dfs(self, node, visites, composantes, power=1000000000):  
    #on a rajouté une condition de puissance afin de pouvoir conditionner les chemins possible dans la question 
    #traitant de la puissance minimale à avoir pour un trajet. On met une valeur très importante par défaut pour 
    #éviter qu'il n'ampute des trajectoires possibles sur des graphes
        visites.append(node) 
    #on prend un point à partir duquel on veut commencer notre dfs et on l'ajoute à la liste 
    #visites qui conservera tous les noeuds déjà visités
        composantes.append(node)
    #la liste composantes n'est ici pas indispensable mais sera utile pour la question qui suit 
        for i in self.graph[node]: 
            if i[0] not in visites and power >= i[1]: 
                self.dfs(i[0], visites, composantes, power=power) 
                #ici est la recursion de la fonction. On lui demande de s'appliquer elle-même à chaque noeud qui n'est pas 
                #encore présent dans la liste "visites". On rajoute également une condition sur la puissance lorsque cela est nécessaire. 
        return visites
                

    def connected_components(self) :
        visites = []
        gde_liste = []
        #on crée une grande liste qui sera une liste de liste de tous les noeuds reliés. C'es-à-dire que chaque liste 
        #présente dans la liste soit un graphe connexe avec tous les noeuds qu'elle contient. 
        for i in self.graph:
            if i not in visites:
                composantes=[]
                #on a donc ici la liste de compostantes qui se reset à chaque itération tandis que la liste de visites
                #reste inchangé. 
                #Comme on parcours chaque noeud et que visites garde en mémoire tous les noeuds qui ont été visités,
                #on a donc une nouvelle liste qui se crée qu'à condition qu'il n'y ait aucuun noeud dans une précédente liste.
                self.dfs(i, visites, composantes)
                gde_liste.append(composantes)
        return gde_liste

              
    def connected_components_set(self):
        return set(map(frozenset, self.connected_components()))


    def bfs(self, beg, dest, power=float('inf')):
       ancetres = dict()
       #le dictionnaire ancetres est le dictonnaire qui permet d'avoir le lien entre chaque sommet, c'est-à-dire que la clé est le
       #sommet en question et sa valeur est le noeud par lequel on est arrivés.
       queue = []
       visited = set()
       #on fait un set pour les noeuds visités pour éviter d'avoir des boucles étant donné que le set ne gardera
       #qu'une fois chaque noeud.
       queue.append(beg)
       while len(queue) > 0:
           n = queue.pop()
           #le while est conditionné par la longueur de la queue du fait de l'utilisation de pop. Comme on a une queue on supprime le
           #dernier élément de cette liste pour chercher les autres sommets.
          
           for v in self.graph[n]:
               #print(v)
               if (type(v)==tuple) is True :
                   if v[0] not in visited and power >= v[1]:
                   #on garde la condition dans les visites pour ne pas faire de boucle et on rajoute celle sur la puissance pour coller
                   #aux conditions de base. De la sorte, on considère qu'il n'y a pas d'arêtes si la puissance de celle-ci
                   #est supérieure à la puissance donnée comme paramètre.
                       queue.append(v[0])
                   #on rajoute tous les voisins du noeud en question à la liste de queue pour avoir tous les chemins
                       ancetres[v[0]] = n
                   #on définit la valeur comme le noeur à partir duquel on est arrivés.
                       visited.add(v[0])
                   #et on le rajoute au set des visites comme pour éviter les boucles.
               else :
                   pass   
      
       return ancetres


    def min_power(self, src, dest):
       debut = 1
       fin = self.max_power
       actu=self.get_path_with_power(src, dest, self.max_power)
       if actu is None or dest not in actu:
           return None, None
       #si les deux noeuds en question ne sont pas sur un graphe connexe, on retourne none car il n'y a pas de chemins possible.
       while debut != fin:
           #on fait une recherche binaire.
           #Pour être tout à fait honnête, la condition sur le while est un peu désuète étant donné qu'on fait un break
           #avant que cette condition puisse se remplir mais c'est la solution qui a le mieux marché sur plusieurs tests :
           #network.1 et network.2, lentement mais sûrement.
           mid = ((debut+fin)//2)
           actu=self.get_path_with_power(src, dest, power=mid)
           #on actualise à chaque itération le graphe des sommets formant un graphe connexe et permettant un chemin.
           if actu is not None and dest in actu:
               fin = mid
           #si le sommet qu'on veut atteindre est dans le graphe fait à partir de la médiane des puissances
           #on redéfinit la "borne sup" comme étant l'ancien milieu pour retrécir notre champ de recherche.
           else:
               debut=mid
           #on procède pareillement mais avec la plus petite puissance dans le cas contraire.
           if fin-debut == 1 :
               break
           #Comme on ne prend pas comme valeurs de power les puissances présentes dans le graphe mais simplement
           #les entiers situés entre la plus grande puissance et la plus petite,
           #la condition pour sortir de la boucle while est que la différence entre les deux extrêmes soit égale à un.
           #Ainsi, cela signifierait que ce sont deux entiers qui se suivent et on doit donc nécessairement prendre
           #"fin" car début serait trop petit.
       minus=fin
       return self.get_path_with_power(src, dest, minus), minus


def graph_from_file(filename):
   """
   Reads a text file and returns the graph as an object of the Graph class.
   The file should have the following format:
       The first line of the file is 'n m'
       The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
       The nodes (node1, node2) should be named 1..n
       All values are integers.
   Parameters:
   -----------
   filename: str
       The name of the file
   Outputs:
   -----------
   G: Graph
       An object of the class Graph with the graph from file_name.
   """
   #start = time.perf_counter()
   file = open(filename, 'r')
   dist=1
   #First line is read in order to properly intialize our graph
   line_1 = file.readline().split(' ')
   total_nodes = int(line_1[0])
   nb_edges = int(line_1[1].strip('\n'))
   new_graph = Graph([node for node in range(1,total_nodes+1)])
   #Then, all lines are read to create a new edge for each line
   for line in file:
       list_line = line.replace("\n","").split(' ')
       start_node = int(list_line[0])
       end_node = int(list_line[1])
       power = int(list_line[2])
       if list_line == []:
           continue
       if len(list_line) == 4:
           #In the case where a distance is included
           dist = float(list_line[3])
       new_graph.max_power = max(new_graph.max_power, power)
       new_graph.add_edge(start_node, end_node, power, dist)
   new_graph.list_of_neighbours = [list(zip(*new_graph.graph[node]))[0] for node in new_graph.nodes if new_graph.graph[node]!=[]]
   #stop = time.perf_counter()
   #print(stop-start)
   file.close()
   return new_graph


class Union_Find():
    """
    A class for union and find operations for later use
    Using union&find as attributes proves to be useful to avoid errors (e.g. index problems)
    """

    def __init__(self):
        self.subtree_size = -1
        self.parent = self

    def set_up(self):
        self.subtree_size = 0
    
# A find function to get to the set a node belongs to
    def find(self):
        while self != self.parent:
            self = self.parent
        return self

# A function that merges two sets of x and y,
# in this case the sets being connected components of nodes    
# we filter by subtree size for efficiency
    def union(self, node_2):
        x = self.find()
        y = node_2.find()    
        if x == y :
            return 
        if x.subtree_size > y.subtree_size:
            y.parent = x
            x.subtree_size += y.subtree_size
        else:
            x.parent = y
            y.subtree_size += x.subtree_size
            if x.subtree_size == y.subtree_size:
                y.subtree_size += 1


def kruskal(input_graph):
    """
    Gives the minimum spanning tree (MST) of an input graph using Kruskal's algorithm
    We use the union-find method to detect cycles as suggested in S. Dasgupta et al. (2006)
    Path compression allows to bring complexity down to O(|V|): 
    See below time-complexity comparisons with BFS/DFS
    (This algorithm works adequately on one graph at a time)
    """
    MST = Graph()
    MST.nb_edges = input_graph.nb_nodes - 1
    # Sorting edges in a nondecreasing order of their power: 
    # the spanning tree produced by iteration will then necessarily be a MST
    input_graph.graph = sorted(input_graph.list_of_edges, key=lambda item: item[2])
    # we use an index (p) to go through these edges in an increasing order of power
    p = 0
    nodes = {}
    for node in input_graph.nodes:
        nodes[node] = Union_Find()
        nodes[node].set_up()
    # When our MST in progress will have |V|-1 edges, it will be complete (see above, Q. 11)
    e = 0
    while e < len(input_graph.nodes)-1 and p < len(input_graph.graph):
        # we consider the edge with the smallest power each time
        n1, n2, power = input_graph.graph[p]
        p = p+1
        # if adding the edge doesn't create a cycle, we add it to our MST in progress
        if nodes[n1].find() != nodes[n2].find():
            MST.add_edge(n1, n2, power)
            e = e+1
            # and we take into account that the nodes are now connected
            nodes[n1].union(nodes[n2])
    return MST


def min_power_kruskal(input_graph, src, dest):
    """
    New version of the min_power function, 
    Gives the path with the minimum power between two given nodes
    A twist to bring complexity down and time performance up:
    - preprocessing with the kruskal algorithm
    The complexity is then lowered to O(|V|)
    """
    # Step n° 1: Preprocessing
    MST = kruskal(input_graph)
    MST.max_power = 99999
    #Step n° 2: running the usual min_power on the generated MST
    path, power = MST.min_power(src, dest)
    return path, power


def LCA(src, dest, ancetres) :
   route_src=[]
   route_dest=[]
   a=src
   b=dest
   visited_src=set()
   visited_dest=set()
   visited_dest.add(b)
   visited_src.add(a)
   if a not in ancetres or b not in ancetres :
       return None
   else :
       while b not in visited_src or a not in visited_dest :


       #while b not in visited_src or a not in visited_dest : #on doit rajouter cette condition car ce n'est pas un arbre oriente
       #donc il est possible qu'il fasse des cycles dans ses allers-retours entre noeud d'où il vient et noeud où il va
           if ancetres[a] in visited_src :
               pass
           elif ancetres[b] in visited_dest :            
               pass
           else :
               route_src.append(a)
               route_dest.append(b)
           visited_src.add(a)
           visited_dest.add(b)
           a=ancetres[a]
           b=ancetres[b]
           #print("routea :", route_src)
           #print("routeb :",route_dest)
           print("vA :", visited_src)
           print("vB :", visited_dest)
      
       for i in visited_dest :
           if i not in route_src+route_dest :
               route_src.append(i)
           else :
               pass
       for i in visited_src :
           if i not in route_src+route_dest :
               route_src.append(i)
           else :
               pass
      
       #while route_dest[-1]!=route_src[-1]:
       for i in visited_dest :
           if i not in visited_src and i not in route_src + route_dest :
               route_src.append(i)
       for i in visited_src :
           if i not in visited_dest and i not in route_src + route_dest :
               route_src.append(i)
       route_dest.reverse()
       #route_src=route_src[:-1]
       trajet_total = route_src + route_dest
       #trajet_total.pop(0)
       #trajet_total.pop(len(trajet_total)-1)
       return trajet_total
       #sur network 10, en 7 secondes entre les noeuds : 9 et 14778.
       #en comptant le temps de la mise en place du kruskal
       #en 8 secondes entre 1 et 10000
        

def min_power_LCA(g, src, dest):
    """
    New version of the min_power function, 
    Gives the path with the minimum power between two given nodes
    Two twists bring complexity down and time performance up:
    - preprocessing with the kruskal algorithm
    - lowest common ancestor (LCA) search instead of DFS to find paths before power-sorting them
    This should allow to bring complexity down to O(|log(V)|)
    """
    # 1. Preprocessing
    g = kruskal(g)
    # 2. LCA
    b = kruskal(g).bfs(src, dest)
    path = LCA(src, dest, b)
    # 3. we have our path, now we have to retrieve power
    power = 0
    for i in range(len(path)-1):
        src, dest = path[i], path[i+1]
        if src-1 < len(g.list_of_neighbours) and dest in g.list_of_neighbours[src-1]:
            print("ntm")
            dest_index = g.list_of_neighbours[src-1].index(dest)
            curr_power = g.graph[src][dest_index][1]
            if current_power > power:
                power = curr_power
    return power, path


def knapsack(filename):
    """
    A main function with embedded driver code and initialisation
    to run the recursive knapsack function below on our graph file
    Returns the trucks bought, the associated trajects, and the total profit in the following format:
    1° index of the path in the routes.x.in file
    2° index of the truck in our path_cost_profit vector
    3° total profit
    (we can then easily retrieve each one but this form is more compact, given that the list is long)
    """
    # 1. Initialisation
    g = graph_from_file(filename)
    Budget = 25*10**3
    trucks_file = open("/home/onyxia/work/ensae-prog23-2/input/trucks.1.in", "r")
    paths_cost_profit = []
    # 2. setting up our matrix with truck costs and profits for each path
    # 2.1. associate each truck power with a cost
    trucks = []
    for i, line in enumerate(trucks_file):
        if i == 0:
            continue  # Skip the first line
        power, cost = map(int, line.split()[:2])
        trucks.append([power, cost])
    # sort the trucks by power in ascending order
    trucks.sort(key=lambda x: x[0])
    
    # 2.2. find the minimum power for each traject 
    routes = open("/home/onyxia/work/ensae-prog23-2/input/routes.1.in", "r")
    min_powers = []
    for i, path in enumerate(routes):
        if i == 0:
            continue # Skip the first line
        n1, n2, amount = map(int, path.split())
        min_power = 100000*min_power_kruskal(g,n1, n2)[1]
        # iterate over the sorted trucks and find the first one that has enough power
        for j in range(len(trucks)):
            if trucks[j][0] >= min_power:
                cost = int(trucks[j][1]*10**(-4))
                profit = 1000*amount//cost
                paths_cost_profit.append((cost, profit))
                break
    # now we have a list of paths with associated profits and costs, on which we can run our knapsack algorithm
    # 3. Run the recursive knapsack function below and keep the associated trucks and paths
    return dynamic_programming(g, paths_cost_profit, Budget)


def dynamic_programming(g, paths_cost_profit, Budget):
    """
    Optimized dynamic programming method applied to our truck allocation problem
    Computes all profits associated to all sets of allocations and gives the global maximum
    We use dynamic programming (DP) as a main resource to reduce complexity:
    Complexity = O(|Number of paths * Budget|)
    Auxiliary space = O(|Budget|)
    Args:
        g (graph): our initialized graph
        paths_cost_profit (list): list of tuples with path cost, profit and associated truck power
        Budget (int): 25*10**9
    Returns:
        A tuple of the form (trucks_and_paths_list, max_profit), where:
        - trucks_and_paths_list is a list of tuples, where each tuple represents an allocation of a truck to a path,
          and has the form (truck_index, path_index).
        - max_profit is the maximum profit that can be obtained with the given budget.
    """
    # Create a dictionary that maps each cost to the maximum profit that can be obtained with that cost.
    # We will use this dictionary to avoid recomputing profits for the same cost.
    max_profits = {}
    
    # Create a list to keep track of the trucks and paths in each allocation.
    trucks_and_paths_list = []
    
    # Initialize DP vector with |Budget+1| columns:
    DP = [0 for _ in range(Budget + 1)]
    
    # Initialize allocation matrix with None values:
    allocation = [[None for _ in range(Budget + 1)] for _ in range(len(paths_cost_profit) + 1)]
    
    # Iterate over the paths, starting with the last one:
    for i in range(len(paths_cost_profit) - 1, -1, -1):
        # Get the path's cost and profit:
        path_cost, path_profit = paths_cost_profit[i]
        
        # Iterate over the possible costs:
        for w in range(Budget, path_cost - 1, -1):
            # Compute the profit of the current allocation:
            allocation_profit = DP[w - path_cost] + path_profit
            
            # Check if the current allocation is more profitable than the previous one:
            if allocation_profit > DP[w]:
                DP[w] = allocation_profit
                allocation[i][w] = (i, w - path_cost)
            else:
                allocation[i][w] = allocation[i + 1][w]
        
        # Retrieve the trucks and paths in the allocation:
        trucks_and_paths = []
        w = Budget
        for j in range(i, len(paths_cost_profit)):
            if allocation[j][w] != allocation[j + 1][w]:
                truck_idx = paths_cost_profit[j][1]
                path_idx = j
                trucks_and_paths.append((truck_idx, path_idx))
                w -= paths_cost_profit[j][0]
        
        # Reverse the list so that it is in the correct order:
        trucks_and_paths.reverse()
        trucks_and_paths_list.append(trucks_and_paths)
    
    # Return the trucks and paths list and the maximum profit:
    return trucks_and_paths_list[::-1], DP[Budget]


def greedy_approach(filename, routesfile, trucksfile):
    """
    The function idea is simple: we sort the paths by profit and then add them one by one
    It ends up being more interesting than a fractional knapsack
    ***
    Limits in comparison with a global max : 
    Possibly the last truck + the leftover budget would have been better spent 
    by saturating the budget completely on less expensive trucks
    See below attempts to fix this with a simulated-annealing inspired approach
    *** 
    returns a list of tuples of the form (truck_cost, profit),..., total_profit)
    """
    # 0. Initialisation
    g = graph_from_file(filename)
    Budget = 25*10**3
    trucks_file = open(trucksfile, "r")
    routes = open(routesfile, "r")
    paths_cost_profit = []
    trucks_and_paths = []
    # 1. find the profit for each path by substracting the truck cost to the amount made
    trucks = []
    for i, line in enumerate(trucks_file):
        if i == 0:
            continue  # Skip the first line
        power, cost = map(int, line.split()[:2])
        trucks.append([power, cost])
    # sort the trucks by power in ascending order
    trucks.sort(key=lambda x: x[0])
    for i, path in enumerate(routes):
        if i == 0:
            continue # Skip the first line
        n1, n2, amount = map(int, path.split())
        min_power = 100000*min_power_kruskal(g,n1, n2)[1]
        # iterate over the sorted trucks and find the first one that has enough power
        for j in range(len(trucks)):
            if trucks[j][0] >= min_power:
                cost = int(trucks[j][1]*10**(-4))
                profit = 1000*amount - cost
                paths_cost_profit.append((cost, profit))
                break
    # 2. sort paths by profit (descending)
    paths_cost_profit.sort(key=lambda x: x[1], reverse = True)
    # 3. add paths until budget is saturated
    for i in range(len(paths_cost_profit)):
        if Budget >= paths_cost_profit[i][0]:
            trucks_and_paths.append(paths_cost_profit[i])
            Budget = Budget - paths_cost_profit[i][0]
    return trucks_and_paths, sum(paths_cost_profit[i][1] for i in range(len(paths_cost_profit)))


def expected_profit(filename, routesfile, trucksfile, eps=0.001, fuel_cost=0.001):
    """
    Same allocation problem with two extra difficulties:
    1° probability for a path to "break"
    2° fuel cost 
    This first approach only computes the expected profit without maximizing on probability
    """
    expected_profit = 0
    trucks_and_paths = greedy_approach(filename, routesfile, trucksfile)[0]
    
    # 1. determine for paths in the earlier optimum the associated number of edges, distances, min_power, and truck to cover it
    for truck, path in trucks_and_paths:
        truck_cost = truck
        for line in enumerate(filename):
            if [path[0], path[1]] == list(map(int, line.split()[:2])):
                distance = int(line.split()[3])
        for line in enumerate(routesfile):
            if map(int, line.split()[:2]) == (path[0], path[1]):
                amount = int(line.split()[2])
        number_of_edges = len(path) - 1
        # 2. compute the expected profit of one traject
        expected_val = amount*(1-eps)**number_of_edges - fuel_cost*distance - truck_cost
        # 3. the total expected profit will be the sum for all paths in our optimum
        expected_profit += expected_val
    return expected_profit


def swaps(graph, allocation, leftover_budget, alpha=0.99, stopping_T=1e-8, stopping_iter=100):
    """
    A simulated annealing application combined with local search for our allocation problem with breaking probability
    Instead of computing all possible paths, etc., we change one and see whether the expected profit is increased
    This algorithm can get stuck in a local optimum: to check this we can change our initial starting point 
    (our starting point is set by default on our previous optimal allocation with a knapsack/greedy approach,
    because intuitively it might be close to our global optimum on expected value, 
    but we could build a function to give random allocations to compare the different outputs of this function)
    The approach here is to start with our min_power to maximize the profit, and then use simulated annealing for
    1° maximizing with given breaking probability
    2° maximizing our earlier greedy approach, compensating its local bias
    """
    p_break = 0.001
    
    # we try to swap each path (always considering only those with minimal power, i.e. one path between two nodes)
    for path, truck in allocation:
        # check if path is not already in allocation
        if path in leftover_budget:
            # for a limited amount of iterations and "temperature" (reset for each path)
            T = 1.0
            i = 0
            while T > stopping_T and i < stopping_iter:
                # randomly perturb solution
                new_path = random.choice(leftover_budget)
                new_allocation = allocation.copy()
                new_allocation[truck] = (new_path, truck)
                
                # accept perturbation if it improves solution or with certain probability if it worsens solution
                new_profit = expected_profit(graph, new_allocation)
                delta = new_profit - expected_profit(graph, allocation)
                if delta > 0 or math.exp(delta/T) > random.random():
                    allocation = new_allocation
                # decrease temperature
                T *= alpha
                i += 1
    return allocation
                
        
def simulated_annealing(graph, allocation, p_break=0.001, fuel_cost=0.001, alpha=0.99, stopping_T=1e-8, stopping_iter=100):
    """
    In this second approach, instead of starting with our constraint of minimized truck cost using min_power,
    we use a more usual simulated annealing in the sense that swaps are random: we consider the same source and destination nodes than in our initial allocation,
    but swap randomly the path that leads from source to destination
    This allows for a comparison: do we get the same results for optimizing with a constraint of minimal truck cost
    and for letting truck cost fluctuate and thus minimizing more the importance of "breaks" in the expected value?
    """
    # initialize solution
    curr_allocation = allocation
    curr_profit = expected_profit(graph, curr_allocation, p_break, fuel_cost)
    best_allocation = curr_allocation
    best_profit = curr_profit
    
    # iterate until stopping condition
    T = 1.0
    i = 0
    while T > stopping_T and i < stopping_iter:
        # randomly perturb solution
        truck = random.choice(range(len(allocation)))
        new_path = min_power_LCA(graph, curr_allocation[truck][0], curr_allocation[truck][1])
        new_allocation = curr_allocation.copy()
        new_allocation[truck] = new_path
        
        # accept perturbation if it improves solution or with certain probability if it worsens solution
        new_profit = expected_profit(graph, new_allocation, p_break, fuel_cost)
        delta = new_profit - curr_profit
        if delta > 0 or math.exp(delta/T) > random.random():
            curr_allocation = new_allocation
            curr_profit = new_profit
        if curr_profit > best_profit:
            best_allocation = curr_allocation
            best_profit = curr_profit
        
        # decrease temperature
        T *= alpha
        i += 1
        
    return best_allocation, best_profit



















