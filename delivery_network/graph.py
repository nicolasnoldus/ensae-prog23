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
       ancetres = self.bfs(src, dest, power) 
       parcours = []
       a = dest 
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
        actu = self.get_path_with_power(src, dest, self.max_power)
        if actu is None or dest not in actu:
            return None, None

        while debut != fin:
            mid = ((debut+fin)//2)
            actu = self.get_path_with_power(src, dest, power=mid)
            if actu is not None and dest in actu:
                fin = mid
            else:
                debut = mid
            if fin-debut == 1:
                break
        minus = fin
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
    g: Graph
        An object of the class Graph with the graph from file_name.
    """
    with open(filename, "r") as file:
        n, m = map(int, file.readline().split())
        g = Graph(range(1, n+1))
        for _ in range(m):
            edge = list(map(int, file.readline().split()))
            if len(edge) == 3:
                node1, node2, power_min = edge
                g.add_edge(node1, node2, power_min) # will add dist=1 by default
            elif len(edge) == 4:
                node1, node2, power_min, dist = edge
                g.add_edge(node1, node2, power_min, dist)
            else:
                raise Exception("Format incorrect")
    return g


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
    #Step n° 2: running the usual min_power on the generated MST
    path, power = MST.min_power(src, dest)
    return path, power


def LCA(src, dest, ancetres):
    """
    Find the lowest common ancestor (LCA) of two nodes in a graph, and returns the path that goes through it 
    (None if not in the graph)
    Args:
        src (int): The ID of the first node.
        dest (int): The ID of the second node.
        ancetres (dict): A dictionary that maps the ID of each node to the ID of its parent node.
    """
    if src not in ancetres or dest not in ancetres:
        return None
    
    route_src = [src]
    route_dest = [dest]
    visited_src = set(route_src)
    visited_dest = set(route_dest)

    while True:
        if ancetres.get(src) is not None and ancetres[src] not in visited_src:
            route_src.append(ancetres[src])
            visited_src.add(ancetres[src])
            src = ancetres[src]
        else:
            break
            
    while True:
        if ancetres.get(dest) is not None and ancetres[dest] not in visited_dest:
            route_dest.append(ancetres[dest])
            visited_dest.add(ancetres[dest])
            dest = ancetres[dest]
        else:
            break
            
    route_dest.reverse()
    path = route_src + route_dest[1:]
    return path

def min_power_LCA(graph, src, dest):
    """
    New version of the min_power function, 
    Gives the path with the minimum power between two given nodes
    Two twists bring complexity down and time performance up:
    - preprocessing with the kruskal algorithm
    - lowest common ancestor (LCA) search instead of DFS to find paths before power-sorting them
    This should allow to bring complexity down to O(|log(V)|)
    """
    # 1. Preprocessing
    g = kruskal(graph)
    # 2. LCA

    # 3. power-sorting

    return path, power


def knapsacked_trucks(filename):
    """
    A main function with embedded driver code and initialisation
    to run the recursive knapsack function below on our graph file
    Returns the trucks bought, the associated trajects, and the total profit
    """
    # 1. Initialisation
    g = graph_from_file(filename)
    Budget = 25*10**9
    trucks = open("trucks.x.in", "r")
    nb_trucks = int(trucks.readline().strip())
    paths_cost_profit = []
    # 2. setting up our matrix with truck costs and profits for each path
    # 2.1. associate each truck power with a cost
    truck_powers = []
    truck_costs = []
    for i, line in enumerate(trucks):
        if i == 0:
            continue  # Skip the first line
        power, cost = map(int, line.split()[:2])
        truck_powers.append(power)
        truck_costs.append(cost)
    # 2.2. find the minimum power for each traject
    routes = read_routes_file("routes.x.in")
    min_powers = []
    for path in routes:
        n1, n2, amount = map(int, path.split())
        min_power = g.min_power_LCA(n1, n2)
        min_powers.append(min_power)
    # 2.3. associate each traject with a cost and a profit
    for i, path in enumerate(routes):
        for j, power in enumerate(truck_powers):
            if power >= min_powers[i]:
                cost = truck_costs[j]
                profit = amount - cost
                paths_cost_profit.append((i, j, cost, profit))
    # now we have a list of paths with associated profits and costs, on which we can run our knapsack algorithm
    # 3. Run the recursive knapsack function below and keep the associated trucks and paths
    allocation, total_profit = knapsack(g, paths_cost_profit, Budget, len(paths_cost_profit))
    trucks_and_paths = []
    for i, j in allocation:
        path = routes[i]
        truck = (truck_powers[j], truck_costs[j])
        trucks_and_paths.append((truck, path))
    return trucks_and_paths, total_profit


def knapsack(g, paths_cost_profit, Budget, N):
    """
    (Optimized) recursive knapsack method applied to our truck allocation problem
    Computes all profits associated to all sets of allocations and gives the global maximum
    We use dynamic programming (DP) as a main resource to reduce complexity:
    Complexity = O(|Number of paths * Budget|)
    Auxiliary space = O(|Budget|)
    Args:
        g (graph): our initialized graph
        paths_cost_profit (list): list of tuples with path cost, profit and associated truck power
        Budget (int): 25*10**9
        N (int): our recursive index for the knapsack algorithm
    """
    # initialisation: DP vector with |Budget+1|-columns:
    DP = [0 for x in range(Budget + 1)]
    allocation = [[None for x in range(Budget + 1)] for y in range(N + 1)]
    
    for i in range(N + 1):
        for w in range(Budget, 0, -1):
            # we only need to consider the case where the cost is lower than the leftover budget
            if paths_cost_profit[i-1][0] <= w:
                # check if adding the path to the knapsack is profitable
                if DP[w - paths_cost_profit[i-1][0]] + paths_cost_profit[i-1][1] > DP[w]:
                    DP[w] = DP[w - paths_cost_profit[i-1][0]] + paths_cost_profit[i-1][1]
                    allocation[i][w] = (i-1, w - paths_cost_profit[i-1][0])
            # keep the previous value of DP[w]
            else:
                allocation[i][w] = allocation[i-1][w]
    
    # retrieve the paths and trucks in the allocation
    trucks_and_paths = []
    i = N
    w = Budget
    while i > 0 and w > 0:
        # if the allocation at (i,w) is not the same as at (i-1,w)
        if allocation[i][w] != allocation[i-1][w]:
            truck_idx = paths_cost_profit[i-1][2]
            path_idx = i-1
            trucks_and_paths.append((truck_idx, path_idx))
            w -= paths_cost_profit[i-1][0]
        i -= 1
        
    return trucks_and_paths, DP[Budget]


def greedy_approach(input_graph, routesfile, ):
        """
    Idea: start by sorting the paths by profit and then go one by one
    This relies heavily on our min_power_LCA earlier on
    ***
    Limits in comparison with a global max : 
    Possibly the last truck + the leftover budget would have been better spent 
    by saturating the budget completely on less expensive trucks
    See below trials to fix this with a simulated-annealing inspired approach
    *** 
        """
        paths_and_trucks = []
        Budget = 25*10**9
        # Step n° 1: create a dict with path, min_power, truck, profit
        X = {}
        for n1, n2 in routes:
            truck with truck_power >= min_power_LCA(input_graph, n1, n2, power)
            cost(path) = cost(truck(path))
            profit = gain(path) - cost(path)
            X.append([path, min_power, truck, profit]) #find a better way than append to reduce time? new class? 
        # Step n° 2: sort by profit (descending)
        g.profit = sorted(X, key= lambda, item: item[profit], reverse = True)
        # Step n° 3: saturate budget
        while Budget > 0:
            for i in range(len(g.profit)):
                while Budget - cost > 0
                paths_and_trucks.append()
                Budget = Budget - cost()
        # We now have a list of trucks and associated paths, sorted by profit
        return paths_and_trucks    



def expected_profit(graph, paths_and_trucks):
    """
    Same allocation problem with two extra difficulties:
    1° probability for a path to "break"
    2° fuel cost 
    This first approach considers the paths produced by our algorithms higher up and only computes the expected value under 1° and 2°
    Problem: the actual maximisation doesn't consider 1° and 2°, and this first draft does not reconsider the paths choices
    """
    # 0. Initialising the given values, epsilon, fuel cost
    eps = 0.001
    fuel_cost = 0.001
    expected_profit = 0
    # 1. determine for paths in the earlier optimum the associated number of edges, distances, min_power, and truck to cover it
    for n1, n2 in  :
        amount, truck_cost = 
        distance = 
        number_of_edges = 
    # 2. compute the expected profit of one traject
        expected_val = amount*(1-eps)**number_of_edges - fuel_cost*distance - truck_cost
    # 3. the total expected profit will be the sum for all paths in our optimum
        expected_profit += expected_val
    return expected_profit


def swaps(graph, allocation, alpha=0.99, stopping_T=1e-8, stopping_iter=100):
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
    # NB: MAIN PROBLEM: one path could be swapped for several! Idea: add real simulated annealing and reallocate budget
    for leftover_path in leftover_budget:
    # for a limited amount of iterations and "temperature" (reset for each path)
        T = 1.0
        i = 0
        while T > stopping_T and i < stopping_iter:
            for path in allocation:
                new_allocation = start_allocation - path + leftover_path
                if expected_profit(new_allocation)>expected_profit(allocation):
                    allocation = new_allocation
    return allocation
                

        
def simulated_annealing(graph, allocation, alpha=0.99, stopping_T=1e-8, stopping_iter=100):
    """
    In this second approach, instead of starting with our constraint of minimized truck cost using min_power,
    we use a more usual simulated annealing in the sense that swaps are random: we consider the same source and destination nodes than in our initial allocation,
    but swap randomly the path that leads from source to destination
    This allows for a comparison: do we get the same results for optimizing with a constraint of minimal truck cost
    and for letting truck cost flucuate and thus minimizing more the importance of "breaks" in the expected value?
    """
    p_break = 
    fuel_cost = 
    # initialize solution
    curr_allocation = allocation
    curr_profit = get_allocation_profit(graph, curr_allocation, p_break, fuel_cost)
    best_allocation = curr_allocation
    best_profit = curr_profit
    
    # iterate until stopping condition
    i = 0
    while T > stopping_T and i < stopping_iter:
        # randomly perturb solution
        truck = random.choice(range(len(allocation)))
        new_path = min_power_LCA(graph, curr_allocation[truck][0], curr_allocation[truck][1])
        new_allocation = curr_allocation.copy()
        new_allocation[truck] = new_path
        
        # accept perturbation if it improves solution or with certain probability if it worsens solution
        new_profit = get_allocation_profit(graph, new_allocation, p_break, fuel_cost)
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



















