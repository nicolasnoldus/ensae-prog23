from graph import Graph, Union_Find, graph_from_file, kruskal, min_power_kruskal_V1# min_power, get_path_with_power, dfs, connected_components, bfs, connected_components_set 
import os
import time
import random

# See below different rudimentary time estimation functions
# They estimate only the actual min_power functions processing time (not e.g. of writing routes.out, etc.)

def time_perf_min_power(file) :
   import time
   start=time.time()
  
   road=open(f"input/routes.{file}.in", "r")
   g = graph_from_file(f"input/network.{file}.in")
   gk = kruskal(g)
   new=open(f"input/route.{file}.out", "w")
   line_1=road.readline().split(' ')
   nb_routes=int(line_1[0])
   new.write(f"le nombre total de trajets est :, {nb_routes} \n")
   for line in road :
       list_line=line.split(" ")
       src=int(list_line[0])
       dest=int(list_line[1])
       new.write(f'{gk.min_power(src, dest)[1]} \n')
   road.close()
   new.close()
   end=time.time()
  
   duree = end - start
   print(duree)
    
    
def performance_estimation_min_power(file, routes):
    g = graph_from_file(file)
    individual_performances = []
    paths = open(routes, "r")
    nb_paths = int(paths.readline().strip())
    for i, line in enumerate(paths):
        if i == 0:
            continue  # Skip the first line
        if i > 30:
            break  # Stop reading after 30 lines
        n1, n2 = map(int, line.split()[:2])
        start = time.perf_counter()
        g.min_power(n1, n2)
        stop = time.perf_counter()
        individual_performances.append(stop - start)
    estimation = sum(individual_performances)
    print("Temps estimé:", nb_paths*(estimation / 30))


def performance_estimation_kruskal(file, routes):
    g = graph_from_file(file)
    out_route = open("routes.1.out", "w")
    MST = kruskal(g)
    individual_performances = []
    paths = open(routes, "r")
    nb_paths = int(paths.readline().strip())
    for i, line in enumerate(paths):
        if i == 0:
            continue # Skip the first line
        if i > 30:
            break # Stop reading after 30 lines
        n1, n2 = map(int, line.split())
        start = time.perf_counter()
        output = min_power_kruskal_V1(g, n1, n2)
        stop = time.perf_counter()
        individual_performances.append(stop - start)
        out_route.write(str(output) + "\n")
    out_route.close()
    estimation = sum(individual_performances)
    print("Temps estimé:", nb_paths*(estimation / 30))
