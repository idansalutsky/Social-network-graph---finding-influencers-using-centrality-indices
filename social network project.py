import numpy as np
import networkx as nx
import pandas as pd
import random


def build_graph_probabilities_centralitys():
    df = pd.read_csv('instaglam0.csv')
    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(df, target='friendID', source='userID', create_using=Graphtype)
    graph = nx.to_undirected(G)

    df = pd.read_csv('instaglam_1.csv')
    Graphtype1 = nx.Graph()
    G1 = nx.from_pandas_edgelist(df, target='friendID', source='userID', create_using=Graphtype1)
    graph1 = nx.to_undirected(G1)

    Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
    big_comp = graph.subgraph(Gcc[0])


    neibor_num1 = {}
    diff = nx.difference(graph, graph1)
    common_sequence = []
    for edge in diff.edges:
        common_sequence.append(len(set(nx.common_neighbors(graph1, edge[0], edge[1]))))
        neibor_num1[edge[0]] = (len(set(nx.neighbors(graph1, edge[0]))), len(set(nx.neighbors(graph, edge[0]))))

    dict_common_diff = {}
    for val in common_sequence:
        if val in dict_common_diff.keys():
            total = dict_common_diff[val] + 1
            dict_common_diff[val] = total
        else:
            dict_common_diff[val] = 1

    nodes_list = sorted(np.array(list(graph1.nodes())))
    org_common = []
    common_dict = {}
    for a in range(len(nodes_list)):
        for b in range(len(nodes_list)):
            if a < b:
                x0 = (len(set(nx.common_neighbors(graph1, nodes_list[a], nodes_list[b]))))
                org_common.append(x0)
                common_dict[(nodes_list[a], nodes_list[b])] = x0

    dict_common_all = {}
    for val in org_common:
        if val in dict_common_all.keys():
            total = dict_common_all[val] + 1
            dict_common_all[val] = total
        else:
            dict_common_all[val] = 1

    list_present = []
    for i, key1 in enumerate(dict_common_diff.keys()):
        list_present.append((i, dict_common_diff[key1]/dict_common_all[key1], key1))

    print(list_present)
    print(dict_common_all)

    # Linear regression on relevant probabilities
    y_avg = sum([y[0] for y in list_present if y[1] <= 13])/12
    x_avg = sum([x[1] for x in list_present if x[1] <= 13]) / 12
    sxy = sum([(y[0]-y_avg)*(y[1]-x_avg) for y in list_present if y[1] <= 13])
    sxx = sum([(y[1]-x_avg)**2 for y in list_present if y[1] <= 13])
    print("b1 is:", sxy/sxx)
    print("b0 is:", y_avg-sxy/sxx*x_avg)

    # Finding central indices and ranking the nodes according to them
    center = nx.center(big_comp, usebounds=True)
    centrality_list = []
    centrality_list.append((center, "center"))
    centrality_list.append([nx.degree_centrality(big_comp), "dc"])
    centrality_list.append([nx.closeness_centrality(big_comp), "cc"])
    centrality_list.append([nx.betweenness_centrality(big_comp), "bc"])
    centrality_list.append([nx.harmonic_centrality(big_comp), "hc"])
    centrality_list.append([nx.pagerank(big_comp), "page rank"])

    centrality_dict = {}
    for centrality in centrality_list:
        if centrality[1] != "center":
            for i, val in enumerate(dict(sorted(centrality[0].items(), key=lambda item: item[1], reverse=True)).items()):
                if i <= 4:
                    if val[0] in centrality_dict.keys():
                        centrality_dict[val[0]][0] += 1
                        centrality_dict[val[0]][1].append([i, centrality[1]])
                    else:
                        centrality_dict[val[0]] = [1, [i, centrality[1]]]
                else:
                    break
        else:
            for val in centrality[0]:
                if val in centrality_dict.keys():
                    centrality_dict[val][0] += 1
                    centrality_dict[val][1].append(centrality[1])
                else:
                    centrality_dict[val] = [1, [centrality[1]]]

    for key, val in dict(sorted(centrality_dict.items(), key=lambda item: item[1][0], reverse=True)).items():

        print("node number: ", key, "has: ", val[0], "centrality measures or is a center and they are : ", val[1])

    """
    
    print(nx.diameter(big_comp)) = 9
    print(nx.radius(big_comp)) = 5
    print(len(centrality_dict)) = 19
    
        centers = nx.center(big_comp, usebounds=True)

    for nd in centers:
        print("node", nd, "have", len(list(nx.neighbors(big_comp, nd))), "neighbors")

    
    components = nx.connected_components(graph)
    for comp in components:
        x = len(comp)
        if x > 10:
           big_comp = nx.set
    one connected componnet with 1843 nodes
    and all the others under 10 nodes
    """

def f(x):
    if x >= 2:
        return 0.0183 + 0.0013 * x
    else:
        return 0


def simulation():
    df = pd.read_csv('instaglam0.csv')
    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(df, target='friendID', source='userID', create_using=Graphtype)
    graph0 = nx.to_undirected(G)
    Gcc = sorted(nx.connected_components(graph0), key=len, reverse=True)
    graph = graph0.subgraph(Gcc[0])
    graph_copy = nx.Graph(graph)
    gNode = nx.nodes(graph)

    artists = [511147, 532992, 70, 150]
    spotifly = pd.read_csv('spotifly.csv')
    spotifly = {(a, b): c for a, b, c in spotifly.values}

    for key in list(spotifly.keys()):
        if key[1] not in artists:
            del spotifly[key]
    plays_dict = {}
    buy_product_dict = {}
    probability_dict = {}
    for node in gNode:
        plays_dict[node] = 0
        buy_product_dict[node] = False
        probability_dict[node] = 0

    influencers_candidates = [411093, 548221, 874459, 999659, 175764, 606675, 308470, 682482]

    artists_dict = {}
    loop_counter = 0
    for k in range(3, 7):
        loop_counter += 1
        print(loop_counter, "loop from 10")
        for inf in range(k+1, 8):
            influencers_iter = [411093, 548221, 874459, influencers_candidates[k], influencers_candidates[inf]]

            for artist in artists:
                # adding num of plays to every node
                for key in spotifly.keys():
                    if key[1] == artist:
                        plays_dict[key[0]] = spotifly[key]
                nx.set_node_attributes(graph, plays_dict, "plays")

                # reset dict for new artist iteration
                for node in gNode:
                    buy_product_dict[node] = False
                    probability_dict[node] = 0

                graph = graph_copy
                for node in influencers_iter:
                    buy_product_dict[node] = True

                # 6 iteration per artist
                for i in range(6):
                    for node in gNode:
                        bt = 0
                        nt = len(list(nx.all_neighbors(graph, node)))
                        for neigh in nx.neighbors(graph, node):
                            if buy_product_dict[neigh]:
                                bt += 1

                        # calculating probabilities per node
                        if plays_dict[node] == 0:
                            if bt / nt > 1:
                                probability_dict[node] = 1
                            else:
                                probability_dict[node] = bt / nt
                        else:
                            h = plays_dict[node]
                            if (h * bt) / (1000 * nt) > 1:
                                probability_dict[node] = 1
                            else:
                                probability_dict[node] = (h * bt) / (1000 * nt)

                    # checking who infected
                    for key in probability_dict.keys():
                        rand = random.random()
                        if probability_dict[key] >= rand:
                            buy_product_dict[key] = True

                    # calculating probabilities for making new connections

                    graph_unfrozen = nx.Graph(graph)
                    nodes_list = np.array(list(graph_unfrozen.nodes()))
                    common_dict = {}
                    for a in range(len(nodes_list)):
                        for b in range(len(nodes_list)):
                            if a < b:
                                common = len(set(nx.common_neighbors(graph_unfrozen, nodes_list[a], nodes_list[b])))
                                common_dict[(nodes_list[a], nodes_list[b])] = common

                    for key in common_dict.keys():
                        rand = random.random()
                        if common_dict[key] != 0.0 and f(common_dict[key]) >= rand:
                            graph_unfrozen.add_edge(key[0], key[1])

                    graph = graph_unfrozen

                # num of infected nodes
                counter = 0
                for key in buy_product_dict:
                    if buy_product_dict[key]:
                        counter += 1

                if artist in artists_dict.keys():
                    artists_dict[artist].append([counter, influencers_iter])
                else:
                    artists_dict[artist] = [[counter, influencers_iter]]

    for key, val in artists_dict.items():
        max_val = max(val, key=lambda x: x[0])
        print("artist number: ", key, "has maximum of: ", max_val[0], "buyers, from the influencers: ", max_val[1])


if __name__ == "__main__":
    build_graph_probabilities_centralitys()
    simulation()
