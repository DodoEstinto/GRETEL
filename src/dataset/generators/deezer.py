import networkx as nx
import numpy as np
from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance


class Deezer(Generator):

    """
    FATTO | Dizionario con key [1,9.629] -> graphs
     FATTO | Leggiamo graph_indicator -> array = graph_ind
    Leggiamo e scorriamo A:
	    Prendiamo il valore g_i in graph_ind[A_i[0]]
	    Aggiungiamo a graphs[g_i] il valore A_i

    FATTO |
    Leggere graph_labels -> array = labels
    FATTO |
    Scorriamo graphs:
	    Creiamo GraphIstance con id=i,data=graphs[i],label= labes[i]
	    Aggiungiamo GraphIstance a self.dataset.instances
    """

    def init(self):
        #we pass the local_config (with check_configuration()) before calling init()
        self.base_path = self.local_config['parameters']['data_dir']
        self._max_nodes = self.local_config['parameters']['max_nodes']
        #self.dataset.node_features_map={"node_causality":self._max_nodes}
        print("Init Dataset Deezer")
        self.generate_dataset()

    

    def get_num_instances(self):
        return len(self.dataset.instances)

    def generate_dataset(self):
        graphs = {}
        # Iterate through the keys and create key-value pairs
        for key in np.arange(1, 9630):
            graphs[key] = np.array([]).reshape([-1,2])

        graph_ind=np.array([])
        # Read deezer_ego_nets_graph_indicator.txt
        with open(self.base_path + '/deezer_ego_nets_graph_indicator.txt') as f:
            lines = f.readlines()
            # Iterate through the lines
            for line in lines:
                # Get the graph indicator
                graphId=int(line)
                # Add the graph indicator to the graph_ind array
                graph_ind=np.append(graph_ind,graphId)

        # Read deezer_ego_nets_A.txt
        with open(self.base_path + '/deezer_ego_nets_A.txt') as f:
            lines = f.readlines()
            for line in lines:
                # line="1, 2" -> [1, 2]

                # Split the line
                nodes = line.split(', ')
                # Get the first node
                node1 = int(nodes[0])
                # Get the second node
                node2 = int(nodes[1])
                # Get the graph indicator
                graph_indicator = graph_ind[node1-1]
                # Add the edge to the graph
                graphs[graph_indicator]=np.vstack([graphs[graph_indicator],[node1,node2]])
        #print(graphs[1].shape)
        #print(graphs[1])
        
        labels = np.array([])
        
        # Read deezer_ego_nets_graph_labels.txt
        with open(self.base_path + '/deezer_ego_nets_graph_labels.txt') as f:
            lines = f.readlines()
            # Iterate through the lines
            for line in lines:
                # Get the label
                label=int(line)
                # Add the label to the labels array
                labels=np.append(labels,label)

        # Iterate through the graphs
        for i in np.arange(1, 100): #9630  #TODO: cambiare in 9630
            # graphs is a dictionary with key [1,9.629], while labels is an array with index [0,9.628]
            data=self.create_adj_mat(graphs[i])
            self.dataset.instances.append(GraphInstance(id=i, data=data, label=int(labels[i-1])))
        self.dataset.node_features_map={"node_causality":self._max_nodes}
        
    def create_adj_mat(self, data):
            adj_list = np.asarray(data)
            min_node = adj_list.min()
            max_node = adj_list.max()
            adj_list = (adj_list - min_node).T

            min_node = min_node - 1 
            nodes = max_node - min_node

            #should not happens
            if nodes > self._max_nodes:
                return None

            mat = np.zeros((self._max_nodes, self._max_nodes), dtype=np.int32)
            edges=zip(adj_list[0],adj_list[1])
            for i in edges:
                i1,i2=int(i[0]),int(i[1])
                mat[i1,i2] = 1
                mat[i2,i1] = 1

            return mat

    def check_configuration(self):
        #manage our default params here (that's an ugly part of GRETEL)
        super().check_configuration()
        if not self.local_config['parameters']['data_dir']:
            self.local_config['parameters']['data_dir'] = 'data/deezer_ego_nets'
      




