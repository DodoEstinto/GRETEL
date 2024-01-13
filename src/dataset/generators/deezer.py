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
        base_path = self.local_config['parameters']['data_dir']

        self.generate_dataset()

    def get_num_instances(self):
        return len(self.dataset.instances)

    def generate_dataset(self):
        graphs = {}
        # Iterate through the keys and create key-value pairs
        for key in np.arange(1, 9630):
            graphs[key] = np.array([])

        graph_ind=np.array([])
        # Read deezer_ego_nets_graph_indicator.txt
        with open(self.local_config['parameters']['data_dir'] + '/deezer_ego_nets_graph_indicator.txt') as f:
            lines = f.readlines()
            # Iterate through the lines
            for line in lines:
                # Get the graph indicator
                graphId=int(line)
                # Add the graph indicator to the graph_ind array
                graph_ind=np.append(graph_ind,graphId)

        # Read deezer_ego_nets_A.txt
        with open(self.local_config['parameters']['data_dir'] + '/deezer_ego_nets_A.txt') as f:
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
                graphs[graph_indicator]=np.append(graphs[graph_indicator],[node1,node2])
        print(graphs[1].shape)
        labels = np.array([])
        
        # Read deezer_ego_nets_graph_labels.txt
        with open(self.local_config['parameters']['data_dir'] + '/deezer_ego_nets_graph_labels.txt') as f:
            lines = f.readlines()
            # Iterate through the lines
            for line in lines:
                # Get the label
                label=int(line)
                # Add the label to the labels array
                labels=np.append(labels,label)

        # Iterate through the graphs
        for i in np.arange(1, 9630):
            # graphs is a dictionary with key [1,9.629], while labels is an array with index [0,9.628]
            self.dataset.instances.append(GraphInstance(id=i, data=graphs[i], label=labels[i-1]))



    def check_configuration(self):
        #manage our default params here (that's an ugly part of GRETEL)
        super().check_configuration()
        if not self.local_config['parameters']['data_dir']:
            self.local_config['parameters']['data_dir'] = 'data/deezer_ego_nets'
      




