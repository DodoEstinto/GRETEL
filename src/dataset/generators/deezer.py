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

    #deezer_ego_nets
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
                graph_ind.append(graphId)

        # Read deezer_ego_nets_A.txt
        with open(self.local_config['parameters']['data_dir'] + '/deezer_ego_nets_A.txt') as f:
            lines = f.readlines()
            for line in lines:
                # line="1, 2" -> [1, 2]
                # instance = [[int(num) for num in line.split(' ')] for line in f]

                # Split the line
                nodes = line.split(', ')
                # Get the first node
                node1 = int(nodes[0])
                # Get the second node
                node2 = int(nodes[1])
                # Get the graph indicator
                graph_indicator = graph_ind[node1]

                # Add the edge to the graph
                graphs[graph_indicator].append([node1,node2])
        
        labels = np.array([])
        
        # Read deezer_ego_nets_graph_labels.txt
        with open(self.local_config['parameters']['data_dir'] + '/deezer_ego_nets_graph_labels.txt') as f:
            lines = f.readlines()
            # Iterate through the lines
            for line in lines:
                # Get the label
                label=int(line)
                # Add the label to the labels array
                labels.append(label)

        # Iterate through the graphs
        for i in np.arange(1, 9630):
        
            self.dataset.instances.append(GraphInstance(id=i, data=graphs[i], label=labels[i]))



    def check_configuration(self):
        #manage our default params here (that's an ugly part of GRETEL)
        super().check_configuration()
        local_config= self.local_config #field of master class which contains necessary params
        
        local_config['parameters']['num_instances'] = local_config['parameters'].get('num_instances', 1000)  #if not specified in config file it's 1000 by default, otherwise it takes it from config file (it's like a python dict)
        local_config['parameters']['num_nodes_per_instance']=local_config['parameters'].get('num_nodes_per_instance', 2)
        local_config['parameters']['infinity_cycle_length']=local_config['parameters'].get('infinity_cycle_length', 6)

        assert(int(local_config['parameters']['infinity_cycle_length']) // 2 >= 3)   #otherwise you can't have an ifinity-shaped cycle
        #we need int above cause float // int doesn't make sense





