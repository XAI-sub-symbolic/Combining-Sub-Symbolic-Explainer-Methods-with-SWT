__author__ = 'martin.ringsquandl'

import torch
import numpy as np
import matplotlib.pyplot as plt

from explainer import explain
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from explainer import models


def deleteNode(edge_list, node_id):
    """
    Removes all entries of node_id from edge_list
    :param edge_list:
    :param node_id:
    :return:
    """
    inds = np.where(edge_list == node_id)
    mask = torch.ones(edge_list.shape[1])
    print(inds[1])
    mask[inds[1]] = 0
    tensor = edge_list[:, mask.to(torch.bool)]

    return tensor
    

def deleteEdge(edge_list, node_id):
    """
    Removes all entries of node_id from edge_list
    :param edge_list:
    :param node_id:
    :return:
    """
    ind_rev = np.where((edge_list[1] == edge_list[0][node_id]) & (edge_list[0] == edge_list[1][node_id]))
    inds = np.array([node_id, ind_rev[0][0]])
    mask = torch.ones(edge_list.shape[1])
    mask[inds] = 0
    tensor = edge_list[:, mask.to(torch.bool)]

    return tensor
    
    
def gnnx_result(node_id, prog_args, unconstrained=False):

    ckpt = io_utils.load_ckpt(prog_args)
    cg_dict = ckpt["cg"]    # get computation graph

    input_dim = cg_dict["feat"].shape[2] 
    num_classes = cg_dict["pred"].shape[2]
 
    model = models.GcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=prog_args.hidden_dim,
            embedding_dim=prog_args.output_dim,
            label_dim=num_classes,
            num_layers=prog_args.num_gc_layers,
            bn=prog_args.bn,
            args=prog_args,
        )
    if prog_args.gpu:
        model = model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load state_dict (obtained by model.state_dict() when saving checkpoint)

    model.load_state_dict(ckpt["model_state"]) 
    writer = None
  

    experiment = explain.Explainer(
        model=model,
        adj=cg_dict["adj"],
        feat=cg_dict["feat"],
        label=cg_dict["label"],
        pred=cg_dict["pred"],
        train_idx=cg_dict["train_idx"],
        args=prog_args,
        writer=writer,
        print_training=True,
        graph_mode=False,
        graph_idx=prog_args.graph_idx,
    )

    result_gnnx = experiment.explain(node_id, prog_args, unconstrained=False)
    return result_gnnx


def visualize_subgraph(node_idx, edge_index, edge_mask, y=None,
                       threshold=None, **kwargs):
    # Only operate on a k-hop subgraph around `node_idx`.
    subset, edge_index, hard_edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index, relabel_nodes=True,
        num_nodes=None, flow='source_to_target')

    if threshold is not None:
        edge_mask = (edge_mask >= threshold).to(torch.float)

    if y is None:
        y = torch.zeros(edge_index.max().item() + 1,
                        device=edge_index.device)
    else:
        y = y[subset].to(torch.float) / y.max().item()

    data = Data(edge_index=edge_index, att=edge_mask, y=y,
                num_nodes=y.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
    mapping = {k: i for k, i in enumerate(subset.tolist())}
    G = nx.relabel_nodes(G, mapping)

    kwargs['with_labels'] = kwargs.get('with_labels') or True
    kwargs['font_size'] = kwargs.get('font_size') or 10
    kwargs['node_size'] = kwargs.get('node_size') or 800
    kwargs['cmap'] = kwargs.get('cmap') or 'cool'

    pos = nx.spring_layout(G)
    ax = plt.gca()
    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->",
                alpha=max(data['att'], 0.1),
                shrinkA=sqrt(kwargs['node_size']) / 2.0,
                shrinkB=sqrt(kwargs['node_size']) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ))
    nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **kwargs)
    nx.draw_networkx_labels(G, pos, **kwargs)
    plt.axis('off')
    return plt


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    subsets = [torch.tensor([node_idx], device=row.device).flatten()]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
    subset = torch.cat(subsets).unique()
    # Add `node_idx` to the beginning of `subset`.
    subset = subset[subset != node_idx]
    subset = torch.cat([torch.tensor([node_idx], device=row.device), subset])

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, edge_mask


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def discount_rewards(rewards, gamma):
    # r = np.array([gamma**i * rewards[i]
    #          for i in range(len(rewards))])
    # # Reverse the array direction for cumsum and then
    # # revert back to the original order
    # r = r[::-1].cumsum()[::-1]
    # mean = r.mean()
    # r = r - mean
    # return r
    t_steps = np.arange(len(rewards))
    r = np.array(rewards) * gamma ** t_steps
    r = r[::-1].cumsum()[::-1] / gamma ** t_steps
    return r


def subgraph(node_idx, x, edge_index, hops_policy):
    subset, edge_index, edge_mask = k_hop_subgraph(
        node_idx, hops_policy, edge_index, relabel_nodes=True,
        num_nodes=None, flow='source_to_target')

    x = x[subset]

    return x, edge_index, edge_mask, subset


def numberOfOutputs(node_idx, edge_index, exclude_node, num_hops):
    nodes_subset, edge_index_subset, edge_mask = k_hop_subgraph(node_idx, num_hops, edge_index)
    nodes_minus = torch.cat([nodes_subset[0:0], nodes_subset[0 + 1:]]).tolist()
    nodes_minus.append(-1)
    possibleActions = [node for node in nodes_minus if node not in exclude_node]
    # print("possible Actions", possibleActions)
    size = len(possibleActions)
    return size


def lowest_entropy(node_idx, edge_index, x, entropy_class):
    nodes_subset, edge_index_subset, edge_mask = k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                                                                num_nodes=None, flow='source_to_target')
    data_1 = Data(edge_index=edge_index_subset, num_nodes=len(edge_index_subset[0]))
    # print("orig graph", edge_index_subset)
    G = to_networkx(data_1, to_undirected=True)
    edge_list = [e for e in G.edges]
    possible_graphs = []

    for length in range(1, len(nodes_subset)):
        for i in itertools.combinations(edge_list, length):
            possible_graphs.append(i)

    entropy_list = []
    final_graphs = []
    print("possible graphs", len(possible_graphs))
    for graph in possible_graphs:
        graph = list(graph)
        flattened = [val for sublist in graph for val in sublist]
        if node_idx in flattened:

            for element in range(len(graph)):
                element_rev = (graph[element][1], graph[element][0])
                graph.append(element_rev)

            final_edge_index_0 = []
            final_edge_index_1 = []

            for edge in range(len(graph)):
                final_edge_index_0.append(graph[edge][0])
                final_edge_index_1.append(graph[edge][1])

            final_edge_index = [final_edge_index_0, final_edge_index_1]
            final_edge_index = torch.LongTensor(final_edge_index).to(device)
            org, diff, final = entropy_class.logits(node_idx, x, edge_index_subset.to(device),
                                                    final_edge_index.to(device))
            label_switch, prob_before, prob_after, label_new, label_orig = entropy_class.switch(node_idx, x,
                                                                                                edge_index_subset.to(
                                                                                                    device),
                                                                                                final_edge_index.to(
                                                                                                    device))
            if label_new == label_orig:
                entropy_list.append(final.item())
                final_graphs.append(final_edge_index)

    print("possible outcomes:", len(final_graphs))
    print("-------------GROUND TRUTH MINIMUM ENTROPY-------------")
    # print("entropies:", entropy_list)
    print("Minimum possible Entropy:")
    min_entropy = min(entropy_list)
    print(min(entropy_list))
    best_idx = entropy_list.index(min(entropy_list))
    print("Minimal Entropy Subgraph:", final_graphs[best_idx])

    return min_entropy, final_graphs[best_idx]

def neighborhoods(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    adj = torch.tensor(adj, dtype=torch.float)
    if use_cuda:
        adj = adj.cuda()
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.cpu().numpy().astype(int)

def extract_neighborhood(adj, feat, label, n_hops, node_idx, graph_idx=0):
        """Returns the neighborhood of a given ndoe."""
        neighborhood = neighborhoods(adj, n_hops, use_cuda=True)
        
        neighbors_adj_row = neighborhood[graph_idx][node_idx, :]
        # index of the query node in the new adj
       
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        
        sub_adj = adj[graph_idx][neighbors][:, neighbors]
        sub_feat = feat[graph_idx, neighbors]
        sub_label = label[graph_idx][neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors