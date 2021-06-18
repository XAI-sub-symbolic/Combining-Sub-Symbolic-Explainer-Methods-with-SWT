
""" explain.py
    Implementation of the explainer.
"""

import math
import time
import os

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)
import seaborn as sns
import tensorboardX.utils

import torch
import torch.nn as nn
from torch.autograd import Variable

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve
from sklearn.cluster import DBSCAN

import pdb

import utils.io_utils as io_utils
import utils.train_utils as train_utils
import utils.graph_utils as graph_utils


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

class Explainer:
    def __init__(
        self,
        model,
        adj,
        feat,
        label,
        pred,
        train_idx,
        args,
        writer=None,
        print_training=False,
        graph_mode=False,
        graph_idx=False,
    ):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        self.n_hops = args.num_gc_layers
        self.graph_mode = graph_mode
        self.graph_idx = graph_idx
        self.neighborhoods = None if self.graph_mode else graph_utils.neighborhoods(adj=self.adj, n_hops=self.n_hops, use_cuda=use_cuda)
        self.args = args
        self.writer = writer
        self.print_training = print_training


    # Main method
    def explain(
        self, node_idx, graph_idx=0, graph_mode=False, unconstrained=False, model="exp"
    ):
        """Explain a single node prediction
        """
        # index of the query node in the new adj
        if graph_mode:
            node_idx_new = node_idx
            sub_adj = self.adj[graph_idx]
            sub_feat = self.feat[graph_idx, :]
            sub_label = self.label[graph_idx]
            #print("label", sub_label)
            neighbors = np.asarray(range(self.adj.shape[0]))
        else:
            outfile.write("node label: ", self.label[graph_idx][node_idx])
            node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                node_idx, graph_idx
            )
            outfile.write("neigh graph idx: ", node_idx, node_idx_new)
            sub_label = np.expand_dims(sub_label, axis=0)

        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)

        adj   = torch.tensor(sub_adj, dtype=torch.float)
        x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)

        if self.graph_mode:
            #print("self pred shape", self.pred.shape)
            #print("pred", self.pred)
            pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
            #print("pred label", pred_label)
            #outfile.write("Graph predicted label: ", pred_label)
        else:
            pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)
            outfile.write("Node predicted label: ", pred_label[node_idx_new])

        explainer = ExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label,
            args=self.args,
            writer=self.writer,
            graph_idx=self.graph_idx,
            graph_mode=self.graph_mode,
        )
        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()


        # gradient baseline
        if model == "grad":
            explainer.zero_grad()
            # pdb.set_trace()
            adj_grad = torch.abs(
                explainer.adj_feat_grad(node_idx_new, pred_label[node_idx_new])[0]
            )[graph_idx]
            masked_adj = adj_grad + adj_grad.t()
            masked_adj = nn.functional.sigmoid(masked_adj)
            masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
        else:
            explainer.train()
            begin_time = time.time()
            for epoch in range(self.args.num_epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                ypred, adj_atts, feat_mask = explainer(node_idx_new, unconstrained=unconstrained)
                loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
                loss.backward()

                explainer.optimizer.step()
                if explainer.scheduler is not None:
                    explainer.scheduler.step()

                mask_density = explainer.mask_density()
                '''
                if self.print_training:
                    outfile.write(
                        "epoch: ",
                        epoch,
                        "; loss: ",
                        loss.item(),
                        "; mask density: ",
                        mask_density.item(),
                        "; pred: ",
                        ypred,
                    )
                '''
                single_subgraph_label = sub_label.squeeze()

                if self.writer is not None:
                    self.writer.add_scalar("mask/density", mask_density, epoch)
                    self.writer.add_scalar(
                        "optimization/lr",
                        explainer.optimizer.param_groups[0]["lr"],
                        epoch,
                    )
                    if epoch % 25 == 0:
                        explainer.log_mask(epoch)
                        explainer.log_masked_adj(
                            node_idx_new, epoch, label=single_subgraph_label
                        )
                        explainer.log_adj_grad(
                            node_idx_new, pred_label, epoch, label=single_subgraph_label
                        )

                    if epoch == 0:
                        if self.model.att:
                            # explain node
                            outfile.write("adj att size: ", adj_atts.size())
                            adj_att = torch.sum(adj_atts[0], dim=2)
                            # adj_att = adj_att[neighbors][:, neighbors]
                            node_adj_att = adj_att * adj.float().cuda()
                            io_utils.log_matrix(
                                self.writer, node_adj_att[0], "att/matrix", epoch
                            )
                            node_adj_att = node_adj_att[0].cpu().detach().numpy()
                            G = io_utils.denoise_graph(
                                node_adj_att,
                                node_idx_new,
                                threshold=3.8,  # threshold_num=20,
                                max_component=True,
                            )
                            io_utils.log_graph(
                                self.writer,
                                G,
                                name="att/graph",
                                identify_self=not self.graph_mode,
                                nodecolor="label",
                                edge_vmax=None,
                                args=self.args,
                            )
                if model != "exp":
                    break

            #outfile.write("finished training in ", time.time() - begin_time)
            if model == "exp":
                masked_adj = (
                    explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
                )
            else:
                adj_atts = nn.functional.sigmoid(adj_atts).squeeze()
                masked_adj = adj_atts.cpu().detach().numpy() * sub_adj.squeeze()

        fname = 'masked_adj_' + io_utils.gen_explainer_prefix(self.args) + (
                'node_idx_'+str(node_idx)+'graph_idx_'+str(self.graph_idx)+'.npy')
        with open(os.path.join(self.args.logdir, fname), 'wb') as outfile:
            np.save(outfile, np.asarray(masked_adj.copy()))
            #outfile.write("Saved adjacency matrix to ", fname)
        return masked_adj, label, pred_label, feat_mask


    # NODE EXPLAINER
    def explain_nodes(self, node_indices, args, graph_idx=0):
        """
        Explain nodes
        Args:
            - node_indices  :  Indices of the nodes to be explained
            - args          :  Program arguments (mainly for logging paths)
            - graph_idx     :  Index of the graph to explain the nodes from (if multiple).
        """
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx) for node_idx in node_indices
        ]
        ref_idx = node_indices[0]
        ref_adj = masked_adjs[0]
        curr_idx = node_indices[1]
        curr_adj = masked_adjs[1]
        new_ref_idx, _, ref_feat, _, _ = self.extract_neighborhood(ref_idx)
        new_curr_idx, _, curr_feat, _, _ = self.extract_neighborhood(curr_idx)

        G_ref = io_utils.denoise_graph(ref_adj, new_ref_idx, ref_feat, threshold=0.1)
        denoised_ref_feat = np.array(
            [G_ref.nodes[node]["feat"] for node in G_ref.nodes()]
        )
        denoised_ref_adj = nx.to_numpy_matrix(G_ref)
        # ref center node
        ref_node_idx = list(G_ref.nodes()).index(new_ref_idx)

        G_curr = io_utils.denoise_graph(
            curr_adj, new_curr_idx, curr_feat, threshold=0.1
        )
        denoised_curr_feat = np.array(
            [G_curr.nodes[node]["feat"] for node in G_curr.nodes()]
        )
        denoised_curr_adj = nx.to_numpy_matrix(G_curr)
        # curr center node
        curr_node_idx = list(G_curr.nodes()).index(new_curr_idx)

        P, aligned_adj, aligned_feat = self.align(
            denoised_ref_feat,
            denoised_ref_adj,
            ref_node_idx,
            denoised_curr_feat,
            denoised_curr_adj,
            curr_node_idx,
            args=args,
        )
        io_utils.log_matrix(self.writer, P, "align/P", 0)

        G_ref = nx.convert_node_labels_to_integers(G_ref)
        io_utils.log_graph(self.writer, G_ref, "align/ref")
        G_curr = nx.convert_node_labels_to_integers(G_curr)
        io_utils.log_graph(self.writer, G_curr, "align/before")

        P = P.cpu().detach().numpy()
        aligned_adj = aligned_adj.cpu().detach().numpy()
        aligned_feat = aligned_feat.cpu().detach().numpy()

        aligned_idx = np.argmax(P[:, curr_node_idx])
        outfile.write("aligned self: ", aligned_idx)
        G_aligned = io_utils.denoise_graph(
            aligned_adj, aligned_idx, aligned_feat, threshold=0.5
        )
        io_utils.log_graph(self.writer, G_aligned, "mask/aligned")

        # io_utils.log_graph(self.writer, aligned_adj.cpu().detach().numpy(), new_curr_idx,
        #        'align/aligned', epoch=1)

        return masked_adjs


    def explain_nodes_gnn_stats(self, node_indices, args, graph_idx=0, model="exp"):
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx, model=model)
            for node_idx in node_indices
        ]
        # pdb.set_trace()
        graphs = []
        feats = []
        adjs = []
        pred_all = []
        real_all = []
        for i, idx in enumerate(node_indices):
            new_idx, _, feat, _, _ = self.extract_neighborhood(idx)
            G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat, threshold_num=20)
            pred, real = self.make_pred_real(masked_adjs[i], new_idx)
            pred_all.append(pred)
            real_all.append(real)
            denoised_feat = np.array([G.nodes[node]["feat"] for node in G.nodes()])
            denoised_adj = nx.to_numpy_matrix(G)
            graphs.append(G)
            feats.append(denoised_feat)
            adjs.append(denoised_adj)
            io_utils.log_graph(
                self.writer,
                G,
                "graph/{}_{}_{}".format(self.args.dataset, model, i),
                identify_self=True,
                args=self.args
            )

        pred_all = np.concatenate((pred_all), axis=0)
        real_all = np.concatenate((real_all), axis=0)

        auc_all = roc_auc_score(real_all, pred_all)
        precision, recall, thresholds = precision_recall_curve(real_all, pred_all)

        plt.switch_backend("agg")
        plt.plot(recall, precision)
        plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")

        plt.close()

        auc_all = roc_auc_score(real_all, pred_all)
        precision, recall, thresholds = precision_recall_curve(real_all, pred_all)

        plt.switch_backend("agg")
        plt.plot(recall, precision)
        plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")

        plt.close()

        with open("log/pr/auc_" + self.args.dataset + "_" + model + ".txt", "w") as f:
            f.write(
                "dataset: {}, model: {}, auc: {}\n".format(
                    self.args.dataset, "exp", str(auc_all)
                )
            )

        return masked_adjs

    # GRAPH EXPLAINER
    def explain_graphs(self, graph_indices):
        """
        Explain graphs.
        """
        
        print("EXPLAIN GRAPHS")
        masked_adjs = []
        cycle_aromatic = []
        
        
        filename = 'owl.txt' #This is simply a string of text
        outfile = open(filename, 'w') # 'r' says we are opening the file to read, infile is the opened file object that we will read from


        # For fidelity & entailment calculation per graph.
        # Explainer Classes Defined.
               
        #hasStructure some Methyl
        methyl_entailment = []
        methyl_fidelity = []
        
        #hasStructure some Nitro
        nitro_entailment = []
        nitro_fidelity = []
        
        #hasThreeofmoreFusedRings value true        
        three_rings_entailment = []
        three_rings_fidelity = []        
        
        #Phenanthrene
        phenanthrene_entailment = []
        phenanthrene_fidelity = []
        
        #phosphorus
        phosphorus_entailment = []
        phosphorus_fidelity = []
        
        #oxygen
        oxygen_entailment = []
        oxygen_fidelity = []
        
        #carbon
        carbon_entailment = []
        carbon_fidelity = []
        
        #nitrogen
        nitrogen_entailment = []
        nitrogen_fidelity = []
        
        #hydrogen
        hydrogen_entailment = []
        hydrogen_fidelity = []
        
        #bromine
        bromine_entailment = []
        bromine_fidelity = []
        
        #sulfur
        sulfur_entailment = []
        sulfur_fidelity = []
        
        #hasStructure Ring_size_5 or Ring_size_6
        has_ring_entailment = []
        has_ring_fidelity = []
        
        #hasStructure Ring_size_5 
        has_ring5_entailment = []
        has_ring5_fidelity = []
        
        #hasStructure Ring_size_6
        has_ring6_entailment = []
        has_ring6_fidelity = []
        
        #hasStructure Carbon ring 6
        has_carbon_ring6_entailment = []
        has_carbon_ring6_fidelity = []
        
        has_carbon_ring5_entailment = []
        has_carbon_ring5_fidelity = []
        
        has_aroma_ring6_entailment = []
        has_aroma_ring6_fidelity = []
        
        has_aroma_ring5_entailment = []
        has_aroma_ring5_fidelity = []
        
        azanide_entailment = []
        azanide_fidelity = []
        
        only_methyl_entailment = []
        only_methyl_fidelity = []
        
        methyl_phosul_entailment = []
        methyl_phosul_fidelity = []
        
        sulfur_methyl_entailment = []
        sulfur_methyl_fidelity = []
        
        
        number_instances  = 0
        
        
        for graph_idx in graph_indices:
            #print("graph idx", graph_idx)
            
            print('"kb:in'+str(graph_idx)+'",')
           
            
            masked_adj, label, pred_label, feat_mask = self.explain(node_idx=0, graph_idx=graph_idx, graph_mode=True)
            label = label.item()
            #print(label, pred_label)
            
            #if label != pred_label: 
            if label != pred_label:
            
            #or label != pred_label: 
                #if label != pred_label:
                 #   print("incorrect")
                number_instances = number_instances + 1
                #print("size graph", len(np.nonzero(masked_adj)[0]))
                #print(len(np.nonzero(masked_adj)[0]))
                if len(np.nonzero(masked_adj)[0]) < 30:
                    thr_var = 12
                elif len(np.nonzero(masked_adj)[0]) >= 30 and  len(np.nonzero(masked_adj)[0]) < 60:
                     thr_var = 15
                else: thr_var = 18
                G_denoised, threshold = io_utils.denoise_graph(
                    masked_adj,
                    0,
                    threshold_num=thr_var,
                    feat=self.feat[graph_idx],
                    max_component=False,
                )
                #outfile.write("threshold", threshold)
                
                G_orig, threshold = io_utils.denoise_graph(
                    masked_adj,
                    0,
                    threshold_num=None,
                    feat=self.feat[graph_idx],
                    max_component=False,
                )
                label = self.label[graph_idx]
                io_utils.log_graph(
                    self.writer,
                    G_denoised,
                    "graph/graphidx_{}_label={}".format(graph_idx, label),
                    identify_self=False,
                    nodecolor="feat",
                    args=self.args
                )
                
    
                orig_nodes = []
                final_nodes = []   
                graph_idx_sub = str(graph_idx) + "_sub"  
                
                #print("edges denoised", G_denoised.edges(data=True))   
                #print("edges normal", G_orig.edges(data=True))

                feat_mask_bool = []
                
                for feat in feat_mask:
                   if feat > -3 : feat_mask_bool.append(1)
                   else: feat_mask_bool.append(0)
                

                features = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca", "Nitro", "Azanide", "Methyl", "Homo5", "Hetero5", "Homo6", "Hetero6"]
                dict_feat = dict(list(zip(features, feat_mask_bool)))
                
                #print("dict feat", dict_feat)
               
                for x in G_denoised.nodes(data='feat'):

                    if x[1].numpy()[0] == 1: n = "C"
                    elif x[1].numpy()[1] == 1: n = "O"
                    elif x[1].numpy()[2] == 1: n = "Cl"
                    elif x[1].numpy()[3] == 1: n = "H"
                    elif x[1].numpy()[4] == 1: n = "N"
                    elif x[1].numpy()[5] == 1: n = "F"
                    elif x[1].numpy()[6] == 1: n = "Br"
                    elif x[1].numpy()[7] == 1: n = "S"
                    elif x[1].numpy()[8] == 1: n = "P"
                    elif x[1].numpy()[9] == 1: n = "I"
                    elif x[1].numpy()[10] == 1: n = "Na"
                    elif x[1].numpy()[11] == 1: n = "K"
                    elif x[1].numpy()[12] == 1: n = "Li"
                    elif x[1].numpy()[13] == 1: n = "Ca"
                    final_nodes.append(n)
                    
    
                important_atoms = []
                
                
    
                for x in G_orig.nodes(data='feat'):
    
                    if x[1].numpy()[0] == 1: n = "C"
                    elif x[1].numpy()[1] == 1: n = "O"
                    elif x[1].numpy()[2] == 1: n = "Cl"
                    elif x[1].numpy()[3] == 1: n = "H"
                    elif x[1].numpy()[4] == 1: n = "N"
                    elif x[1].numpy()[5] == 1: n = "F"
                    elif x[1].numpy()[6] == 1: n = "Br"
                    elif x[1].numpy()[7] == 1: n = "S"
                    elif x[1].numpy()[8] == 1: n = "P"
                    elif x[1].numpy()[9] == 1: n = "I"
                    elif x[1].numpy()[10] == 1: n = "Na"
                    elif x[1].numpy()[11] == 1: n = "K"
                    elif x[1].numpy()[12] == 1: n = "Li"
                    elif x[1].numpy()[13] == 1: n = "Ca"
                    orig_nodes.append(n)
    
                    
                
                l = range(len(orig_nodes))
                dict_atoms = dict(zip(l, orig_nodes))
                #print("dict final", dict_atoms)
                edges_final = G_denoised.edges()
                edges_orig = G_orig.edges()
                mapped_final = []
                for item in edges_final: 
                    mapped_final.append(list((pd.Series(item)).map(dict_atoms)))
                #print('mapped', mapped_final)
                mapped_orig = [] 
                for item in edges_orig: 
                    mapped_orig.append(list((pd.Series(item)).map(dict_atoms)))

                if "P" in orig_nodes:
                   phosphorus_entailment.append(1)
                   if "P" in final_nodes:
                         phosphorus_fidelity.append(1)
                         fidelity_phosphorus = 1
                   else: 
                        phosphorus_fidelity.append(0)
                        fidelity_phosphorus = 0
                else: phosphorus_entailment.append(0)
                
                
                if "Br" in orig_nodes:
                   bromine_entailment.append(1)
                   if "Br" in final_nodes:
                         bromine_fidelity.append(1)
                   else: bromine_fidelity.append(0)
                else: bromine_entailment.append(0)
                

                
                if "C" in orig_nodes:
                   carbon_entailment.append(1)
                   if "C" in final_nodes:
                        carbon_fidelity.append(1)
                   else: carbon_fidelity.append(0)
                else: carbon_entailment.append(0)
                
                
                if "H" in orig_nodes:
                   hydrogen_entailment.append(1)
                   if "H" in final_nodes:
                         hydrogen_fidelity.append(1)
                   else: hydrogen_fidelity.append(0)
                else: hydrogen_entailment.append(0)
                
                
                if "N" in orig_nodes:
                   nitrogen_entailment.append(1)
                   if "N" in final_nodes:
                         nitrogen_fidelity.append(1)
                   else: nitrogen_fidelity.append(0)
                else: nitrogen_entailment.append(0)
                
                
                if "O" in orig_nodes:
                   oxygen_entailment.append(1)
                   if "O" in final_nodes:
                         oxygen_fidelity.append(1)
                   else: oxygen_fidelity.append(0)
                else: oxygen_entailment.append(0)
                

                
                if "S" in orig_nodes:
                   sulfur_entailment.append(1)
                   if "S" in final_nodes:
                        sulfur_fidelity.append(1)
                        fidelity_sulfur = 1
                   else: 
                         sulfur_fidelity.append(0)
                         fidelity_sulfur = 0
                else: sulfur_entailment.append(0)
                

                
                counter = 0
                
                for v in G_orig.nodes:
                   if v in G_denoised.nodes: important_atoms.append(1)
                   else: important_atoms.append(0)
                      
                
                
                '''
                for v in range(len(orig_nodes)):
    
                       if (v - counter) > (len(final_nodes) - 1):
                          important_atoms.append(0)
                       else:
                          if orig_nodes[v] == final_nodes[v - counter]:
                             important_atoms.append(1)
                          else: 
                             important_atoms.append(0)
                             counter = counter + 1
                '''
                
                
      
                #outfile.write("node attributes", G_denoised.nodes(data="feat"))
                
                #outfile.write("edges data org", G_orig.edges.data())               
                #outfile.write("edges data", G_denoised.edges.data())
                
                
                def find_all_cycles(G, source=None, cycle_length_limit=None):
                    """forked from networkx dfs_edges function. Assumes nodes are integers, or at least
                    types which work with min() and > ."""
                    if source is None:
                        # produce edges for all components
                        nodes=[list(i)[0] for i in nx.connected_components(G)]
                    else:
                        # produce edges for components with source
                        nodes=[source]
                    # extra variables for cycle detection:
                    cycle_stack = []
                    output_cycles = set()
                    
                    def get_hashable_cycle(cycle):
                        """cycle as a tuple in a deterministic order."""
                        m = min(cycle)
                        mi = cycle.index(m)
                        mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
                        if cycle[mi-1] > cycle[mi_plus_1]:
                            result = cycle[mi:] + cycle[:mi]
                        else:
                            result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
                        return tuple(result)
                    

                    for start in nodes:
                        if start in cycle_stack:
                            continue
                        cycle_stack.append(start)
                        
                        stack = [(start,iter(G[start]))]
                        while stack:
                            parent,children = stack[-1]
                            try:
                                child = next(children)
                                
                                if child not in cycle_stack:
                                    cycle_stack.append(child)
                                    stack.append((child,iter(G[child])))
                                   
                                else:
                                    i = cycle_stack.index(child)
                                                                   
                                    if i < len(cycle_stack) - 2: 
                                      output_cycles.add(get_hashable_cycle(cycle_stack[i:]))
                                   
                                
                            except StopIteration:
                                stack.pop()
                                cycle_stack.pop()
                    
                    #outfile.write("output cycles", output_cycles)

                    #print("output cycles", output_cycles)
                    final_cycles = []
                    for i in output_cycles:
                       if len(i) >= 5:
                          if len(i) <= cycle_length_limit:
                             final_cycles.append(i)
                    return [list(i) for i in final_cycles]
                    
                #outfile.write("find cycles", len(list(nx.find_cycle(G_denoised, orientation="ignore"))))
                
                list_cycles_6 = find_all_cycles(G_denoised, cycle_length_limit=6)
                list_cycles_5 = find_all_cycles(G_denoised, cycle_length_limit=5)
                #outfile.write(find_all_cycles(G_denoised))
               
                orig_cycles_6 = find_all_cycles(G_orig, cycle_length_limit = 6)
                orig_cycles_5= find_all_cycles(G_orig, cycle_length_limit = 5)
                #outfile.write("list cycles orig", orig_cycles)
                

                
                if orig_cycles_6 or orig_cycles_5:
                    has_ring_entailment.append(1)
                else: has_ring_entailment.append(0)

                if orig_cycles_6 or orig_cycles_5:
                     cycle_6_total = []
                     sum_cycle = []
                     for item in orig_cycles_6:
                         subtotal = 0
                         #print("item", item)
                         for subitem in item:    
                            #print("item", subitem)                   
                            if subitem in G_denoised.nodes:
                               subtotal = subtotal + 1
                            
                         cycle_6_total.append(subtotal)
                     
                     if not cycle_6_total:  sum_cycle.append(0)
                     else: sum_cycle.append(max(cycle_6_total)/6)
                     
                     cycle_5_total = []
                     for item in orig_cycles_5:
                         subtotal = 0
                         #print("item", item)
                         for subitem in item:    
                            #print("item", subitem)                   
                            if subitem in G_denoised.nodes:
                               subtotal = subtotal + 1
                            
                         cycle_5_total.append(subtotal)
                     
                     if not cycle_5_total:  sum_cycle.append(0)
                     else: sum_cycle.append(max(cycle_5_total)/5)
                     
     
                     if not cycle_5_total and not cycle_6_total: has_ring_fidelity.append(0)
                     else: has_ring_fidelity.append(max(sum_cycle))
                     
                     
               
                
                #Check if orig cycle is hetero or homo aromatic
                hetero_orig_6 = []
                homo_orig_6 = []
                homo_list_6 = []
                hetero_list_6 = []
                for l in range(len(orig_cycles_6)):
                   hetero = []
                   for x in orig_cycles_6[l]:
                         value = orig_nodes[x]
                         hetero.append(value)
                      
                   if all(x==hetero[0] for x in hetero): 
                          homo_orig_6.append(1)
                          homo_list_6.append(orig_cycles_6[l])
                   else: 
                         hetero_list_6.append(orig_cycles_6[l])
                         hetero_orig_6.append(1)
                   

                hetero_6 = []
                homo_6 = []
                for l in range(len(list_cycles_6)):
                   hetero = []
                   for x in list_cycles_6[l]:
                         value = orig_nodes[x]
                         #outfile.write("value", value)
                         hetero.append(value)
                      
                   if all(x==hetero[0] for x in hetero): 
                            homo_6.append(1)
                            
                   else: 
                         hetero_6.append(1)
                         
                hetero_5 = []
                homo_5 = []
                for l in range(len(list_cycles_5)):
                   hetero = []
                   for x in list_cycles_5[l]:
                         value = orig_nodes[x]
                         #outfile.write("value", value)
                         hetero.append(value)
                      
                   if all(x==hetero[0] for x in hetero): 
                            homo_5.append(1)
                            
                   else: 
                         hetero_5.append(1)
                         
                   
                   
                if orig_cycles_6:
                    has_ring6_entailment.append(1)
                else: has_ring6_entailment.append(0)
                
                
                if orig_cycles_6:
                     cycle_6_total = []
                     sum_cycle = []
                     for item in orig_cycles_6:
                         subtotal = 0
                         #print("item", item)
                         for subitem in item:    
                            #print("item", subitem)                   
                            if subitem in G_denoised.nodes:
                               subtotal = subtotal + 1
                            
                         cycle_6_total.append(subtotal)
                     
                     if not cycle_6_total:  sum_cycle.append(0)
                     else: sum_cycle.append(max(cycle_6_total)/6)
                    
                     
                     if not cycle_6_total: has_ring6_fidelity.append(0)
                     else: 
                        
                        has_ring6_fidelity.append(max(sum_cycle))
                   
                     
                if len(homo_orig_6)>0:
                    has_carbon_ring6_entailment.append(1)
                else: has_carbon_ring6_entailment.append(0)
                
                
                if len(homo_orig_6)>0:
                     cycle_6_total = []
                     sum_cycle = []
                     for item in homo_list_6:
                         subtotal = 0
                         #print("item", item)
                         for subitem in item:    
                            #print("item", subitem)                   
                            if subitem in G_denoised.nodes:
                               subtotal = subtotal + 1
                            
                         cycle_6_total.append(subtotal)
                     
                     if not cycle_6_total:  sum_cycle.append(0)
                     else: sum_cycle.append(max(cycle_6_total)/6)
                     
                     if not cycle_6_total: has_carbon_ring6_fidelity.append(0)
                     else: has_carbon_ring6_fidelity.append(max(sum_cycle))
                     


                if len(hetero_orig_6)>0:
                     has_aroma_ring6_entailment.append(1)
                else: has_aroma_ring6_entailment.append(0)
                 

                if len(hetero_orig_6)>0:
                     cycle_6_total = []
                     sum_cycle = []
                     for item in hetero_list_6:
                         subtotal = 0
                       
                         for subitem in item:    
                                     
                            if subitem in G_denoised.nodes:
                               subtotal = subtotal + 1
                            
                         cycle_6_total.append(subtotal)
                     
                     if not cycle_6_total:  sum_cycle.append(0)
                     else: sum_cycle.append(max(cycle_6_total)/6)
                     
                     if not cycle_6_total: has_aroma_ring6_fidelity.append(0)
                     else: has_aroma_ring6_fidelity.append(max(sum_cycle))
 
                if orig_cycles_5:
                    has_ring5_entailment.append(1)
                else: has_ring5_entailment.append(0)
                
                if orig_cycles_5:
                     cycle_5_total = []
                     sum_cycle = []
                     for item in orig_cycles_5:
                         subtotal = 0
                         #print("item", item)
                         for subitem in item:    
                            #print("item", subitem)                   
                            if subitem in G_denoised.nodes:
                               subtotal = subtotal + 1
                            
                         cycle_5_total.append(subtotal)
                     
                     if not cycle_5_total:  sum_cycle.append(0)
                     else: sum_cycle.append(max(cycle_5_total)/5)
                     
                     if not cycle_5_total: has_ring5_fidelity.append(0)
                     else: has_ring5_fidelity.append(max(sum_cycle))
                
                hetero_orig_5 = []
                homo_orig_5 = []
                hetero_list_5 = []
                homo_list_5 = []
                for l in range(len(orig_cycles_5)):
                   hetero = []
                   for x in orig_cycles_5[l]:
                         value = orig_nodes[x]
                         #outfile.write("value", value)
                         hetero.append(value)
                      
                   if all(x==hetero[0] for x in hetero):
                          homo_orig_5.append(1)
                          homo_list_5.append(orig_cycles_5[l])
                   else: 
                         hetero_orig_5.append(1)
                         hetero_list_5.append(orig_cycles_5[l])
                         
                         
                if len(hetero_orig_5)>0:
                    has_aroma_ring5_entailment.append(1)
                else: has_aroma_ring5_entailment.append(0)
                
                 
                if len(hetero_orig_5)>0:
                     cycle_6_total = []
                     sum_cycle = []
                     for item in hetero_list_5:
                         subtotal = 0
                         #print("item", item)
                         for subitem in item:    
                            #print("item", subitem)                   
                            if subitem in G_denoised.nodes:
                               subtotal = subtotal + 1
                            
                         cycle_6_total.append(subtotal)
                     
                     if not cycle_6_total:  sum_cycle.append(0)
                     else: sum_cycle.append(max(cycle_6_total)/5)
                     
                     if not cycle_6_total: has_aroma_ring5_fidelity.append(0)
                     else: has_aroma_ring5_fidelity.append(max(sum_cycle))
                    
                if len(homo_orig_5)>0:
                    has_carbon_ring5_entailment.append(1)
                else: has_carbon_ring5_entailment.append(0)
                
                
                if len(homo_orig_5)>0:
                     cycle_6_total = []
                     sum_cycle = []
                     for item in homo_list_5:
                         subtotal = 0
                         #print("item", item)
                         for subitem in item:    
                            #print("item", subitem)                   
                            if subitem in G_denoised.nodes:
                               subtotal = subtotal + 1
                            
                         cycle_6_total.append(subtotal)
                     
                     if not cycle_6_total:  sum_cycle.append(0)
                     else: sum_cycle.append(max(cycle_6_total)/5)
                     
                     if not cycle_6_total: has_carbon_ring5_fidelity.append(0)
                     else: has_carbon_ring5_fidelity.append(max(sum_cycle))
                     


                if (len(homo_orig_6) + len(hetero_orig_6) + len(homo_orig_5) + len(hetero_orig_5)) >=3:
                        three_rings_entailment.append(1)
                else: three_rings_entailment.append(0)
                
                
                if (len(homo_orig_6) + len(hetero_orig_6) + len(homo_orig_5) + len(hetero_orig_5)) >=3:
                      rings_total = []
                      total_list = orig_cycles_6 + orig_cycles_5
                      rings_total = []
                      length_list = []
                      for l in total_list:
                           subtotal = 0
                           length_list.append(len(l))
                           for subitem in l:                       
                              if subitem in G_denoised.nodes:
                                 subtotal = subtotal + 1
                       
                           rings_total.append(subtotal)
                    
                      if not rings_total:  three_rings_fidelity.append(0)
                      else: 
                           sorted_list = sorted(rings_total)  
                           zipped_lists = zip(rings_total, length_list)
                           sorted_zipped_lists = sorted(zipped_lists)
                           sorted_list1 = [element for length_list, element in sorted_zipped_lists]
                           top_three = sorted_list[-3:]
                           length =  sorted_list1[-3:]
                           three_rings_fidelity.append(sum(top_three)/sum(length))
                

                
                if len(homo_orig_6)>= 3:
                     phenanthrene_entailment.append(1)
                else: phenanthrene_entailment.append(0)

                
                if len(homo_orig_6)>= 3:
                  homo_6_total = []
                  subtotal = 0
                  for l in range(len(orig_cycles_6)):
                      hetero = []
                      for x in orig_cycles_6[l]:

                            value = orig_nodes[x]
                            hetero.append(value)
                      
                      if all(x==hetero[0] for x in hetero):
                          homo_6_total.append(orig_cycles_6[l])
                          
                  
                  #homo_6_total_flat = [item for sublist in homo_6_total for item in sublist]
                  homo_total = []
                  for l in homo_6_total:
                           subtotal = 0
                           length_list.append(len(l))
                           for subitem in l:                       
                              if subitem in G_denoised.nodes:
                                 subtotal = subtotal + 1
                       
                           homo_total.append(subtotal)
                
                  if not homo_total:  phenanthrene_fidelity.append(0)
                  
                  else: 
                           sorted_list = sorted(homo_total)  
                           top_three = sorted_list[-3:]
                           phenanthrene_fidelity.append(sum(top_three)/18)
                       
          
                
                #NH2 or similar may refer to: Azanide (chemical formula NH - 2) Amino radical (chemical formula NH • 2)
                                
                atoms_nh = []
                counter_nh = 0
                for x in range(len(mapped_orig)-1):
                    if (mapped_orig[x-1] == ['N', 'H'] or mapped_orig[x-1] == ['H', 'N']):
                       continue
                    if (mapped_orig[x] == ['N', 'H'] or mapped_orig[x] == ['H', 'N']) and (mapped_orig[x+1] == ['N', 'H'] or mapped_orig[x+1] == ['H', 'N']):
                       atoms_nh.append(x-1)
                       atoms_nh.append(x)
                       atoms_nh.append(x+1)
                       counter_nh = counter_nh + 1
                     
                if counter_nh> 0: azanide_entailment.append(1)
                else: azanide_entailment.append(0)
                
                azanide_list = []
                for number in range(0, len(atoms_nh), 3):
                    azanide_list.append([atoms_nh[number], atoms_nh[number+1], atoms_nh[number+2]])
                    

                azanide_total = []
                if counter_nh > 0:
                   for item in azanide_list:
                       subtotal = 0
                       for subitem in item:                       
                          if subitem in G_denoised.nodes:
                             subtotal = subtotal + 1
                          
                       azanide_total.append(subtotal)
                       #print("methyl total", methyl_total)
                   
                   if not azanide_total:  azanide_fidelity.append(0)
                   else: azanide_fidelity.append(max(azanide_total)/3)
               
                
                counter_nh_final = 0
                for x in range(len(mapped_final)-1):
                    if (mapped_final[x-1] == ['N', 'H'] or mapped_final[x-1] == ['N', 'H']):
                       continue
                    if (mapped_final[x] == ['N', 'H'] or mapped_final[x] == ['H', 'N']) and (mapped_final[x+1] == ['N', 'H'] or mapped_final[x+1] == ['H', 'N']):
                       counter_nh_final = counter_nh_final + 1
                       
                
                atoms_nitro = []
                number_nitro_orig = 0
                for x in range(len(mapped_orig)-1):
                    if (mapped_orig[x-1] == ['N', 'O'] or mapped_orig[x-1] == ['N', 'O']):
                       continue
                    if (mapped_orig[x] == ['N', 'O'] or mapped_orig[x] == ['O', 'N']) and (mapped_orig[x+1] == ['N', 'O'] or mapped_orig[x+1] == ['O', 'N']):
                       atoms_nitro.append(x-1)
                       atoms_nitro.append(x)
                       atoms_nitro.append(x+1)
                       number_nitro_orig = number_nitro_orig + 1
                      
                if number_nitro_orig > 0: nitro_entailment.append(1)
                else: nitro_entailment.append(0)
                
                nitro_list = []
                for number in range(0, len(atoms_nitro), 3):
                    nitro_list.append([atoms_nitro[number], atoms_nitro[number+1], atoms_nitro[number+2]])
                    

                nitro_total = []
                if number_nitro_orig > 0:
                   for item in nitro_list:
                       subtotal = 0
                       for subitem in item:                       
                          if subitem in G_denoised.nodes:
                             subtotal = subtotal + 1
                          
                       nitro_total.append(subtotal)
                       #print("methyl total", methyl_total)
                   
                   if not nitro_total:  nitro_fidelity.append(0)
                   else: nitro_fidelity.append(max(nitro_total)/3)
                          
               
                number_nitro_final = 0
                for x in range(len(mapped_final)-1):
                    if (mapped_final[x-1] == ['N', 'O'] or mapped_final[x-1] == ['N', 'O']):
                       continue
                    if (mapped_final[x] == ['N', 'O'] or mapped_final[x] == ['O', 'N']) and (mapped_final[x+1] == ['N', 'O'] or mapped_final[x+1] == ['O', 'N']):
                       number_nitro_final = number_nitro_final + 1
       
               
    
                #methyl             CH3
                
                atoms_methyl = []
                number_methyl_orig = 0
                for x in range(len(mapped_orig)-2):
                    if (mapped_orig[x-1] == ['C', 'H'] or mapped_orig[x-1] == ['H', 'C']):
                       continue
                    if (mapped_orig[x] == ['C', 'H'] or mapped_orig[x] == ['H', 'C']) and (mapped_orig[x+1] == ['C', 'H'] or mapped_orig[x+1] == ['H', 'C']) and (mapped_orig[x+2] == ['C', 'H'] or mapped_orig[x+2] == ['H', 'C']):
                       atoms_methyl.append(x-1)
                       atoms_methyl.append(x)
                       atoms_methyl.append(x+1)
                       atoms_methyl.append(x+2)
                       number_methyl_orig = number_methyl_orig + 1
                     
                if number_methyl_orig > 0: methyl_entailment.append(1)
                else: 
                      methyl_entailment.append(0)
                      
                
                methyl_list = []
                for number in range(0, len(atoms_methyl), 4):
                    methyl_list.append([atoms_methyl[number], atoms_methyl[number+1], atoms_methyl[number+2], atoms_methyl[number+3]])

                methyl_total = []
                if number_methyl_orig > 0:
                  for item in methyl_list:
                      subtotal = 0
                      for subitem in item:                       
                         if subitem in G_denoised.nodes:
                            subtotal = subtotal + 1
                         
                      methyl_total.append(subtotal)
                      #print("methyl total", methyl_total)
                  
                  if not methyl_total:  
                            methyl_fidelity.append(0)
                            fidelity_methyl = 0
                  else: 
                         methyl_fidelity.append(max(methyl_total)/4)
                         fidelity_methyl = max(methyl_total)/4
                  
             

                number_methyl_final = 0
                for x in range(len(mapped_final)-2):
                    if (mapped_final[x-1] == ['C', 'H'] or mapped_final[x-1] == ['H', 'C']):
                       continue
                    if (mapped_final[x] == ['C', 'H'] or mapped_final[x] == ['H', 'C']) and (mapped_final[x+1] == ['C', 'H'] or mapped_final[x+1] == ['H', 'C']) and (mapped_final[x+2] == ['C', 'H'] or mapped_final[x+2] == ['H', 'C']):
                       number_methyl_final = number_methyl_final + 1
                       


                if methyl_entailment[number_instances-1] == 1 and nitro_entailment[number_instances-1] == 0 and azanide_entailment[number_instances-1] == 0 and phenanthrene_entailment[number_instances-1] == 0 and has_ring5_entailment[number_instances-1] == 0 and has_ring6_entailment[number_instances-1] == 0:
                     only_methyl_entailment.append(1)
                else: only_methyl_entailment.append(0)
                #Phenanthrene/anthracene
                 
                if methyl_entailment[number_instances-1] == 1:
                    if sulfur_entailment[number_instances-1] == 1:
                        methyl_phosul_entailment.append(1)
                        methyl_phosul_fidelity.append((fidelity_methyl + fidelity_sulfur)/2)
                        
                    elif phosphorus_entailment[number_instances-1] ==1:
                        methyl_phosul_entailment.append(1)
                        methyl_phosul_fidelity.append((fidelity_methyl + fidelity_phosphorus)/2)
                    else:
                       methyl_phosul_entailment.append(0)
                else: 
                    methyl_phosul_entailment.append(0)    
                    
                if methyl_entailment[number_instances-1] == 1:
                    if sulfur_entailment[number_instances-1] == 1:
                        sulfur_methyl_entailment.append(1)
                        sulfur_methyl_fidelity.append((fidelity_methyl + fidelity_sulfur)/2)
                        
                    else:
                       sulfur_methyl_entailment.append(0)
                else: 
                    sulfur_methyl_entailment.append(0)  
                
   
    
    
                outfile.write("<!-- http://dl-learner.org/mutagenesis#in" + str(graph_idx) +" -->\n")      
                outfile.write('<owl:NamedIndividual rdf:about="http://dl-learner.org/mutagenesis#in'+str(graph_idx)+'">\n')
                outfile.write('<rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Compound"/>')
                
                #print("edges", G_orig.edges) 
              
                for atom in range(len(orig_nodes)):
                      outfile.write('<hasAtom rdf:resource="http://dl-learner.org/mutagenesis#in'+str(graph_idx)+'_'+str(atom)+'"/>\n')
                      
                      for edge in G_orig.edges:
                         if edge[0]==atom:
                          second_atom = edge[1]
                          outfile.write('<hasBond rdf:resource="http://dl-learner.org/mutagenesis#in'+str(graph_idx)+'_'+str(atom)+'_'+str(second_atom)+'"/>\n')
                         elif edge[1]==atom:
                          second_atom = edge[1]
                          first_atom = edge[0]
                          outfile.write('<hasBond rdf:resource="http://dl-learner.org/mutagenesis#in'+str(graph_idx)+'_'+str(first_atom)+'_'+str(second_atom)+'"/>\n')
                          

                #for atom in range(len(orig_nodes)):
                 #    if important_atoms[atom] == 1:
                  #         outfile.write('<hasImportantAtom rdf:resource="http://dl-learner.org/mutagenesis#in'+str(graph_idx)+'_'+str(atom)+'"/>')
                      
            
                for m in range(number_nitro_orig):
                    outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#nitro-'+str(graph_idx)+'_'+str(m)+'"/>\n')
                    
                for n in range(number_methyl_orig):
                    outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#methyl-'+str(graph_idx)+'_'+str(n)+'"/>\n')
                    
                for h in range(counter_nh):
                    outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#azanide-'+str(graph_idx)+'_'+str(h)+'"/>\n')
                             
                                  
                for cycle in range(len(homo_orig_6)):
                    outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#carbon_6_ring_in'+str(graph_idx)+'_'+str(cycle)+'"/>\n')
                    
                for cycle in range(len(hetero_orig_6)):
                    outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#hetero_aromatic_6_ring_in'+str(graph_idx)+'_'+str(cycle)+'"/>\n')
                    
    
                if len(homo_orig_6)>= 3:
                     outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#phenanthrene_'+str(graph_idx)+'"/>\n')
                    
                for cycle in range(len(homo_orig_5)):
                    outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#carbon_5_ring_in'+str(graph_idx)+'_'+str(cycle)+'"/>\n')
                    
                for cycle in range(len(hetero_orig_5)):
                    outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#hetero_aromatic_5_ring_in'+str(graph_idx)+'_'+str(cycle)+'"/>\n')
                    
    
                
                #outfile.write('<hasFifeExamplesOfAcenthrylenes rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">false</hasFifeExamplesOfAcenthrylenes>\n')
                if (len(homo_orig_6) + len(hetero_orig_6) + len(homo_orig_5) + len(hetero_orig_5)) >=3:
                            outfile.write('<hasThreeOrMoreFusedRings rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</hasThreeOrMoreFusedRings>\n')
                else:
                            outfile.write('<hasThreeOrMoreFusedRings rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">false</hasThreeOrMoreFusedRings>\n')
                
                
                #outfile.write('<hasFifeExamplesOfAcenthrylenes rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">false</hasFifeExamplesOfAcenthrylenes>\n')
                if label == 1: 
                            outfile.write('<mutagenic rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">false</mutagenic>\n')
                            if pred_label == 1:
                                       outfile.write('<prediction rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</prediction>\n')
                            elif pred_label == 0: 
                                       outfile.write('<prediction rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">false</prediction>\n')
                elif label == 0:
                            outfile.write('<mutagenic rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</mutagenic>\n')
                            if pred_label == 0:
                                       outfile.write('<prediction rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</prediction>\n')
                            elif pred_label == 1: 
                                       outfile.write('<prediction rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">false</prediction>\n')  
                    
                outfile.write('</owl:NamedIndividual>\n')
                          
    
                
                outfile.write("<!-- http://dl-learner.org/mutagenesis#in" + graph_idx_sub +" -->\n")
      
                outfile.write('<owl:NamedIndividual rdf:about="http://dl-learner.org/mutagenesis#in'+graph_idx_sub+'">\n')
                outfile.write('<rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Compound"/>\n')
              
               
                for atom in range(len(orig_nodes)):
                     if important_atoms[atom] == 1:

                           if dict_feat[orig_nodes[atom]] == 1:
                                  #print(orig_nodes[atom])
                                  outfile.write('<hasAtom rdf:resource="http://dl-learner.org/mutagenesis#in'+str(graph_idx)+'_'+str(atom)+'"/>\n')
                           
                           for edge in G_denoised.edges:
                              if edge[0]==atom:
                               
                               second_atom = edge[1]
                               outfile.write('<hasBond rdf:resource="http://dl-learner.org/mutagenesis#in'+str(graph_idx)+'_'+str(atom)+'_'+str(second_atom)+'"/>\n')
                              elif edge[1]==atom:
                               
                               second_atom = edge[1]
                               first_atom = edge[0]
                               outfile.write('<hasBond rdf:resource="http://dl-learner.org/mutagenesis#in'+str(graph_idx)+'_'+str(first_atom)+'_'+str(second_atom)+'"/>\n')
                               
                  
                for cycle in range(len(homo_6)):
                    outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#carbon_6_ring_in'+str(graph_idx)+'_'+str(cycle)+'"/>\n')
                    
                for cycle in range(len(hetero_6)):
                    outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#hetero_aromatic_6_ring_in'+str(graph_idx)+'_'+str(cycle)+'"/>\n')
                 
                for cycle in range(len(homo_5)):
                    outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#carbon_5_ring_in'+str(graph_idx)+'_'+str(cycle)+'"/>\n')
                    
                for cycle in range(len(hetero_5)):
                    outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#hetero_aromatic_5_ring_in'+str(graph_idx)+'_'+str(cycle)+'"/>\n')
                    
                for m in range(number_nitro_final):
                    outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#nitro-'+str(graph_idx)+'_'+str(m)+'"/>\n')
                    
                for n in range(number_methyl_final):
                    outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#methyl-'+str(graph_idx)+'_'+str(n)+'"/>\n')
                    
                for h in range(counter_nh_final):
                    outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#azanide-'+str(graph_idx)+'_'+str(h)+'"/>\n')
                             
                         
         
                if len(homo_6)>= 3:
                     outfile.write('<hasStructure rdf:resource="http://dl-learner.org/mutagenesis#phenanthrene_'+str(graph_idx)+'"/>\n')
                    
    
                #outfile.write('<hasFifeExamplesOfAcenthrylenes rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">false</hasFifeExamplesOfAcenthrylenes>\n')
                if (len(homo_6) + len(hetero_6) + len(homo_5) + len(hetero_5)) >=3:
                            outfile.write('<hasThreeOrMoreFusedRings rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</hasThreeOrMoreFusedRings>\n')
                else:
                            outfile.write('<hasThreeOrMoreFusedRings rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">false</hasThreeOrMoreFusedRings>\n')
                            
                if label == 1: 
                            outfile.write('<mutagenic rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">false</mutagenic>\n')
                            if pred_label == 1:
                                       outfile.write('<prediction rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</prediction>\n')
                            elif pred_label == 0: 
                                       outfile.write('<prediction rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">false</prediction>\n')
                elif label == 0:
                            outfile.write('<mutagenic rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</mutagenic>\n')
                            if pred_label == 0:
                                       outfile.write('<prediction rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</prediction>\n')
                            elif pred_label == 1: 
                                       outfile.write('<prediction rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">false</prediction>\n')  
                    
                           
                outfile.write('</owl:NamedIndividual>\n')
                #Nitro 
                
                for a in range(len(orig_nodes)):
                
                    outfile.write('<!-- http://dl-learner.org/mutagenesis#in'+str(graph_idx)+'_'+str(a)+' -->\n')
                    outfile.write("\n")
        
                    outfile.write('<owl:NamedIndividual rdf:about="http://dl-learner.org/mutagenesis#in'+str(graph_idx)+'_'+str(a)+'">\n')
                    if orig_nodes[a] == 'C':
                            outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Carbon"/>\n')
                    elif orig_nodes[a] == 'N':
                            outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Nitrogen"/>\n')
                    elif orig_nodes[a] == 'O':
                            outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Oxygen"/>\n')
                    elif orig_nodes[a] == 'F':
                            outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Fluorine"/>\n')
                    elif orig_nodes[a] == 'I':
                            outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Iodine"/>\n')
                    elif orig_nodes[a] == 'Cl':
                            outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Chlorine"/>\n')
                    elif orig_nodes[a] == 'Br':
                            outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Bromine"/>\n')
                    elif orig_nodes[a] == 'P':
                            outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Posphorus"/>\n')
                    elif orig_nodes[a] == 'S':
                            outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Sulfur"/>\n')
                    elif orig_nodes[a] == 'Na':
                            outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Sodium"/>\n')
                    elif orig_nodes[a] == 'K':
                            outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Potassium"/>\n')
                    elif orig_nodes[a] == 'Li':
                            outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Lithium"/>\n')
                    elif orig_nodes[a] == 'H':
                            outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Hydrogen"/>\n')
                    elif orig_nodes[a] == 'Ca':
                            outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Calcium"/>\n')
       
                    methyl_sum = []
                    if important_atoms[a] == 1 :
                          outfile.write('     <isImportant rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</isImportant>\n')
                          
               

                              
                          
                    if a in atoms_methyl:
                       if atoms_methyl.index(a) < 4: methyl_number = 0
                       elif atoms_methyl.index(a) < 8 and atoms_methyl.index(a) > 3 : methyl_number = 1
                       elif atoms_methyl.index(a) < 12 and atoms_methyl.index(a) > 7 : methyl_number = 2
                       elif atoms_methyl.index(a) < 16 and atoms_methyl.index(a) > 11 : methyl_number = 3
                       outfile.write('     <inStructure rdf:resource="http://dl-learner.org/mutagenesis#methyl-'+str(graph_idx)+'_'+str(methyl_number)+'"/>\n')
                       
                       
                    if a in atoms_nitro:
                       if atoms_nitro.index(a) < 3: nitro_number = 0
                       elif atoms_nitro.index(a) < 6 and atoms_nitro.index(a) > 2 : nitro_number = 1
                       elif atoms_nitro.index(a) < 9 and atoms_nitro.index(a) > 5 : nitro_number = 2
                       elif atoms_nitro.index(a) < 12 and atoms_nitro.index(a) > 8 : nitro_number = 3
                       outfile.write('     <inStructure rdf:resource="http://dl-learner.org/mutagenesis#nitro-'+str(graph_idx)+'_'+str(nitro_number)+'"/>\n')


                    if a in atoms_nh:
                       if atoms_nh.index(a) < 3: nh_number = 0
                       elif atoms_nh.index(a) < 6 and atoms_nh.index(a) > 2 : nh_number = 1
                       elif atoms_nh.index(a) < 9 and atoms_nh.index(a) > 5 : nh_number = 2
                       elif atoms_nh.index(a) < 12 and atoms_nh.index(a) > 8 : nh_number = 3
                       outfile.write('     <inStructure rdf:resource="http://dl-learner.org/mutagenesis#nitro-'+str(graph_idx)+'_'+str(nh_number)+'"/>\n')

                    outfile.write('</owl:NamedIndividual>\n')
                	
                 
                 
                if len(homo_orig_6)>= 3:
                    outfile.write('<!-- http://dl-learner.org/mutagenesis#phenanthrene_'+str(graph_idx)+' -->\n')
                    outfile.write("\n")
                    outfile.write('<owl:NamedIndividual rdf:about="http://dl-learner.org/mutagenesis#phenanthrene_'+str(graph_idx)+'">\n')
                    outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Phenanthrene"/>\n')
                    if len(homo_6)>= 3:
                                         outfile.write('     <isImportant rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</isImportant>\n')                
                    outfile.write('</owl:NamedIndividual>')
                    
                 
                                                
                for m in range(number_nitro_orig):
    
                    outfile.write('<!-- http://dl-learner.org/mutagenesis#nitro-'+str(graph_idx)+'_'+str(m)+' -->\n')
                    outfile.write("\n")
                    outfile.write('<owl:NamedIndividual rdf:about="http://dl-learner.org/mutagenesis#nitro-'+str(graph_idx)+'_'+str(m)+'">\n')
                    outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Nitro"/>\n')
                    if number_nitro_final <= m: 
                                         outfile.write('     <isImportant rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</isImportant>\n')                
                    outfile.write('</owl:NamedIndividual>')
                    
    
                for m2 in range(number_methyl_orig):
    
                    outfile.write('<!-- http://dl-learner.org/mutagenesis#nmethyl-'+str(graph_idx)+'_'+str(m2)+' -->\n')
                    outfile.write("\n")
                    outfile.write('<owl:NamedIndividual rdf:about="http://dl-learner.org/mutagenesis#methyl-'+str(graph_idx)+'_'+str(m2)+'">\n')
                    outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Methyl"/>\n')
                    if number_methyl_final <= m2: 
                                         outfile.write('     <isImportant rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</isImportant>\n')                
                    outfile.write('</owl:NamedIndividual>')
                    
                    
                for a in range(counter_nh):
    
                    outfile.write('<!-- http://dl-learner.org/mutagenesis#azanide-'+str(graph_idx)+'_'+str(a)+' -->\n')
                    outfile.write("\n")
                    outfile.write('<owl:NamedIndividual rdf:about="http://dl-learner.org/mutagenesis#azanide-'+str(graph_idx)+'_'+str(a)+'">\n')
                    outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Azanide"/>\n')
                    if counter_nh_final <= a: 
                                         outfile.write('     <isImportant rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</isImportant>\n')                
                    outfile.write('</owl:NamedIndividual>')
                    
    
                                   
                for cycle1 in range(len(homo_orig_6)):
                    outfile.write('<!-- http://dl-learner.org/mutagenesis#carbon_6_ring_in'+str(graph_idx)+'_'+str(cycle1)+' -->\n')
                    outfile.write("\n")
                    outfile.write('<owl:NamedIndividual rdf:about="http://dl-learner.org/mutagenesis#carbon_6_ring_in'+str(graph_idx)+'_'+str(cycle1)+'">\n')
                    outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Carbon_6_ring"/>\n')
                    if len(homo_6) <= cycle1: 
                                         outfile.write('     <isImportant rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</isImportant>\n')
                    outfile.write('</owl:NamedIndividual>\n')
                    
                    
                for cycle2 in range(len(hetero_orig_6)):
                    outfile.write('<!-- http://dl-learner.org/mutagenesis#hetero_aromatic_6_ring_in'+str(graph_idx)+'_'+str(cycle2)+' -->\n')
                    outfile.write("\n")
                    outfile.write('<owl:NamedIndividual rdf:about="http://dl-learner.org/mutagenesis#hetero_aromatic_6_ring_in'+str(graph_idx)+'_'+str(cycle2)+'">\n')
                    outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Hetero_aromatic_6_ring"/>\n')
                    if len(hetero_6) <= cycle2: 
                                         outfile.write('     <isImportant rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</isImportant>\n')
                    outfile.write('</owl:NamedIndividual>\n')
    
    
                for cycle3 in range(len(homo_orig_5)):
                    outfile.write('<!-- http://dl-learner.org/mutagenesis#carbon_5_ring_in'+str(graph_idx)+'_'+str(cycle3)+' -->\n')
                    outfile.write("\n")
                    outfile.write('<owl:NamedIndividual rdf:about="http://dl-learner.org/mutagenesis#carbon_5_ring_in'+str(graph_idx)+'_'+str(cycle3)+'">\n')
                    outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Carbon_5_ring"/>\n')
                    if len(homo_5) <= cycle3: 
                                         outfile.write('     <isImportant rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</isImportant>\n')
                    outfile.write('</owl:NamedIndividual>\n')
                    
                    
                for cycle4 in range(len(hetero_orig_5)):
                    outfile.write('<!-- http://dl-learner.org/mutagenesis#hetero_aromatic_5_ring_in'+str(graph_idx)+'_'+str(cycle4)+' -->\n')
                    outfile.write("\n")
                    outfile.write('<owl:NamedIndividual rdf:about="http://dl-learner.org/mutagenesis#hetero_aromatic_5_ring_in'+str(graph_idx)+'_'+str(cycle4)+'">\n')
                    outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Hetero_aromatic_5_ring"/>\n')
                    if len(hetero_5) <= cycle4: 
                                         outfile.write('     <isImportant rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</isImportant>\n')
                    outfile.write('</owl:NamedIndividual>\n')
                	
         
                 
                for edge in G_orig.edges:
                         atom = edge[0]
                         second_atom = edge[1]
                         outfile.write('<!-- http://dl-learner.org/mutagenesis#in'+str(graph_idx)+'_'+str(atom)+'_'+str(second_atom)+' -->\n')
                         outfile.write("\n")
                         outfile.write('<owl:NamedIndividual rdf:about="http://dl-learner.org/mutagenesis#in'+str(graph_idx)+'_'+str(atom)+'_'+str(second_atom)+'">\n')
                         outfile.write('     <rdf:type rdf:resource="http://dl-learner.org/mutagenesis#Bond"/>\n')
                         outfile.write('     <inBond rdf:resource="http://dl-learner.org/mutagenesis#in'+str(graph_idx)+'_'+str(atom)+'"/>\n')
                         outfile.write('     <inBond rdf:resource="http://dl-learner.org/mutagenesis#in'+str(graph_idx)+'_'+str(second_atom)+'"/>\n')
                         outfile.write('</owl:NamedIndividual>\n')
                          

    
                masked_adjs.append(masked_adj)
               
                
                io_utils.log_graph(
                    self.writer,
                    G_orig,
                    "graph/graphidx_{}".format(graph_idx),
                    identify_self=False,
                    nodecolor="feat",
                    args=self.args
                )
                
    
            # plot cmap for graphs' node features
            io_utils.plot_cmap_tb(self.writer, "tab20", 20, "tab20_cmap")
            #outfile.write("5er cycles", cycle_aromatic)
            
            #close the file
            
        print("-----------############################------------------")
        print("Total Instances", number_instances)
        print("-----------IO Classes Mutagenic Prediction------------------")
        #print("Class: has ThreeOrMoreFusedRings")
        #print("Entailment Percentage", round(sum(three_rings_entailment)/number_instances,2))
        #print("three rings entailment", three_rings_entailment)
        #print("three rings_fidelity", three_rings_fidelity)
        #if len(three_rings_fidelity) > 0:
        #     print("Average Fidelity", round(sum(three_rings_fidelity)/len(three_rings_fidelity), 2))
        print("   ")
        print("Class: hasStructure some Nitro")
        print("Entailment Percentage", round(sum(nitro_entailment)/number_instances,2))
        #print("nitro entailment", nitro_entailment)
        #print("nitro_fidelity", nitro_fidelity)
        if len(nitro_fidelity) > 0:
             print("Average Fidelity", round(sum(nitro_fidelity)/len(nitro_fidelity), 2))
        print("   ")
        print("Class: hasStructure some Phenanthrene")
        print("Entailment Percentage", round(sum(phenanthrene_entailment)/number_instances,2))
        #print("Phenanthrene entailment", phenanthrene_entailment)
        #print("Phenanthrene fidelity", phenanthrene_fidelity)
        if len(phenanthrene_fidelity) >0 :
            print("Average Fidelity", round(sum(phenanthrene_fidelity)/len(phenanthrene_fidelity), 2))
        print("   ")
        
        print("   ")
        print("Class: hasStructure some Ring_size 5")
        print("Entailment Percentage", round(sum(has_ring5_entailment)/number_instances,2))
        if len(has_ring5_fidelity) >0 :
            print("Average Fidelity", round(sum(has_ring5_fidelity)/len(has_ring5_fidelity), 2))
        print("   ")

        print("Class: hasStructure some Ring_size 6")
        print("Entailment Percentage", round(sum(has_ring6_entailment)/number_instances,2))
        if len(has_ring6_fidelity) >0 :
            print("Average Fidelity", round(sum(has_ring6_fidelity)/len(has_ring6_fidelity), 2))
        print("   ")
        
        print("Class: hasStructure some carbon Ring_size 6")
        print("Entailment Percentage", round(sum(has_carbon_ring6_entailment)/number_instances,2))
        if len(has_carbon_ring6_fidelity) >0 :
            print("Average Fidelity", round(sum(has_carbon_ring6_fidelity)/len(has_carbon_ring6_fidelity), 2))
        print("   ")
        
        
        
        
        print("-----------I Classes Mutagenic Prediction------------------")
        #print("phosphorus entailment", phosphorus_entailment)
        #print("phosphorus_fidelity", phosphorus_fidelity)
        print("Class: hasAtom Oxygen")
        print("Entailment Percentage", round(sum(oxygen_entailment)/number_instances,2))
        if len(oxygen_fidelity) > 0:
           print("Average Fidelity", round(sum(oxygen_fidelity)/len(oxygen_fidelity), 2))
        print("   ")
        print("Class: hasAtom hydrogen")
        print("Entailment Percentage", round(sum(hydrogen_entailment)/number_instances,2))
        if len(hydrogen_fidelity) > 0:
           print("Average Fidelity", round(sum(hydrogen_fidelity)/len(hydrogen_fidelity), 2))
        print("   ")
        print("Class: hasAtom carbon")
        print("Entailment Percentage", round(sum(carbon_entailment)/number_instances,2))
        if len(carbon_fidelity) > 0:
           print("Average Fidelity", round(sum(carbon_fidelity)/len(carbon_fidelity), 2))
        print("   ")
        print("Class: hasAtom nitrogen")
        print("Entailment Percentage", round(sum(nitrogen_entailment)/number_instances,2))
        if len(nitrogen_fidelity) > 0:
           print("Average Fidelity", round(sum(nitrogen_fidelity)/len(nitrogen_fidelity), 2))
        print("   ")

        
        print("-----------Instance Information------------------")
        print("   ")
        sum_entailments = sum(nitro_entailment) + sum(phenanthrene_entailment) + sum(has_ring5_entailment) +sum(has_ring6_entailment)+sum(has_carbon_ring6_entailment)+sum(oxygen_entailment)+sum(hydrogen_entailment)+sum(nitrogen_entailment)+sum(carbon_entailment)
        print("sum entailments", sum_entailments)
        print("Average Entailments per Instance", round(sum_entailments/number_instances, 2))
        
        print("   ")
        no_entailment = 0
        for i in range(len(nitro_entailment)):
           if has_ring5_entailment[i] == 0 and nitro_entailment[i] == 0 and phenanthrene_entailment[i] == 0 and has_ring6_entailment[i] == 0 and has_carbon_ring6_entailment[i] ==0            and oxygen_entailment[i] ==0 and hydrogen_entailment[i] ==0 and nitrogen_entailment[i] ==0 and carbon_entailment[i] ==0:
                  no_entailment = no_entailment + 1
           
        print("Percentage No Entailment:", round(no_entailment/number_instances, 2))
        

        print("   ")
        print("-----------IO Classes NonMutagenic Prediction------------------")
        print("Class: hasAtom Phosphorus")
        print("Entailment Percentage", round(sum(phosphorus_entailment)/number_instances,2))
        if len(phosphorus_fidelity) > 0:
           print("Average Fidelity", round(sum(phosphorus_fidelity)/len(phosphorus_fidelity), 2))
        print("   ")
        print("Class: hasStructure some carbon Ring_size 5")
        print("Entailment Percentage", round(sum(has_carbon_ring5_entailment)/number_instances,2))
        if len(has_carbon_ring5_fidelity) >0 :
            print("Average Fidelity", round(sum(has_carbon_ring5_fidelity)/len(has_carbon_ring5_fidelity), 2))
        print("   ")
        print("Class: hasStructure ONLY Methyl")
        print("Entailment Percentage", round(sum(only_methyl_entailment)/number_instances,2))
        if len(only_methyl_entailment)>0:
            print("Average Fidelity", round(sum(methyl_fidelity)/len(methyl_fidelity), 2))
        print("   ")      
        print("Class: hasAtom some (Posphorus or Sulfur) and hasStructure some Methyl")
        print("Entailment Percentage", round(sum(methyl_phosul_entailment)/number_instances,2))
        if len(methyl_phosul_entailment)>0:
            print("Average Fidelity", round(sum(methyl_phosul_fidelity)/len(methyl_phosul_fidelity), 2))
        print("   ")    
        print("Class: hasAtom some Sulfur and hasStructure some Methyl")
        print("Entailment Percentage", round(sum(sulfur_methyl_entailment)/number_instances,2))
        if len(sulfur_methyl_entailment)>0:
            print("Average Fidelity", round(sum(sulfur_methyl_fidelity)/len(sulfur_methyl_fidelity), 2))
        
        print("-----------I Classes NonMutagenic Prediction------------------")
        print("Class: hasStructure Methyl")
        print("Entailment Percentage", round(sum(methyl_entailment)/number_instances,2))
        if len(methyl_entailment)>0:
            print("Average Fidelity", round(sum(methyl_fidelity)/len(methyl_fidelity), 2))
        print("   ")
        print("Class: hasStructure Azanide")
        print("Entailment Percentage", round(sum(azanide_entailment)/number_instances,2))
        if len(azanide_entailment)>0:
            print("Average Fidelity", round(sum(azanide_fidelity)/len(azanide_fidelity), 2))
        
        print("   ")
        print("Class: hasStructure some Ring_size 5")
        print("Entailment Percentage", round(sum(has_ring5_entailment)/number_instances,2))
        if len(has_ring5_fidelity) >0 :
            print("Average Fidelity", round(sum(has_ring5_fidelity)/len(has_ring5_fidelity), 2))
        print("   ")
        print("   ")
        print("Class: hasStructure some Ring_size 6")
        print("Entailment Percentage", round(sum(has_ring6_entailment)/number_instances,2))
        if len(has_ring6_fidelity) >0 :
            print("Average Fidelity", round(sum(has_ring6_fidelity)/len(has_ring6_fidelity), 2))
        print("   ")
        print("Class: hasStructure some carbon Ring_size 6")
        print("Entailment Percentage", round(sum(has_carbon_ring6_entailment)/number_instances,2))
        if len(has_carbon_ring6_fidelity) >0 :
            print("Average Fidelity", round(sum(has_carbon_ring6_fidelity)/len(has_carbon_ring6_fidelity), 2))
        print("   ")
        print("Class: hasStructure some hetero Ring_size 5")
        print("Entailment Percentage", round(sum(has_aroma_ring5_entailment)/number_instances,2))
        if len(has_aroma_ring5_fidelity) >0 :
            print("Average Fidelity", round(sum(has_aroma_ring5_fidelity)/len(has_aroma_ring5_fidelity), 2))
        print("   ")
        print("Class: hasStructure some hetero Ring_size 6")
        print("Entailment Percentage", round(sum(has_aroma_ring6_entailment)/number_instances,2))
        if len(has_aroma_ring6_fidelity) >0 :
            print("Average Fidelity", round(sum(has_aroma_ring6_fidelity)/len(has_aroma_ring6_fidelity), 2))
        

        print("-----------Instance Information------------------")
        print("   ")
        sum_entailment_non = sum(phosphorus_entailment) + sum(has_carbon_ring5_entailment) + sum(only_methyl_entailment) + sum(sulfur_methyl_entailment) + sum(methyl_phosul_entailment) + sum(has_aroma_ring6_entailment) + sum(has_aroma_ring5_entailment) + sum(has_carbon_ring6_entailment) + sum(has_ring6_entailment) + sum(has_ring5_entailment) + sum(azanide_entailment) + sum(methyl_entailment)
        print("sum entailment", sum_entailment_non)
        print("Average Entailments per Instance", round(sum_entailment_non / number_instances, 2))
        print("   ")
        no_entailment = 0
        for i in range(len(methyl_entailment)):
           if phosphorus_entailment[i] == 0 and  has_carbon_ring5_entailment[i] == 0 and only_methyl_entailment[i] == 0 and sulfur_methyl_entailment[i] == 0 and methyl_phosul_entailment[i] == 0 and has_aroma_ring6_entailment[i] == 0 and has_aroma_ring5_entailment[i] == 0 and has_carbon_ring6_entailment[i] == 0 and has_ring6_entailment[i] == 0 and has_ring5_entailment[i] == 0 and azanide_entailment[i] == 0 and methyl_entailment[i] == 0:
                  no_entailment = no_entailment + 1
           
        print("Percentage No Entailment:", round(no_entailment/number_instances, 2))
        print("   ")

        outfile.close()
        return masked_adjs

    def log_representer(self, rep_val, sim_val, alpha, graph_idx=0):
        """ visualize output of representer instances. """
        rep_val = rep_val.cpu().detach().numpy()
        sim_val = sim_val.cpu().detach().numpy()
        alpha = alpha.cpu().detach().numpy()
        sorted_rep = sorted(range(len(rep_val)), key=lambda k: rep_val[k])
        outfile.write(sorted_rep)
        topk = 5
        most_neg_idx = [sorted_rep[i] for i in range(topk)]
        most_pos_idx = [sorted_rep[-i - 1] for i in range(topk)]
        rep_idx = [most_pos_idx, most_neg_idx]

        if self.graph_mode:
            pred = np.argmax(self.pred[0][graph_idx], axis=0)
        else:
            pred = np.argmax(self.pred[graph_idx][self.train_idx], axis=1)
        outfile.write(metrics.confusion_matrix(self.label[graph_idx][self.train_idx], pred))
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(5, 3), dpi=600)
        for i in range(2):
            for j in range(topk):
                idx = self.train_idx[rep_idx[i][j]]
                outfile.write(
                    "node idx: ",
                    idx,
                    "; node label: ",
                    self.label[graph_idx][idx],
                    "; pred: ",
                    pred,
                )

                idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                    idx, graph_idx
                )
                G = nx.from_numpy_matrix(sub_adj)
                node_colors = [1 for i in range(G.number_of_nodes())]
                node_colors[idx_new] = 0
                # node_color='#336699',

                ax = plt.subplot(2, topk, i * topk + j + 1)
                nx.draw(
                    G,
                    pos=nx.spring_layout(G),
                    with_labels=True,
                    font_size=4,
                    node_color=node_colors,
                    cmap=plt.get_cmap("Set1"),
                    vmin=0,
                    vmax=8,
                    edge_vmin=0.0,
                    edge_vmax=1.0,
                    width=0.5,
                    node_size=25,
                    alpha=0.7,
                )
                ax.xaxis.set_visible(False)
        fig.canvas.draw()
        self.writer.add_image(
            "local/representer_neigh", tensorboardX.utils.figure_to_image(fig), 0
        )

    def representer(self):
        """
        experiment using representer theorem for finding supporting instances.
        https://papers.nips.cc/paper/8141-representer-point-selection-for-explaining-deep-neural-networks.pdf
        """
        self.model.train()
        self.model.zero_grad()
        adj = torch.tensor(self.adj, dtype=torch.float)
        x = torch.tensor(self.feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(self.label, dtype=torch.long)
        if self.args.gpu:
            adj, x, label = adj.cuda(), x.cuda(), label.cuda()

        preds, _ = self.model(x, adj)
        preds.retain_grad()
        self.embedding = self.model.embedding_tensor
        loss = self.model.loss(preds, label)
        loss.backward()
        self.preds_grad = preds.grad
        pred_idx = np.expand_dims(np.argmax(self.pred, axis=2), axis=2)
        pred_idx = torch.LongTensor(pred_idx)
        if self.args.gpu:
            pred_idx = pred_idx.cuda()
        self.alpha = self.preds_grad


    # Utilities
    def extract_neighborhood(self, node_idx, graph_idx=0):
        """Returns the neighborhood of a given ndoe."""
        neighbors_adj_row = self.neighborhoods[graph_idx][node_idx, :]
        # index of the query node in the new adj
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        sub_adj = self.adj[graph_idx][neighbors][:, neighbors]
        sub_feat = self.feat[graph_idx, neighbors]
        sub_label = self.label[graph_idx][neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

    def align(
        self, ref_feat, ref_adj, ref_node_idx, curr_feat, curr_adj, curr_node_idx, args
    ):
        """ Tries to find an alignment between two graphs.
        """
        ref_adj = torch.FloatTensor(ref_adj)
        curr_adj = torch.FloatTensor(curr_adj)

        ref_feat = torch.FloatTensor(ref_feat)
        curr_feat = torch.FloatTensor(curr_feat)

        P = nn.Parameter(torch.FloatTensor(ref_adj.shape[0], curr_adj.shape[0]))
        with torch.no_grad():
            nn.init.constant_(P, 1.0 / ref_adj.shape[0])
            P[ref_node_idx, :] = 0.0
            P[:, curr_node_idx] = 0.0
            P[ref_node_idx, curr_node_idx] = 1.0
        opt = torch.optim.Adam([P], lr=0.01, betas=(0.5, 0.999))
        for i in range(args.align_steps):
            opt.zero_grad()
            feat_loss = torch.norm(P @ curr_feat - ref_feat)

            aligned_adj = P @ curr_adj @ torch.transpose(P, 0, 1)
            align_loss = torch.norm(aligned_adj - ref_adj)
            loss = feat_loss + align_loss
            loss.backward()  # Calculate gradients
            self.writer.add_scalar("optimization/align_loss", loss, i)
            outfile.write("iter: ", i, "; loss: ", loss)
            opt.step()

        return P, aligned_adj, P @ curr_feat

    def make_pred_real(self, adj, start):
        # house graph
        if self.args.dataset == "syn1" or self.args.dataset == "syn2":
            # num_pred = max(G.number_of_edges(), 6)
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()

            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start][start + 3] > 0:
                real[start][start + 3] = 10
            if real[start][start + 4] > 0:
                real[start][start + 4] = 10
            if real[start + 1][start + 4]:
                real[start + 1][start + 4] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        # cycle graph
        elif self.args.dataset == "syn4":
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()
            # pdb.set_trace()
            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start + 3][start + 4] > 0:
                real[start + 3][start + 4] = 10
            if real[start + 4][start + 5] > 0:
                real[start + 4][start + 5] = 10
            if real[start][start + 5]:
                real[start][start + 5] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        return pred, real


class ExplainModule(nn.Module):
    def __init__(
        self,
        adj,
        x,
        model,
        label,
        args,
        graph_idx=0,
        writer=None,
        use_sigmoid=True,
        graph_mode=False,
    ):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.graph_idx = graph_idx
        self.args = args
        self.writer = writer
        self.mask_act = args.mask_act
        self.use_sigmoid = use_sigmoid
        self.graph_mode = graph_mode

        init_strategy = "normal"
        num_nodes = adj.size()[1]
        self.mask, self.mask_bias = self.construct_edge_mask(
            num_nodes, init_strategy=init_strategy
        )

        self.feat_mask = self.construct_feat_mask(x.size(-1), init_strategy="constant")
        params = [self.mask, self.feat_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        self.scheduler, self.optimizer = train_utils.build_optimizer(args, params)

        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
        }

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                # mask[0] = 2
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def _masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj
        masked_adj = adj * sym_mask
        if self.args.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj * self.diag_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def forward(self, node_idx, unconstrained=False, mask_features=True, marginalize=False):
        x = self.x.cuda() if self.args.gpu else self.x

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = (
                torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0) * self.diag_mask
            )
        else:
            self.masked_adj = self._masked_adj()
            if mask_features:
                feat_mask = (
                    torch.sigmoid(self.feat_mask)
                    if self.use_sigmoid
                    else self.feat_mask
                )
                if marginalize:
                    std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                    mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                    z = torch.normal(mean=mean_tensor, std=std_tensor)
                    x = x + z * (1 - feat_mask)
                else:
                    x = x * feat_mask

        ypred, adj_att = self.model(x, self.masked_adj)
        if self.graph_mode:
            res = nn.Softmax(dim=0)(ypred[0])
        else:
            node_pred = ypred[self.graph_idx, node_idx, :]
            res = nn.Softmax(dim=0)(node_pred)
        #print("feature mask", self.feat_mask)
        return res, adj_att, self.feat_mask

    def adj_feat_grad(self, node_idx, pred_label_node):
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
            label = self.label.cuda()
        else:
            x, adj = self.x, self.adj
        ypred, _ = self.model(x, adj)
        if self.graph_mode:
            logit = nn.Softmax(dim=0)(ypred[0])
        else:
            logit = nn.Softmax(dim=0)(ypred[self.graph_idx, node_idx, :])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return self.adj.grad, self.x.grad

    def loss(self, pred, pred_label, node_idx, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = False
        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))
        else:
            pred_label_node = pred_label if self.graph_mode else pred_label[node_idx]
            gt_label_node = self.label if self.graph_mode else self.label[0][node_idx]
            logit = pred[gt_label_node]
            pred_loss = -torch.log(logit)
        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs["size"] * torch.sum(mask)

        # pre_mask_sum = torch.sum(self.feat_mask)
        feat_mask = (
            torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        feat_mask_ent = - feat_mask             \
                        * torch.log(feat_mask)  \
                        - (1 - feat_mask)       \
                        * torch.log(1 - feat_mask)

        feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(feat_mask_ent)

        # laplacian
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        m_adj = self.masked_adj if self.graph_mode else self.masked_adj[self.graph_idx]
        L = D - m_adj
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            L = L.cuda()
        if self.graph_mode:
            lap_loss = 0
        else:
            lap_loss = (self.coeffs["lap"]
                * (pred_label_t @ L @ pred_label_t)
                / self.adj.numel()
            )

        # grad
        # adj
        # adj_grad, x_grad = self.adj_feat_grad(node_idx, pred_label_node)
        # adj_grad = adj_grad[self.graph_idx]
        # x_grad = x_grad[self.graph_idx]
        # if self.args.gpu:
        #    adj_grad = adj_grad.cuda()
        # grad_loss = self.coeffs['grad'] * -torch.mean(torch.abs(adj_grad) * mask)

        # feat
        # x_grad_sum = torch.sum(x_grad, 1)
        # grad_feat_loss = self.coeffs['featgrad'] * -torch.mean(x_grad_sum * mask)

        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss
        if self.writer is not None:
            self.writer.add_scalar("optimization/size_loss", size_loss, epoch)
            self.writer.add_scalar("optimization/feat_size_loss", feat_size_loss, epoch)
            self.writer.add_scalar("optimization/mask_ent_loss", mask_ent_loss, epoch)
            self.writer.add_scalar(
                "optimization/feat_mask_ent_loss", mask_ent_loss, epoch
            )
            # self.writer.add_scalar('optimization/grad_loss', grad_loss, epoch)
            self.writer.add_scalar("optimization/pred_loss", pred_loss, epoch)
            self.writer.add_scalar("optimization/lap_loss", lap_loss, epoch)
            self.writer.add_scalar("optimization/overall_loss", loss, epoch)
        return loss

    def log_mask(self, epoch):
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(4, 3), dpi=400)
        plt.imshow(self.mask.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/mask", tensorboardX.utils.figure_to_image(fig), epoch
        )
        
        

        # fig = plt.figure(figsize=(4,3), dpi=400)
        # plt.imshow(self.feat_mask.cpu().detach().numpy()[:,np.newaxis], cmap=plt.get_cmap('BuPu'))
        # cbar = plt.colorbar()
        # cbar.solids.set_edgecolor("face")

        # plt.tight_layout()
        # fig.canvas.draw()
        # self.writer.add_image('mask/feat_mask', tensorboardX.utils.figure_to_image(fig), epoch)
        io_utils.log_matrix(
            self.writer, torch.sigmoid(self.feat_mask), "mask/feat_mask", epoch
        )
        #print("feature mask", self.feat_mask)
        fig = plt.figure(figsize=(4, 3), dpi=400)
        # use [0] to remove the batch dim
        plt.imshow(self.masked_adj[0].cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/adj", tensorboardX.utils.figure_to_image(fig), epoch
        )

        if self.args.mask_bias:
            fig = plt.figure(figsize=(4, 3), dpi=400)
            # use [0] to remove the batch dim
            plt.imshow(self.mask_bias.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
            cbar = plt.colorbar()
            cbar.solids.set_edgecolor("face")

            plt.tight_layout()
            fig.canvas.draw()
            self.writer.add_image(
                "mask/bias", tensorboardX.utils.figure_to_image(fig), epoch
            )

    def log_adj_grad(self, node_idx, pred_label, epoch, label=None):
        log_adj = False

        if self.graph_mode:
            predicted_label = pred_label
            # adj_grad, x_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[0]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[0]
            x_grad = torch.sum(x_grad[0], 0, keepdim=True).t()
        else:
            predicted_label = pred_label[node_idx]
            # adj_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[self.graph_idx]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[self.graph_idx]
            x_grad = x_grad[self.graph_idx][node_idx][:, np.newaxis]
            # x_grad = torch.sum(x_grad[self.graph_idx], 0, keepdim=True).t()
        adj_grad = (adj_grad + adj_grad.t()) / 2
        adj_grad = (adj_grad * self.adj).squeeze()
        if log_adj:
            io_utils.log_matrix(self.writer, adj_grad, "grad/adj_masked", epoch)
            self.adj.requires_grad = False
            io_utils.log_matrix(self.writer, self.adj.squeeze(), "grad/adj_orig", epoch)

        masked_adj = self.masked_adj[0].cpu().detach().numpy()

        # only for graph mode since many node neighborhoods for syn tasks are relatively large for
        # visualization
        if self.graph_mode:
            G, threshold = io_utils.denoise_graph(
                masked_adj, node_idx, feat=self.x[0], threshold=None, max_component=False
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph_orig",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        io_utils.log_matrix(self.writer, x_grad, "grad/feat", epoch)

        adj_grad = adj_grad.detach().numpy()
        if self.graph_mode:
            #outfile.write("GRAPH model")
            G, threshold = io_utils.denoise_graph(
                adj_grad,
                node_idx,
                feat=self.x[0],
                threshold=0.0003,  # threshold_num=20,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        else:
            # G = io_utils.denoise_graph(adj_grad, node_idx, label=label, threshold=0.5)
            G, threshold = io_utils.denoise_graph(adj_grad, node_idx, threshold_num=12)
            io_utils.log_graph(
                self.writer, G, name="grad/graph", epoch=epoch, args=self.args
            )

        # if graph attention, also visualize att

    def log_masked_adj(self, node_idx, epoch, name="mask/graph", label=None):
        # use [0] to remove the batch dim
        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        if self.graph_mode:
            G, threshold = io_utils.denoise_graph(
                masked_adj,
                node_idx,
                feat=self.x[0],
                threshold=0.2,  # threshold_num=20,
                max_component=True,
            )
            '''
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=False,
                nodecolor="feat",
                epoch=epoch,
                label_node_feat=True,
                edge_vmax=None,
                args=self.args,
            )
            '''
        else:
            G, threshold = io_utils.denoise_graph(
                masked_adj, node_idx, threshold_num=12, max_component=True
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=True,
                nodecolor="label",
                epoch=epoch,
                edge_vmax=None,
                args=self.args,
)