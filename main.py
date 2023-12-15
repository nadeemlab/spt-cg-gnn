#!/usr/bin/env python3
"""Convert SPT graph objects to CG-GNN graph objects and run training and evaluation with them."""

from spatialprofilingtoolbox.cggnn.util import load_hs_graphs, save_hs_graphs

from cggnn.util import save_cell_graphs, load_cell_graphs
from cggnn.run import train_and_evaluate
from cggnn.scripts.train import parse_arguments
from spt_helper import convert_spt_graphs_data, convert_dgl_graphs_data

if __name__ == '__main__':
    args = parse_arguments()
    working_directory = '.'

    save_cell_graphs(convert_spt_graphs_data(load_hs_graphs(args.cg_directory)[0]),
                     working_directory)

    model, graphs_data, hs_id_to_importances = train_and_evaluate(working_directory,
                                                                  args.in_ram,
                                                                  args.batch_size,
                                                                  args.epochs,
                                                                  args.learning_rate,
                                                                  args.k_folds,
                                                                  args.explainer,
                                                                  args.merge_rois,
                                                                  args.random_seed)

    save_hs_graphs(convert_dgl_graphs_data(load_cell_graphs(working_directory)[0]), '.')
