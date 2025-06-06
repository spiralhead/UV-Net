import os
import torch
from glob import glob
from tqdm import tqdm
from dgl.data.utils import load_graphs
from occwl.io import load_step
from occwl.graph import face_adjacency
import itertools


def base_name_func(x):
    return os.path.splitext(os.path.basename(x))[0]


bin_dir = "/mnt/d/Work/Datasets/synthcad/bin/"
step_dir = "/mnt/d/Work/Datasets/synthcad/step"
bin_files = glob(os.path.join(bin_dir, "*.bin"))
step_files = glob(os.path.join(step_dir, "*.stp"))

solid = load_step(step_files[1])[0]
graph = load_graphs(bin_files[1])[0][0]
graph
areas = []
loops = []
counts = []
solid_graph = face_adjacency(solid)
for node_idx, face in solid_graph.nodes(data=True):
    areas.append(face["face"].area())
    loops.append(len(list(face["face"].wires())))
    counts.append(solid_graph.in_degree[node_idx])
print(counts)
