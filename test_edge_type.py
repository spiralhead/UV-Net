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

# bin_files = glob(os.path.join(bin_dir, "*"))
# bin_files += glob(os.path.join(bin_dir, "/*.bin"), recursive=True)
# bin_files += glob(os.path.join(bin_dir, "*.BIN"))
# bin_files += glob(os.path.join(bin_dir, "/*.BIN"), recursive=True)

print(f"Найдено .bin файлов: {len(bin_files)}")
# step_files = list()
all_unique_t_values = set()
unique_len = len(all_unique_t_values)
edge_type_map = {}
for bin_path, step_path in tqdm(list(zip(bin_files, step_files))):
    if base_name_func(bin_path) != base_name_func(step_path):
        assert False, f"Names mismatch for {bin_path} and {step_path}"
    graphs, _ = load_graphs(bin_path)
    if len(graphs) != 1:
        assert False, "Graphs len is more than one"
    try:
        solids = load_step(step_path)
        if len(solids) > 1:
            print(bin_path)
        #     from occwl.viewer import Viewer
        #     v = Viewer(backend="wx")
        #     v.display(solids)
        #     v.fit()
        #     v.show()
        edges = []
        for solid in solids:
            solid_graph = face_adjacency(solid)
            edges.extend([solid_graph.edges[idx]["edge"] for idx in solid_graph.edges])
    except IndexError:
        continue
    edges_types = [i.curve_type() for i in edges]
    t_tensor = graphs[0].edata["t"]
    unique_t_values, idxs, counts = torch.unique(
        t_tensor, sorted=True, return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idxs, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    if len(edges_types) != len(t_tensor):
        print(
            f"For path {bin_path} is some problem with edges_types {edges_types} with len={len(edges_types)} and {t_tensor} with len={len(t_tensor)}"
        )
        continue
    for t, idx in zip(unique_t_values.tolist(), first_indicies.tolist()):
        if t not in edge_type_map.keys():
            edge_type_map[t] = edges_types[idx]
        else:
            if t != 5 and edge_type_map[t] != edges_types[idx]:
                # Здесь значиения прыгают очень сильно, я пока решил забить на это обстоятельство, и сделать типизацию самостоятельно
                print(edge_type_map)
                print(
                    f"Problem with {t} key and value {edge_type_map[t]}. Should be {edges_types[idx]}. Detail number {bin_path}"
                )
                if edges_types[idx] in edge_type_map.values():
                    print(
                        f"Value {edges_types[idx]} is already in surface map, doing nothing"
                    )
                else:
                    print(
                        f"Changing edge_type_map key {t} with value {edges_types[idx]}"
                    )
                    edge_type_map[t] = edges_types[idx]
print("Отображение", edge_type_map)
