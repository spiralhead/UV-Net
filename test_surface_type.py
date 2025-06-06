import os
import torch
from glob import glob
from tqdm import tqdm
from dgl.data.utils import load_graphs
from occwl.io import load_step
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
all_unique_z_values = set()
unique_len = len(all_unique_z_values)
surface_type_map = {5: "other"}
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
        faces = []
        for solid in solids:
            faces.extend(solid.faces())
    except IndexError:
        continue
    faces_types = [i.surface_type() for i in faces]
    z_tensor = graphs[0].ndata["z"]
    unique_z_values, idxs, counts = torch.unique(
        z_tensor, sorted=True, return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idxs, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    if len(faces_types) != len(z_tensor):
        print(
            f"For path {bin_path} is some problem with faces_types {faces_types} with len={len(faces_types)} and {z_tensor} with len={len(z_tensor)}"
        )
        continue
    for z, idx in zip(unique_z_values.tolist(), first_indicies.tolist()):
        if z not in surface_type_map.keys():
            surface_type_map[z] = faces_types[idx]
        else:
            if z != 5 and surface_type_map[z] != faces_types[idx]:
                # Если убрать z=5 то можно увидеть, что пятерку используют часто для описания других поверхностей,
                # включая обычные поверхности cone,plane и другие поверхности bspline,extrusion. Явно использовался какой-то рандомизатор, который иногда всем присваивал 5
                # Возможно torus и revolution обрабатывается типом 4, это объясняет, что этот тип используется и в тех и других случаев, но другой тип не используется
                print(surface_type_map)
                print(
                    f"Problem with {z} key and value {surface_type_map[z]}. Should be {faces_types[idx]}. Detail number {bin_path}"
                )
                if faces_types[idx] in surface_type_map.values():
                    print(
                        f"Value {faces_types[idx]} is already in surface map, doing nothing"
                    )
                else:
                    print(
                        f"Changing surface_type_map key {z} with value {faces_types[idx]}"
                    )
                    surface_type_map[z] = faces_types[idx]
print("Отображение", surface_type_map)
