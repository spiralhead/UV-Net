import os
import torch
from glob import glob
from tqdm import tqdm
from dgl.data.utils import load_graphs
from occwl.io import load_step
from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid
import numpy as np
import itertools
import random
from occwl.face import Face
import networkx as nx
from itertools import combinations, permutations

# import matplotlib.pyplot as plt
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add


def plot_two_histograms(
    hist1: np.ndarray, hist2: np.ndarray, filename: str = "histograms.png"
):
    bins = np.arange(len(hist1))

    plt.figure(figsize=(10, 6))

    # Смещение второй гистограммы для визуального разделения
    width = 0.4
    plt.bar(bins - width / 2, hist1, width=width, label="Histogram 1", alpha=0.7)
    plt.bar(bins + width / 2, hist2, width=width, label="Histogram 2", alpha=0.7)

    plt.xlabel("Bin Index")
    plt.ylabel("Value")
    plt.title("Comparison of Two Histograms")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Сохранение в файл
    plt.savefig(filename)
    plt.close()


def base_name_func(x):
    return os.path.splitext(os.path.basename(x))[0]


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def vector_dot_avg(v1, v2):
    return np.sum(v1 * v2, axis=1).mean()


def mean_angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(vector_dot_avg(v1_u, v2_u), -1.0, 1.0))


def dihedral_angle(edge, face1, face2):
    edge_points = ugrid(edge, 10)
    face1_normals = np.apply_along_axis(
        lambda point: face1.normal(face1.point_to_parameter(point)),
        axis=1,
        arr=edge_points,
    )
    face2_normals = np.apply_along_axis(
        lambda point: face2.normal(face2.point_to_parameter(point)),
        axis=1,
        arr=edge_points,
    )
    return mean_angle_between(face1_normals, face2_normals)


def sample_random_points_on_face(faces: [Face], num_points: int):
    # Получаем триангуляцию
    all_verts_chunks = []  # list of (n_i, 3) float arrays
    all_tris_chunks = []  # list of (m_i, 3) int  arrays
    vertex_offset = 0
    for face in faces:
        face.triangulate_all_faces()
        verts, tris = face.get_triangles()
        verts = np.array(verts)
        all_verts_chunks.append(verts)
        all_tris_chunks.append(tris + vertex_offset)  # shift indices
        vertex_offset += len(verts)
    verts = np.concatenate(all_verts_chunks, axis=0)  # (Σn_i, 3)
    tris = np.concatenate(all_tris_chunks, axis=0)

    # Вычисляем площади всех треугольников
    def triangle_area(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

    def triangle_areas(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
        a = verts[tris[:, 0]]
        b = verts[tris[:, 1]]
        c = verts[tris[:, 2]]

        ab = b - a
        ac = c - a

        cross = np.cross(ab, ac)
        area = 0.5 * np.sqrt(np.einsum("ij,ij->i", cross, cross))
        return area

    areas = triangle_areas(verts, tris)
    # areas = np.array(
    #     [triangle_area(verts[i0], verts[i1], verts[i2]) for i0, i1, i2 in tris]
    # )
    total_area = np.sum(areas)

    # Вычисляем распределение вероятностей по треугольникам
    probs = areas / total_area

    # Выборка случайных точек
    sampled_points = []

    def random_point_in_triangle(A, B, C):
        r1 = np.random.rand()
        r2 = np.random.rand()

        # Отражаем, если точка вышла за пределы треугольника
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2

        # Вычисляем координаты случайной точки
        P = A + r1 * (B - A) + r2 * (C - A)
        return P

    for _ in range(num_points):
        # Выбираем треугольник с учётом его площади
        tri_idx = np.random.choice(len(tris), p=probs)
        i0, i1, i2 = tris[tri_idx]
        a, b, c = verts[i0], verts[i1], verts[i2]

        # # Генерация случайной точки внутри треугольника с равномерным распределением
        # r1 = np.sqrt(random.random())
        # r2 = random.random()
        # point = (1 - r1) * a + r1 * (1 - r2) * b + r1 * r2 * c
        sampled_points.append(random_point_in_triangle(a, b, c))

    return np.array(sampled_points)


def sample_random_points_on_2_faces(face1: Face, face2: Face, num_points: int):
    points_1 = sample_random_points_on_face([face1], num_points)
    points_2 = sample_random_points_on_face([face2], num_points)
    points = np.stack((points_1, points_2), axis=2)
    return points


def sample_random_points_on_2_faces_for_A3(face1: Face, face2: Face, num_points: int):
    # num_points_1 = 3 * num_points // 2
    # num_points_1 = num_points // 2
    # num_points_2 = num_points - num_points_1
    # num_points_2 = 3 * num_points - num_points_1
    points_1 = sample_random_points_on_face([face1], num_points)
    points_21 = sample_random_points_on_face([face2], num_points)
    # points = np.concatenate((points_1, points_2))
    # np.random.shuffle(
    #     points,
    # )
    points_22 = sample_random_points_on_face(
        [face2], num_points
    )  # sample_random_points_on_face(face2, num_points)
    # points_part_1 = np.stack((points_1, points_21, points_22), axis=2)
    # points_1 = sample_random_points_on_face(face2, num_points_2)
    # points_21 = sample_random_points_on_face(face1, num_points_2)
    # points_22 = sample_random_points_on_face(face1, num_points_2)
    # points = np.stack((points_1, points_21, points_22), axis=2)
    # return np.concatenate((points_part_1, points_part_2), axis=0)
    return points_1, points_21, points_22
    # return points.reshape(num_points, 3, 3)


def d2_metric_evaluation(face1, face2, diagonal_length):
    points = sample_random_points_on_2_faces(face1, face2, 512)
    diff = points[:, :, 0] - points[:, :, 1]  # Shape: (n, n, d)

    # Compute squared Euclidean distances
    dist_squared = np.sum(diff**2, axis=-1)  # Shape: (n, n)

    # Compute Euclidean distances
    distances = np.sqrt(dist_squared) / diagonal_length
    hist_counts, bin_edges = np.histogram(
        distances,
        bins=64,
        range=(0, 1),
    )
    total_count = hist_counts.sum()
    normalized_frequencies = hist_counts / total_count
    return normalized_frequencies


def a3_metric_evaluation(face1, face2):
    P1, P2, P3 = sample_random_points_on_2_faces_for_A3(face1, face2, 512)
    # P1 = triplets[:, :, 0]
    # P2 = triplets[:, :, 1]
    # P3 = triplets[:, :, 2]
    vec_a = P2 - P1
    vec_b = P3 - P1
    dot_products = np.einsum("ij,ij->i", vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a, axis=1)
    norm_b = np.linalg.norm(vec_b, axis=1)

    # Compute cosine of angles
    cos_theta = dot_products / (norm_a * norm_b)

    # Handle numerical errors that might lead to values slightly outside [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles_rad = np.arccos(cos_theta)
    hist_counts, bin_edges = np.histogram(
        angles_rad,
        bins=64,
        range=(0, np.pi),
    )
    total_count = hist_counts.sum()
    normalized_frequencies = hist_counts / total_count
    return normalized_frequencies


def main():
    bin_dir = "/mnt/d/Work/Datasets/synthcad/bin/"
    step_dir = "/mnt/d/Work/Datasets/synthcad/step"
    bin_files = glob(os.path.join(bin_dir, "*.bin"))
    step_files = glob(os.path.join(step_dir, "*.stp"))

    solid = load_step(step_files[0])[0]
    (graph,), add_data = load_graphs(bin_files[0])
    graph
    areas = []
    loops = []
    counts = []
    solid_graph = face_adjacency(solid)
    for node_idx, face in solid_graph.nodes(data=True):
        areas.append(face["face"].area())
        loops.append(len(list(face["face"].wires())))
        counts.append(solid_graph.in_degree[node_idx])
    # Edge features
    lengths = []
    angle = []
    for node1_idx, node2_idx, edge in solid_graph.edges(data=True):
        lengths.append(edge["edge"].length())
        angle.append(
            dihedral_angle(
                edge["edge"],
                solid_graph.nodes[node1_idx]["face"],
                solid_graph.nodes[node2_idx]["face"],
            )
        )

    print(lengths)
    print(graph.edata["l"])
    # Getting diagonal length bounding box
    # Get the bounding box of the solid
    shape = solid._shape
    # getting additional data

    # region a3 and d2
    # getting a3 and d2 data
    # Compute the bounding box
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)

    # Retrieve the bounding box limits
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    # Compute the diagonal length
    min_corner = np.array([xmin, ymin, zmin])
    max_corner = np.array([xmax, ymax, zmax])
    diagonal_length = np.linalg.norm(max_corner - min_corner)

    print("Bounding box diagonal length:", diagonal_length)

    # For highly distorted faces I have to use area-weighted sampling
    # So if i would get problem with those surfaces i'll know that problem
    # is with sampling method
    # def sample_random_points_on_face(face: Face, num_points: int):
    #     uv_box = face.uv_bounds()
    #     umin, vmin = uv_box.min_point()
    #     umax, vmax = uv_box.max_point()
    #     points = []
    #     while len(points) < num_points:
    #         u = random.uniform(umin, umax)
    #         v = random.uniform(vmin, vmax)
    #         pt = face.point((u, v))
    #         if face.inside((u, v)):
    #             pt = face.point((u, v))
    #             points.append(pt)
    #     points = np.array(points)
    #     return points

    # engregion

    # region
    # Compute the distance between faces
    graph_diameter = nx.diameter(solid_graph)
    number_of_faces = len(solid_graph.nodes)
    edge_index_map = {edge: idx for idx, edge in enumerate(solid_graph.edges())}
    d2_distances = np.zeros((number_of_faces, number_of_faces, 64))
    a3_distances = np.zeros((number_of_faces, number_of_faces, 64))
    spatial_pos = np.zeros((number_of_faces, number_of_faces))
    edges_path = -np.ones((number_of_faces, number_of_faces, graph_diameter))
    for face1_idx, face2_idx in permutations(solid_graph.nodes, 2):
        face1 = solid_graph.nodes[face1_idx]["face"]
        face2 = solid_graph.nodes[face2_idx]["face"]
        path_between_faces = nx.shortest_path(solid_graph, face1_idx, face2_idx)
        path_length = len(path_between_faces) - 1
        spatial_pos[face1_idx, face2_idx] = path_length
        spatial_pos[face2_idx, face1_idx] = path_length
        edges_path_list = [
            edge_index_map[(i, j)]
            for i, j in zip(path_between_faces, path_between_faces[1:])
        ]
        edges_path_array = np.asarray(edges_path_list)
        edges_path[face1_idx, face2_idx, : len(edges_path_list)] = edges_path_array
        d2_distances[face1_idx, face2_idx, :] = d2_metric_evaluation(
            face1, face2, diagonal_length
        )
        a3_distances[face1_idx, face2_idx, :] = a3_metric_evaluation(face1, face2)
        # Гистограммы сходятся не очень хорошо, возможны два варианта я неправильно раскидываю точки
        # либо я неправильно считаю угол, причем мне кажется что скорее точки не очень раскидываю, возможно надо взвешенно по площади
        # я не попадаю на поверхность
        # plot_two_histograms(
        #     a3_distances[face1_idx, face2_idx, :],
        #     add_data["angle_distance"][face1_idx, face2_idx, :],
        #     filename="hystogram{face1_idx}_{face2_idx}.png".format(
        #         face1_idx=face1_idx, face2_idx=face2_idx
        #     ),
        # )
    print(edges_path)
    return a3_distances
    # plot_two_histograms(a3_distances[4, 0, :], add_data["angle_distance"][4, 0, :])
    # for i in range(100):a
    #     (graph,), add_data = load_graphs(bin_files[i])
    #     print(add_data["edges_path"].shape)


if __name__ == "__main__":
    print(main())
