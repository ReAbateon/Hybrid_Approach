#
# Copyright (c) 2026 Lorenzo Abate <lorenzo.abate@unina.it>.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import csv
import os
import shutil
from collections import deque
from pathlib import Path

import joblib
import numpy as np

import Code_Gen.Utils.trainRF as trainRF


def testset_gen(csv_path, fh, n_rows):
    fh.write("RAM_BIG int16_t testset[TEST_ROWS][SAMPLE_SIZE] = {\n")

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if i >= n_rows:
                break
            int_row = [int(float(x)) for x in row]
            rows.append(int_row)

    for r, row in enumerate(rows):
        values = ", ".join(str(v) for v in row)
        comma = "," if r < n_rows - 1 else ""
        fh.write(f"   {{{values}}}{comma}\n")

    fh.write("};\n\n")


def get_triangle_node_indices(tree, root_idx):
    # Returns the 7 node indices of a triangle starting at root_idx
    # following a level-order traversal (0: root, 1:L, 2:R, 3:LL, 4:LR, 5:RL, 6:RR)
    nodes = [-1] * 7
    nodes[0] = root_idx

    # Level 1
    if nodes[0] != -1:
        nodes[1] = tree.children_left[nodes[0]]
        nodes[2] = tree.children_right[nodes[0]]

    # Level 2
    if nodes[1] != -1:
        nodes[3] = tree.children_left[nodes[1]]
        nodes[4] = tree.children_right[nodes[1]]
    if nodes[2] != -1:
        nodes[5] = tree.children_left[nodes[2]]
        nodes[6] = tree.children_right[nodes[2]]

    return nodes


def simulate_path(tree, tri_nodes, bitmask, root_to_id, current_tri_id, task):
    # tri_nodes are [0, 1, 2, 3, 4, 5, 6] in level-order
    curr_idx_in_tri = 0

    while True:
        node_id = tri_nodes[curr_idx_in_tri]

        if node_id == -1:
            return 0  # Should not happen

        # Is it a tree leaf?
        if tree.children_left[node_id] == -1:
            if task == 0:
                val = int(np.argmax(tree.value[node_id][0]))
            else:
                val = int(tree.value[node_id][0][0])
            return -val

        # Is it a root of another triangle? (Not for the first node of this tri)
        if node_id in root_to_id and node_id != tri_nodes[0]:
            return int(root_to_id[node_id] - current_tri_id)

        # Comparison result for this node (from bitmask)
        go_left = (bitmask >> curr_idx_in_tri) & 1

        # Decide next step
        if curr_idx_in_tri == 0:
            curr_idx_in_tri = 1 if go_left else 2
        elif curr_idx_in_tri == 1:
            curr_idx_in_tri = 3 if go_left else 4
        elif curr_idx_in_tri == 2:
            curr_idx_in_tri = 5 if go_left else 6
        else:
            # We are at Level 2 (3, 4, 5, 6). The next step exits the triangle.
            target_node = (
                tree.children_left[node_id] if go_left else tree.children_right[node_id]
            )

            if target_node == -1:
                return 0

            if tree.children_left[target_node] == -1:  # Target is a leaf
                if task == 0:
                    val = int(np.argmax(tree.value[target_node][0]))
                else:
                    val = int(tree.value[target_node][0][0])
                return -val
            else:
                # Target must be a root of another triangle
                if target_node in root_to_id:
                    return int(root_to_id[target_node] - current_tri_id)
                else:
                    return 0


def fast_initialization(model, task):
    DELTA = 32767
    all_trees_triangles = []

    for estimator in model.estimators_:
        tree = estimator.tree_
        # 1. BFS to find all triangle roots (only non-leaf nodes)
        triangle_roots = []
        if tree.children_left[0] != -1:  # Root is not a leaf
            queue = deque([0])
            seen = {0}
            while queue:
                root = queue.popleft()
                triangle_roots.append(root)

                tri_nodes = get_triangle_node_indices(tree, root)
                # Next roots are children of level 2 nodes (3, 4, 5, 6)
                for i in [3, 4, 5, 6]:
                    node_idx = tri_nodes[i]
                    if node_idx != -1:
                        for child in [
                            tree.children_left[node_idx],
                            tree.children_right[node_idx],
                        ]:
                            if (
                                child != -1 and tree.children_left[child] != -1
                            ):  # Not a leaf
                                if child not in seen:
                                    queue.append(child)
                                    seen.add(child)
        else:
            triangle_roots = [0]

        # 2. Map root node index to triangle ID
        root_to_id = {root: i for i, root in enumerate(triangle_roots)}

        # 3. For each triangle, generate struct data
        tree_triangles = []
        for i, root in enumerate(triangle_roots):
            tri_nodes = get_triangle_node_indices(tree, root)

            feats = []
            threshs = []
            for node_idx in tri_nodes:
                if node_idx != -1 and tree.children_left[node_idx] != -1:
                    feats.append(int(tree.feature[node_idx]))
                    threshs.append(int(tree.threshold[node_idx]))
                else:
                    feats.append(0)
                    threshs.append(DELTA)
            feats.append(0)
            threshs.append(0)

            m_matrix = [0] * 128
            for bitmask in range(128):
                m_matrix[bitmask] = simulate_path(
                    tree, tri_nodes, bitmask, root_to_id, i, task
                )

            tree_triangles.append(
                {"features": feats, "thresholds": threshs, "M": m_matrix}
            )
        all_trees_triangles.append(tree_triangles)
    return all_trees_triangles


def FAST_generator(forest_triangles, fh, hybrid, int16_flags):
    # Struct definition is in generate_header

    for tree_idx, tree_triangles in enumerate(forest_triangles):
        section = "DTCM" if (hybrid == -1 or tree_idx < hybrid) else "RAM_BIG"
        tree_name = (
            f"tree_dtcm{tree_idx}"
            if (hybrid == -1 or tree_idx < hybrid)
            else f"tree{tree_idx}"
        )

        fh.write(
            f"{section} Triangle {tree_name}_triangles[{len(tree_triangles)}] = {{\n"
        )
        for tri_idx, tri in enumerate(tree_triangles):
            fh.write(f"    {{ {{")
            fh.write(", ".join(map(str, tri["thresholds"])))
            fh.write("}, {")
            fh.write(", ".join(map(str, tri["features"])))
            fh.write("}, {")
            fh.write(", ".join(map(str, tri["M"])))
            fh.write("} }")
            if tri_idx < len(tree_triangles) - 1:
                fh.write(",")
            fh.write("\n")
        fh.write("};\n\n")


def generate_header(trees, sample_size, test_rows, int16_flags):
    m_type = "int16_t" if any(int16_flags) else "int8_t"
    return f"""\
#ifndef INC_FAST_H_
#define INC_FAST_H_

#include "arm_mve.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdbool.h"

#define TREES          {trees}
#define DELTA          32767
#define SAMPLE_SIZE    {sample_size}
#define TEST_ROWS      {test_rows}

#define DTCM __attribute__((section(".dtcm_data"), aligned(32)))
#define RAM_BIG __attribute__((section(".big_data"), aligned(32)))

int16_t classes[TREES] = {{0}};

typedef struct Triangle{{
	int16_t thresholds[8];
	uint16_t features[8];
	{m_type} M[128];
}}Triangle;

"""


def generate_kernel():
    return f"""\
int16_t fast_v2(Triangle *arr, int16_t *sample) {{
    static const int16x8_t twos = {{1, 2, 4, 8, 16, 32, 64, 0}};

    int16_t next_index = 1;
    Triangle *tri = arr;

    while (next_index > 0) {{
        int16x8_t thresh = vld1q_s16(tri->thresholds);
        uint16x8_t samp_offset = vld1q_u16(tri->features);

        int16x8_t samp = vldrhq_gather_shifted_offset_s16(sample, samp_offset);

        mve_pred16_t pred = vcmpleq_s16(samp, thresh);

        int16x8_t dst = vpselq_s16(vdupq_n_s16(1), vdupq_n_s16(0), pred);
      
		int32_t index = vmladavaq_s16(0, dst, twos);

        next_index = tri->M[index];
        tri += next_index;
    }}

    return -next_index;
}}

"""


def internal_nodes_calc(path, task):
    model = joblib.load(path)
    forest_triangles = fast_initialization(model, task)

    total_triangles = 0
    size_bytes = 0
    hybrid = -1
    int16_flags = []

    for i, tree_triangles in enumerate(forest_triangles):
        num_triangles = len(tree_triangles)
        total_triangles += num_triangles

        max_offset = 0
        min_offset = 0
        for tri in tree_triangles:
            max_offset = max(max_offset, max(tri["M"]))
            min_offset = min(min_offset, min(tri["M"]))

        is_int16 = max_offset > 127 or min_offset < -128
        int16_flags.append(is_int16)

        # Triangle size: 16 (thresh) + 16 (feat) + (128 or 256 for M)
        tri_size = 32 + (256 if is_int16 else 128)
        size_bytes += num_triangles * tri_size

        if size_bytes > 130000 and hybrid == -1:
            hybrid = i

    size_kb = size_bytes / 1024

    print(f"Model Stats:")
    print(f"  - Total Triangles: {total_triangles}")
    print(f"  - Dimension: {size_bytes} Bytes ({size_kb:.2f} KB)")

    return forest_triangles, hybrid, int16_flags


def generate_inference(fh, trees, hybrid):
    fh.write("static inline void inference(int16_t* sample){\n")
    for i in range(trees):
        tree_name = f"tree_dtcm{i}" if (hybrid == -1 or i < hybrid) else f"tree{i}"
        fh.write(f"   classes[{i}] = fast_v2({tree_name}_triangles, sample);\n")
    fh.write("}\n\n")


def generate_fast(
    path, number_of_trees, max_depth, random_seed, number_of_test_samples
):
    test_path, joblib_path, sample_size, task, accuracy, mse, mae, r2, csv_stem = (
        trainRF.training(
            path, number_of_trees, max_depth, random_seed, number_of_test_samples
        )
    )

    forest_triangles, hybrid, int16_flags = internal_nodes_calc(joblib_path, task)

    models_path = Path(f"Models/{csv_stem}/fast_RF_T{number_of_trees}_D{max_depth}")
    os.makedirs(models_path, exist_ok=True)

    suffix = "_hybrid" if hybrid != -1 else ""
    header_name = (
        f"{csv_stem}_RF_fast{suffix}_T{number_of_trees}_D{max_depth}_RS{random_seed}.h"
    )
    header_path = models_path / header_name

    header = generate_header(
        number_of_trees, sample_size, number_of_test_samples, int16_flags
    )
    kernel = generate_kernel()

    with open(header_path, "w") as f:
        f.write(header)
        FAST_generator(forest_triangles, f, hybrid, int16_flags)
        f.write(kernel)
        generate_inference(f, number_of_trees, hybrid)
        testset_gen(test_path, f, number_of_test_samples)
        f.write("#endif\n")

    print(f"Code successfully generated in {header_path}")
