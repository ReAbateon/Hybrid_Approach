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
import re
import shutil
from collections import deque
from pathlib import Path

import joblib
import numpy as np

import Code_Gen.Utils.trainXGB as trainXGB


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


def parse_xgb_tree(tree_text):
    re_split = re.compile(
        r"^(?P<nodeid>\d+):\[(?P<feature>[^<]+)<(?P<thresh>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\]\s+yes=(?P<yes>\d+),no=(?P<no>\d+),missing=(?P<missing>\d+)"
    )
    re_leaf = re.compile(
        r"^(?P<nodeid>\d+):leaf=(?P<leaf>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
    )

    nodes = {}
    for line in tree_text.splitlines():
        line = line.strip()
        if not line:
            continue

        m_split = re_split.match(line)
        if m_split:
            d = m_split.groupdict()
            nodes[int(d["nodeid"])] = {
                "id": int(d["nodeid"]),
                "feature": int(d["feature"]),
                "threshold": float(d["thresh"]),
                "left": int(d["yes"]),
                "right": int(d["no"]),
                "is_leaf": False,
            }
        else:
            m_leaf = re_leaf.match(line)
            if m_leaf:
                d = m_leaf.groupdict()
                nodes[int(d["nodeid"])] = {
                    "id": int(d["nodeid"]),
                    "value": float(d["leaf"]),
                    "is_leaf": True,
                }

    max_id = max(nodes.keys())
    children_left = [-1] * (max_id + 1)
    children_right = [-1] * (max_id + 1)
    feature = [0] * (max_id + 1)
    threshold = [0.0] * (max_id + 1)
    value = [0.0] * (max_id + 1)

    for i in range(max_id + 1):
        if i in nodes:
            if nodes[i]["is_leaf"]:
                value[i] = nodes[i]["value"]
                children_left[i] = -1
                children_right[i] = -1
            else:
                children_left[i] = nodes[i]["left"]
                children_right[i] = nodes[i]["right"]
                feature[i] = nodes[i]["feature"]
                threshold[i] = nodes[i]["threshold"]

    class Tree:
        def __init__(self, cl, cr, f, t, v):
            self.children_left = cl
            self.children_right = cr
            self.feature = f
            self.threshold = t
            self.value = v

    return Tree(children_left, children_right, feature, threshold, value)


def get_triangle_node_indices(tree, root_idx):
    nodes = [-1] * 7
    nodes[0] = root_idx

    if nodes[0] != -1:
        nodes[1] = tree.children_left[nodes[0]]
        nodes[2] = tree.children_right[nodes[0]]

    if nodes[1] != -1:
        nodes[3] = tree.children_left[nodes[1]]
        nodes[4] = tree.children_right[nodes[1]]
    if nodes[2] != -1:
        nodes[5] = tree.children_left[nodes[2]]
        nodes[6] = tree.children_right[nodes[2]]

    return nodes


def simulate_path(
    tree, tri_nodes, bitmask, root_to_id, current_tri_id, min_v, range_v, task
):
    curr_idx_in_tri = 0

    while True:
        node_id = tri_nodes[curr_idx_in_tri]
        if node_id == -1:
            return 0

        if tree.children_left[node_id] == -1:
            val = tree.value[node_id]
            if task == 0:
                q = np.round((val - min_v) / range_v * 65535).astype(np.uint16)
            else:
                q = 1  # Simplified for regression placeholder
            return -int(np.int16(q))

        if node_id in root_to_id and node_id != tri_nodes[0]:
            return int(root_to_id[node_id] - current_tri_id)

        go_left = (bitmask >> curr_idx_in_tri) & 1

        if curr_idx_in_tri == 0:
            curr_idx_in_tri = 1 if go_left else 2
        elif curr_idx_in_tri == 1:
            curr_idx_in_tri = 3 if go_left else 4
        elif curr_idx_in_tri == 2:
            curr_idx_in_tri = 5 if go_left else 6
        else:
            target_node = (
                tree.children_left[node_id] if go_left else tree.children_right[node_id]
            )

            if target_node == -1:
                return 0

            if tree.children_left[target_node] == -1:
                val = tree.value[target_node]
                if task == 0:
                    q = np.round((val - min_v) / range_v * 65535).astype(np.uint16)
                else:
                    q = 1
                return -int(np.int16(q))
            else:
                if target_node in root_to_id:
                    return int(root_to_id[target_node] - current_tri_id)
                else:
                    return 0


def fast_initialization(dump, min_vector, range_vector, task, n_classes):
    DELTA = 32767
    all_trees_triangles = []
    trees_per_class = len(dump) // n_classes if n_classes > 2 else len(dump)

    for tree_idx, tree_text in enumerate(dump):
        tree = parse_xgb_tree(tree_text)
        class_idx = tree_idx // trees_per_class if n_classes > 2 else 0
        min_v = min_vector[class_idx] if isinstance(min_vector, list) else min_vector
        range_v = (
            range_vector[class_idx] if isinstance(range_vector, list) else range_vector
        )

        triangle_roots = []
        if tree.children_left[0] != -1:
            queue = deque([0])
            seen = {0}
            while queue:
                root = queue.popleft()
                triangle_roots.append(root)
                tri_nodes = get_triangle_node_indices(tree, root)
                for i in [3, 4, 5, 6]:
                    node_idx = tri_nodes[i]
                    if node_idx != -1:
                        for child in [
                            tree.children_left[node_idx],
                            tree.children_right[node_idx],
                        ]:
                            if child != -1 and tree.children_left[child] != -1:
                                if child not in seen:
                                    queue.append(child)
                                    seen.add(child)
        else:
            triangle_roots = [0]

        root_to_id = {root: i for i, root in enumerate(triangle_roots)}
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
                    tree, tri_nodes, bitmask, root_to_id, i, min_v, range_v, task
                )

            tree_triangles.append(
                {"features": feats, "thresholds": threshs, "M": m_matrix}
            )
        all_trees_triangles.append(tree_triangles)
    return all_trees_triangles


def FAST_generator(forest_triangles, fh, hybrid, int16_flags):
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


def generate_header(trees, n_classes, sample_size, test_rows, maxv, minv, int16_flags):
    m_type = "int16_t" if any(int16_flags) else "int8_t"
    max_str = ", ".join(f"{m}f" for m in maxv) if isinstance(maxv, list) else f"{maxv}f"
    min_str = ", ".join(f"{m}f" for m in minv) if isinstance(minv, list) else f"{minv}f"
    max_len = len(maxv) if isinstance(maxv, list) else 1
    min_len = len(minv) if isinstance(minv, list) else 1

    return f"""\
#ifndef INC_FAST_H_
#define INC_FAST_H_

#include "arm_mve.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdbool.h"
#include "math.h"
#include "stdint.h"

#define TREES          {trees}
#define CLASSES        {int(n_classes)}
#define DELTA          32767
#define SAMPLE_SIZE    {sample_size}
#define TEST_ROWS      {test_rows}

const float MAX[{max_len}] = {{{max_str}}};
const float MIN[{min_len}] = {{{min_str}}};

#define DTCM __attribute__((section(".dtcm_data"), aligned(32)))
#define RAM_BIG __attribute__((section(".big_data"), aligned(32)))

int16_t classes[TREES] = {{0}};

typedef struct Triangle{{
	int16_t thresholds[8];
	uint16_t features[8];
	{m_type} M[128];
}}Triangle;

static inline float dequantize_uint16(uint16_t q, float min, float max) {{
    float range = max - min;
    return ((float)q / 65535.0f) * range + min;
}}

static inline void softmax(const float *logits, float *probs, uint16_t num_classes) {{
    float max_logit = logits[0];
    for (uint16_t i = 1; i < num_classes; ++i) {{
        if (logits[i] > max_logit)
            max_logit = logits[i];
    }}
    float sum = 0.0f;
    for (uint16_t i = 0; i < num_classes; ++i) {{
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }}
    float inv_sum = 1.0f / sum;
    for (uint16_t i = 0; i < num_classes; ++i) {{
        probs[i] *= inv_sum;
    }}
}}

static inline float sigmoidf(float z) {{
    if (z >= 0) {{
        float ez = expf(-z);
        return 1.0f / (1.0f + ez);
    }} else {{
        float ez = expf(z);
        return ez / (1.0f + ez);
    }}
}}

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


def generate_inference(fh, trees, hybrid):
    fh.write("static inline void inference(int16_t* sample){\n")
    for i in range(trees):
        tree_name = f"tree_dtcm{i}" if (hybrid == -1 or i < hybrid) else f"tree{i}"
        fh.write(f"   classes[{i}] = fast_v2({tree_name}_triangles, sample);\n")
    fh.write("}\n\n")


def internal_nodes_calc(path, task, n_classes, acc, mae, mse, r2):
    model = joblib.load(path)
    booster = model.get_booster()
    dump = booster.get_dump()

    if n_classes > 2:
        # Reorder logic from DT_Rec_XGBoost
        grouped = {c: [] for c in range(n_classes)}
        for i, tree_text in enumerate(dump):
            rnd, cls = i // n_classes, i % n_classes
            grouped[cls].append((rnd, tree_text))
        dump = []
        for cls in range(n_classes):
            for rnd, tree_text in grouped[cls]:
                dump.append(tree_text)

    # Simplified min/max calc (mirroring DT_Rec_XGBoost)
    re_leaf = re.compile(r"leaf=(?P<leaf>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)")
    if n_classes > 2:
        trees_per_class = len(dump) // n_classes
        max_vector = []
        min_vector = []
        for c in range(n_classes):
            c_leaves = []
            for t_idx in range(c * trees_per_class, (c + 1) * trees_per_class):
                c_leaves.extend(
                    [float(m.group("leaf")) for m in re_leaf.finditer(dump[t_idx])]
                )
            max_vector.append(max(c_leaves))
            min_vector.append(min(c_leaves))
        range_vector = [max_vector[i] - min_vector[i] for i in range(n_classes)]
    else:
        all_leaves = [float(m.group("leaf")) for m in re_leaf.finditer(" ".join(dump))]
        max_vector = max(all_leaves)
        min_vector = min(all_leaves)
        range_vector = max_vector - min_vector

    forest_triangles = fast_initialization(
        dump, min_vector, range_vector, task, n_classes
    )

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
        tri_size = 32 + (256 if is_int16 else 128)
        size_bytes += num_triangles * tri_size
        if size_bytes > 130000 and hybrid == -1:
            hybrid = i

    print(f"Model Stats:")
    print(f"  - Total Triangles: {total_triangles}")
    print(f"  - Dimension: {size_bytes} Bytes ({size_bytes / 1024:.2f} KB)")

    return forest_triangles, hybrid, int16_flags, max_vector, min_vector


def generate_fast(path, n_trees, random_seed, number_of_test_samples):
    (
        test_path,
        joblib_path,
        sample_size,
        task,
        acc,
        mse,
        mae,
        r2,
        csv_stem,
        n_classes,
    ) = trainXGB.training(path, n_trees, random_seed, number_of_test_samples)

    forest_triangles, hybrid, int16_flags, maxv, minv = internal_nodes_calc(
        joblib_path, task, n_classes, acc, mae, mse, r2
    )

    # Get max_depth from model for path naming
    model = joblib.load(joblib_path)
    max_depth = model.get_params()["max_depth"]

    models_path = Path(f"Models/{csv_stem}/fast_XGB_T{n_trees}_D{max_depth}")
    os.makedirs(models_path, exist_ok=True)

    suffix = "_hybrid" if hybrid != -1 else ""
    header_name = (
        f"{csv_stem}_XGB_fast{suffix}_T{n_trees}_D{max_depth}_RS{random_seed}.h"
    )
    header_path = models_path / header_name

    header = generate_header(
        len(forest_triangles),
        n_classes,
        sample_size,
        number_of_test_samples,
        maxv,
        minv,
        int16_flags,
    )
    kernel = generate_kernel()

    with open(header_path, "w") as f:
        f.write(header)
        FAST_generator(forest_triangles, f, hybrid, int16_flags)
        f.write(kernel)
        generate_inference(f, len(forest_triangles), hybrid)
        testset_gen(test_path, f, number_of_test_samples)
        f.write("#endif\n")

    print(f"Code successfully generated in {header_path}")
