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
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

import Code_Gen.Utils.trainXGB as trainXGB

DELTA = 32767


class Tree:
    def __init__(self, cl, cr, f, t, v, n_nodes):
        self.children_left = cl
        self.children_right = cr
        self.feature = f
        self.threshold = t
        self.value = v
        self.node_count = n_nodes


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
                threshold[i] = DELTA
            else:
                children_left[i] = nodes[i]["left"]
                children_right[i] = nodes[i]["right"]
                feature[i] = nodes[i]["feature"]
                threshold[i] = nodes[i]["threshold"]

    return Tree(children_left, children_right, feature, threshold, value, max_id + 1)


def get_nodes_by_level(tree):
    children_left = tree.children_left
    children_right = tree.children_right
    threshold = tree.threshold
    levels = []
    queue = deque([(0, 0)])  # (node_id, level)
    while queue:
        node_id, level = queue.popleft()
        if level == len(levels):
            levels.append([])
        levels[level].append(node_id)
        if threshold[node_id] != DELTA:
            queue.append((children_left[node_id], level + 1))
            queue.append((children_right[node_id], level + 1))
    return levels


def get_triangle_node_indices(tree, root_idx):
    nodes = [-1] * 7
    nodes[0] = root_idx
    if nodes[0] != -1 and tree.threshold[nodes[0]] != DELTA:
        nodes[1] = tree.children_left[nodes[0]]
        nodes[2] = tree.children_right[nodes[0]]
    if nodes[1] != -1 and tree.threshold[nodes[1]] != DELTA:
        nodes[3] = tree.children_left[nodes[1]]
        nodes[4] = tree.children_right[nodes[1]]
    if nodes[2] != -1 and tree.threshold[nodes[2]] != DELTA:
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
        if tree.threshold[node_id] == DELTA:
            val = tree.value[node_id]
            if task == 0:
                q = np.round((val - min_v) / range_v * 65535).astype(np.uint16)
            else:
                q = 1
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
            if tree.threshold[target_node] == DELTA:
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


def fast_initialization(trees, min_vector, range_vector, task, n_classes):
    all_trees_triangles = []
    trees_per_class = len(trees) // n_classes if n_classes > 2 else len(trees)
    for tree_idx, tree in enumerate(trees):
        class_idx = tree_idx // trees_per_class if n_classes > 2 else 0
        min_v = min_vector[class_idx] if isinstance(min_vector, list) else min_vector
        range_v = (
            range_vector[class_idx] if isinstance(range_vector, list) else range_vector
        )
        triangle_roots = []
        if tree.threshold[0] != DELTA:
            queue = deque([0])
            seen = {0}
            while queue:
                root = queue.popleft()
                triangle_roots.append(root)
                tri_nodes = get_triangle_node_indices(tree, root)
                for i in [3, 4, 5, 6]:
                    node_idx = tri_nodes[i]
                    if node_idx != -1 and tree.threshold[node_idx] != DELTA:
                        for child in [
                            tree.children_left[node_idx],
                            tree.children_right[node_idx],
                        ]:
                            if child != -1 and tree.threshold[child] != DELTA:
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
                if node_idx != -1 and tree.threshold[node_idx] != DELTA:
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


def SIMT_Generator(
    group_levels,
    group_idx,
    section,
    fh,
    parallelism,
    task,
    min_vector,
    range_vector,
    n_classes,
    tree_start_idx,
):
    trees_per_class = len(min_vector) * len(
        group_levels
    )  # This is not quite right, but we need the class_idx
    # We'll calculate class_idx based on the absolute tree index in the forest

    max_depth = max(len(levels) for (_, levels) in group_levels)
    num_nodes = 0
    num_added_nodes = 0
    feature_list = []
    threshold_list = []
    childL_list = []
    childR_list = []

    total_trees = n_classes * (
        len(group_levels) * 10
    )  # Dummy large value to be safe, actual trees per class is better
    # Actually, we know n_classes and we can find trees_per_class if we pass it.

    for level_idx in range(max_depth):
        for local_tree_idx, (tree, levels) in enumerate(group_levels):
            abs_tree_idx = tree_start_idx + local_tree_idx
            if n_classes > 2:
                # We need to know how many trees are there in total per class
                # Assuming trees are ordered class by class as in reorder_dump_by_class_text
                # This should be passed from generate_hybrid_xgb
                pass

            if level_idx < len(levels):
                node_ids = levels[level_idx]
                for node_id in node_ids:
                    feature = tree.feature[node_id]
                    threshold = tree.threshold[node_id]
                    left = tree.children_left[node_id]
                    right = tree.children_right[node_id]

                    if threshold == DELTA:
                        val = tree.value[node_id]
                        class_idx = abs_tree_idx // (
                            len(group_levels) * 10
                        )  # Place holder
                        # Actually class_idx is already handled if we reorder the dump.
                        # The value in tree.value for XGBoost is the logit.
                        # We need to quantize it.

                        # Wait, we already have quantized values if we use the "set_model" style trees.
                        # But here I'm using the parse_xgb_tree result.
                        # Let's pass class_idx or the min/range vectors directly.

                        # Better: calculate class_idx outside and pass it in or rely on tree_start_idx
                        # In multiclass, trees are [C0_T0, C0_T1, ..., C1_T0, ...]
                        # So class_idx = abs_tree_idx // trees_per_class

                        # Let's assume trees_per_class is passed.
                        tpc = (
                            total_trees_in_forest // n_classes
                            if n_classes > 2
                            else total_trees_in_forest
                        )
                        c_idx = abs_tree_idx // tpc if n_classes > 2 else 0
                        mv = (
                            min_vector[c_idx]
                            if isinstance(min_vector, list)
                            else min_vector
                        )
                        rv = (
                            range_vector[c_idx]
                            if isinstance(range_vector, list)
                            else range_vector
                        )

                        if task == 0:
                            q = np.round((val - mv) / rv * 65535).astype(np.uint16)
                        else:
                            q = 1

                        feature_list.append(np.uint16(0))
                        threshold_list.append(np.int16(DELTA))
                        childL_list.append(np.uint16(num_nodes))
                        childR_list.append(np.uint16(q))
                        num_nodes += 1
                    else:
                        feature_list.append(np.uint16(feature))
                        threshold_list.append(np.int16(threshold))
                        childL_list.append(
                            np.uint16((2 * num_added_nodes) + parallelism)
                        )
                        childR_list.append(
                            np.uint16((2 * num_added_nodes) + parallelism + 1)
                        )
                        num_nodes += 1
                        num_added_nodes += 1

    suffix = "_dtcm" if section == "DTCM" else ""
    # Extract class and group for naming if multiclass
    if n_classes > 2:
        tpc = total_trees_in_forest // n_classes
        c_idx = tree_start_idx // tpc
        g_idx = (tree_start_idx % tpc) // parallelism
        name_suffix = f"_{c_idx}_{g_idx}"
    else:
        name_suffix = f"{group_idx}"

    fh.write(
        f"{section} uint16_t feature{suffix}{name_suffix}[{len(feature_list)}] = {{{', '.join(map(str, feature_list))}}};\n"
    )
    fh.write(
        f"{section} int16_t threshold{suffix}{name_suffix}[{len(threshold_list)}] = {{{', '.join(map(str, threshold_list))}}};\n"
    )
    fh.write(
        f"{section} uint16_t childL{suffix}{name_suffix}[{len(childL_list)}] = {{{', '.join(map(str, childL_list))}}};\n"
    )
    fh.write(
        f"{section} uint16_t childR{suffix}{name_suffix}[{len(childR_list)}] = {{{', '.join(map(str, childR_list))}}};\n\n"
    )


# Global for SIMT_Generator to know forest size
total_trees_in_forest = 0


def DTRec_Generator(tree, tree_idx, section, fh, task, min_v, range_v):
    node_count = tree.node_count
    orig_to_internal = [-1] * node_count
    internal_idx = 0
    for node_id in range(node_count):
        if tree.threshold[node_id] != DELTA:
            orig_to_internal[node_id] = internal_idx
            internal_idx += 1
    num_internal = internal_idx
    struct_init_list = []
    suffix = "_dtcm" if section == "DTCM" else ""
    tree_name = f"tree{suffix}{tree_idx}"
    for node_id in range(node_count):
        if tree.threshold[node_id] == DELTA:
            continue
        feat = tree.feature[node_id]
        thresh = int(tree.threshold[node_id])
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        if tree.threshold[left_child] == DELTA:
            left_ptr = "&delta"
            val = tree.value[left_child]
            if task == 0:
                left_class = np.round((val - min_v) / range_v * 65535).astype(np.uint16)
            else:
                left_class = 1
        else:
            left_ptr = f"&{tree_name}[{orig_to_internal[left_child]}]"
            left_class = 0

        if tree.threshold[right_child] == DELTA:
            right_ptr = "&delta"
            val = tree.value[right_child]
            if task == 0:
                right_class = np.round((val - min_v) / range_v * 65535).astype(
                    np.uint16
                )
            else:
                right_class = 1
        else:
            right_ptr = f"&{tree_name}[{orig_to_internal[right_child]}]"
            right_class = 0

        struct_init_list.append(
            f"{{ {feat}, {thresh}, {left_ptr}, {right_ptr}, {left_class}, {right_class} }}"
        )
    fh.write(
        f"{section} Node_Rec {tree_name}[{num_internal}] = {{{', '.join(struct_init_list)}}};\n\n"
    )


def FAST_Generator(tree_triangles, tree_idx, section, fh, is_int16):
    suffix = "_dtcm" if section == "DTCM" else ""
    tree_name = f"tree{suffix}{tree_idx}"
    m_type = "int16_t" if is_int16 else "int8_t"
    fh.write(f"{section} Triangle {tree_name}_triangles[{len(tree_triangles)}] = {{\n")
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


def generate_header(
    trees,
    n_classes,
    parallelism,
    sample_size,
    test_rows,
    has_simt,
    has_dtrec,
    has_fast,
    fast_int16,
    maxv,
    minv,
):
    m_type = "int16_t" if fast_int16 else "int8_t"
    max_str = ", ".join(f"{m}f" for m in maxv) if isinstance(maxv, list) else f"{maxv}f"
    min_str = ", ".join(f"{m}f" for m in minv) if isinstance(minv, list) else f"{minv}f"
    max_len = len(maxv) if isinstance(maxv, list) else 1
    min_len = len(minv) if isinstance(minv, list) else 1

    header = f"""\
#ifndef INC_HYBRID_H_
#define INC_HYBRID_H_

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
#define PARALLELISM    {parallelism}
#define SAMPLE_SIZE    {sample_size}
#define TEST_ROWS      {test_rows}

const float MAX[{max_len}] = {{{max_str}}};
const float MIN[{min_len}] = {{{min_str}}};

#define DTCM __attribute__((section(".dtcm_data"), aligned(32)))
#define RAM_BIG __attribute__((section(".big_data"), aligned(32)))

int16_t classes[TREES] = {{0}};

"""
    if has_simt:
        header += """
typedef struct{
    uint16_t class[PARALLELISM];
}classes_simt;
"""
    if n_classes > 2:
        header += f"uint16_t final_results[CLASSES][TREES / CLASSES];\n"
    else:
        header += f"uint16_t final_results[TREES];\n"

    if has_dtrec:
        header += """
typedef struct Node_Rec{
	uint16_t feature;
	int16_t  threshold;
	struct Node_Rec *left;
	struct Node_Rec *right;
	uint16_t left_class;
	uint16_t right_class;
}Node_Rec;

Node_Rec delta;
"""
    if has_fast:
        header += f"""
typedef struct Triangle{{
	int16_t thresholds[8];
	uint16_t features[8];
	{m_type} M[128];
}}Triangle;
"""
    header += """
static inline float dequantize_uint16(uint16_t q, float min, float max) {
    float range = max - min;
    return ((float)q / 65535.0f) * range + min;
}

static inline void softmax(const float *logits, float *probs, uint16_t num_classes) {
    float max_logit = logits[0];
    for (uint16_t i = 1; i < num_classes; ++i) {
        if (logits[i] > max_logit)
            max_logit = logits[i];
    }
    float sum = 0.0f;
    for (uint16_t i = 0; i < num_classes; ++i) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    float inv_sum = 1.0f / sum;
    for (uint16_t i = 0; i < num_classes; ++i) {
        probs[i] *= inv_sum;
    }
}

static inline float sigmoidf(float z) {
    if (z >= 0) {
        float ez = expf(-z);
        return 1.0f / (1.0f + ez);
    } else {
        float ez = expf(z);
        return ez / (1.0f + ez);
    }
}
"""
    return header


def generate_kernels(has_simt, has_dtrec, has_fast):
    kernels = ""
    if has_simt:
        kernels += """
bool all_active(mve_pred16_t mask){
	return (mask == 0xFFFF);
}

classes_simt kernel_simt(int16_t* threshold, uint16_t* features, uint16_t* childL, uint16_t* childR, int16_t* sample){
	uint16x8_t a = vld1q_u16(&childL[0]);
	uint16x8_t b = vld1q_u16(&childR[0]);
	int16x8_t thresh = vld1q_s16(&threshold[0]);
	uint16x8_t samp_offset = vld1q_u16(&features[0]);
	int16x8_t samp = vldrhq_gather_shifted_offset_s16(sample, samp_offset);
	mve_pred16_t mask = vcmpeqq_u16((uint16x8_t)vdupq_n_u16(0), (uint16x8_t)vdupq_n_u16(1));
	while(!all_active(mask)){
		mve_pred16_t pred = vcmpleq_s16(samp, thresh);
		uint16x8_t dst = vpselq_u16(a, b, pred);
		thresh = vldrhq_gather_shifted_offset_s16(threshold, dst);
		mask = vcmpeqq_n_s16(thresh, DELTA);
		samp_offset = vldrhq_gather_shifted_offset_u16(features, dst);
		samp = vldrhq_gather_shifted_offset_s16(sample, samp_offset);
		a = vldrhq_gather_shifted_offset_u16(childL, dst);
		b = vldrhq_gather_shifted_offset_u16(childR, dst);
	}
	classes_simt result;
	vst1q_u16(result.class, b);
	return result;
}
"""
    if has_dtrec:
        kernels += """
int16_t dt_rec(Node_Rec *node, int16_t *sample){
	if(sample[node->feature] <= node->threshold){
		if(node->left != &delta){
			return dt_rec(node->left, sample);
		}else{
			return node->left_class;
		}
	}else{
		if(node->right != &delta){
			return dt_rec(node->right, sample);
		}else{
			return node->right_class;
		}
	}
}
"""
    if has_fast:
        kernels += """
int16_t fast_v2(Triangle *arr, int16_t *sample) {
    static const int16x8_t twos = {1, 2, 4, 8, 16, 32, 64, 0};
    int16_t next_index = 1;
    Triangle *tri = arr;
    while (next_index > 0) {
        int16x8_t thresh = vld1q_s16(tri->thresholds);
        uint16x8_t samp_offset = vld1q_u16(tri->features);
        int16x8_t samp = vldrhq_gather_shifted_offset_s16(sample, samp_offset);
        mve_pred16_t pred = vcmpleq_s16(samp, thresh);
        int16x8_t dst = vpselq_s16(vdupq_n_s16(1), vdupq_n_s16(0), pred);
		int32_t index = vmladavaq_s16(0, dst, twos);
        next_index = tri->M[index];
        tri += next_index;
    }
    return -next_index;
}
"""
    return kernels


def generate_inference(
    fh, simt_dtcm, simt_cache, dtrec_dtcm, dtrec_sram, fast_sram, parallelism, n_classes
):
    fh.write("static inline void inference(int16_t* sample){\n")
    if n_classes > 2:
        fh.write("   uint16_t i = 0;\n")
        fh.write("   uint16_t c = 0;\n")
        trees_per_class = total_trees_in_forest // n_classes
    else:
        fh.write("   uint16_t i = 0;\n")

    if simt_dtcm or simt_cache:
        fh.write("   classes_simt myclass;\n")

    # We need to handle multiclass indexing correctly.
    # Trees are ordered by class.

    def write_call(t_idx, kernel_call, is_simt=False):
        nonlocal fh
        if n_classes > 2:
            tpc = total_trees_in_forest // n_classes
            c_idx = t_idx // tpc
            local_idx = t_idx % tpc
            if is_simt:
                fh.write(f"   myclass = {kernel_call};\n")
                fh.write(
                    f"   memcpy(&final_results[{c_idx}][{local_idx}], myclass.class, sizeof(uint16_t) * PARALLELISM);\n"
                )
            else:
                fh.write(f"   final_results[{c_idx}][{local_idx}] = {kernel_call};\n")
        else:
            if is_simt:
                fh.write(f"   myclass = {kernel_call};\n")
                fh.write(
                    f"   memcpy(&final_results[i], myclass.class, sizeof(uint16_t) * PARALLELISM);\n"
                )
                fh.write(f"   i += PARALLELISM;\n")
            else:
                fh.write(f"   final_results[i++] = {kernel_call};\n")

    for g_idx in simt_dtcm:
        t_start = g_idx * parallelism
        if n_classes > 2:
            tpc = total_trees_in_forest // n_classes
            c_idx = t_start // tpc
            local_g_idx = (t_start % tpc) // parallelism
            call = f"kernel_simt(threshold_dtcm_{c_idx}_{local_g_idx}, feature_dtcm_{c_idx}_{local_g_idx}, childL_dtcm_{c_idx}_{local_g_idx}, childR_dtcm_{c_idx}_{local_g_idx}, sample)"
        else:
            call = f"kernel_simt(threshold_dtcm{g_idx}, feature_dtcm{g_idx}, childL_dtcm{g_idx}, childR_dtcm{g_idx}, sample)"
        write_call(t_start, call, is_simt=True)

    for g_idx in simt_cache:
        t_start = g_idx * parallelism
        if n_classes > 2:
            tpc = total_trees_in_forest // n_classes
            c_idx = t_start // tpc
            local_g_idx = (t_start % tpc) // parallelism
            call = f"kernel_simt(threshold_{c_idx}_{local_g_idx}, feature_{c_idx}_{local_g_idx}, childL_{c_idx}_{local_g_idx}, childR_{c_idx}_{local_g_idx}, sample)"
        else:
            call = f"kernel_simt(threshold{g_idx}, feature{g_idx}, childL{g_idx}, childR{g_idx}, sample)"
        write_call(t_start, call, is_simt=True)

    for t_idx in dtrec_dtcm:
        write_call(t_idx, f"dt_rec(tree_dtcm{t_idx}, sample)")
    for t_idx in dtrec_sram:
        write_call(t_idx, f"dt_rec(tree{t_idx}, sample)")
    for t_idx in fast_sram:
        write_call(t_idx, f"fast_v2(tree{t_idx}_triangles, sample)")

    fh.write("}\n\n")


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


def reorder_dump_by_class_text(dump, num_class):
    grouped = {c: [] for c in range(num_class)}
    for i, tree_text in enumerate(dump):
        rnd, cls = i // num_class, i % num_class
        grouped[cls].append((rnd, tree_text))
    new_dump = []
    for cls in range(num_class):
        for rnd, tree_text in grouped[cls]:
            new_dump.append(tree_text)
    return new_dump


def generate_hybrid_xgb(
    path, number_of_trees, max_depth, random_seed, parallelism, number_of_test_samples
):
    global total_trees_in_forest
    (
        test_path,
        joblib_path,
        sample_size,
        task,
        accuracy,
        mse,
        mae,
        r2,
        csv_stem,
        n_classes,
    ) = trainXGB.training(path, number_of_trees, random_seed, number_of_test_samples)
    model = joblib.load(joblib_path)
    booster = model.get_booster()
    dump = booster.get_dump()
    if n_classes > 2:
        dump = reorder_dump_by_class_text(dump, n_classes)

    trees = [parse_xgb_tree(t) for t in dump]
    num_trees = len(trees)
    total_trees_in_forest = num_trees
    num_groups = num_trees // parallelism

    DTCM_LIMIT = 128 * 1024
    CACHE_LIMIT = 32 * 1024
    DTREC_THRESHOLD = 128 * 1024

    simt_dtcm_groups, simt_cache_groups = [], []
    dtrec_dtcm_trees, dtrec_sram_trees = [], []
    fast_sram_trees = []
    current_dtcm, current_cache = 0, 0

    # Min/Max calculation for quantization
    re_leaf = re.compile(r"leaf=(?P<leaf>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)")
    if n_classes > 2:
        trees_per_class = num_trees // n_classes
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

    for g_idx in range(num_groups):
        group_trees = trees[g_idx * parallelism : (g_idx + 1) * parallelism]
        group_size = sum(t.node_count for t in group_trees) * 8
        if current_dtcm + group_size <= DTCM_LIMIT:
            simt_dtcm_groups.append(g_idx)
            current_dtcm += group_size
        else:
            remaining_simt_weight = (
                sum(t.node_count for t in trees[g_idx * parallelism :]) * 8
            )
            if remaining_simt_weight <= CACHE_LIMIT:
                for g_idx2 in range(g_idx, num_groups):
                    simt_cache_groups.append(g_idx2)
                current_cache = remaining_simt_weight
            break

    simt_trees_count = (len(simt_dtcm_groups) + len(simt_cache_groups)) * parallelism
    remaining_trees_start = simt_trees_count

    if remaining_trees_start < num_trees:
        rem_trees = trees[remaining_trees_start:]
        for i, t in enumerate(rem_trees):
            t_idx = remaining_trees_start + i
            # For XGBoost internal nodes calculation
            num_internal = sum(1 for thresh in t.threshold if thresh != DELTA)
            t_size = num_internal * 16
            if current_dtcm + t_size <= DTCM_LIMIT:
                dtrec_dtcm_trees.append(t_idx)
                current_dtcm += t_size
            else:
                sram_rem_trees = rem_trees[i:]
                dtrec_sram_total_size = sum(
                    sum(1 for thresh in t2.threshold if thresh != DELTA) * 16
                    for t2 in sram_rem_trees
                )
                if dtrec_sram_total_size <= DTREC_THRESHOLD:
                    dtrec_sram_trees = list(range(t_idx, num_trees))
                else:
                    fast_sram_trees = list(range(t_idx, num_trees))
                break

    simt_dtcm_weight = (
        sum(
            sum(trees[g * parallelism + i].node_count for i in range(parallelism))
            for g in simt_dtcm_groups
        )
        * 8
    )
    simt_cache_weight = (
        sum(
            sum(trees[g * parallelism + i].node_count for i in range(parallelism))
            for g in simt_cache_groups
        )
        * 8
    )
    dtrec_dtcm_weight = sum(
        sum(1 for thresh in trees[t].threshold if thresh != DELTA) * 16
        for t in dtrec_dtcm_trees
    )
    dtrec_sram_weight = sum(
        sum(1 for thresh in trees[t].threshold if thresh != DELTA) * 16
        for t in dtrec_sram_trees
    )
    fast_weight, fast_int16 = 0, False
    if fast_sram_trees:
        forest_triangles = fast_initialization(
            trees, min_vector, range_vector, task, n_classes
        )
        for t_idx in fast_sram_trees:
            tree_triangles = forest_triangles[t_idx]
            is_int16 = any(
                max(tri["M"]) > 127 or min(tri["M"]) < -128 for tri in tree_triangles
            )
            if is_int16:
                fast_int16 = True
            tri_size = 32 + (256 if is_int16 else 128)
            fast_weight += len(tree_triangles) * tri_size

    print(f"Hybrid Model Weights (L-Version):")
    print(
        f"  - SIMT DTCM: {simt_dtcm_weight / 1024:.2f} KB -> {len(simt_dtcm_groups) * parallelism} Trees"
    )
    print(
        f"  - SIMT Cache: {simt_cache_weight / 1024:.2f} KB -> {len(simt_cache_groups) * parallelism} Trees"
    )
    if dtrec_dtcm_trees:
        print(
            f"  - DT-Rec DTCM: {dtrec_dtcm_weight / 1024:.2f} KB -> {len(dtrec_dtcm_trees)} Trees"
        )
    if dtrec_sram_trees:
        print(
            f"  - DT-Rec SRAM: {dtrec_sram_weight / 1024:.2f} KB -> {len(dtrec_sram_trees)} Trees"
        )
    if fast_sram_trees:
        print(
            f"  - FAST SRAM: {fast_weight / 1024:.2f} KB -> {len(fast_sram_trees)} Trees"
        )

    # Header generation
    # Get max_depth from model for path naming
    actual_max_depth = model.get_params()["max_depth"]
    models_path = Path(
        f"Models/{csv_stem}/hybrid_L_XGB_T{number_of_trees}_D{actual_max_depth}"
    )
    os.makedirs(models_path, exist_ok=True)
    header_name = f"{csv_stem}_XGB_hybrid_T{number_of_trees}_D{actual_max_depth}_RS{random_seed}.h"
    header_path = models_path / header_name

    with open(header_path, "w") as f:
        f.write(
            generate_header(
                num_trees,
                n_classes,
                parallelism,
                sample_size,
                number_of_test_samples,
                (simt_dtcm_groups or simt_cache_groups),
                (dtrec_dtcm_trees or dtrec_sram_trees),
                bool(fast_sram_trees),
                fast_int16,
                max_vector,
                min_vector,
            )
        )

        forest_levels = []
        for t in trees:
            levels = get_nodes_by_level(t)
            forest_levels.append((t, levels))

        for g_idx in simt_dtcm_groups:
            group = forest_levels[g_idx * parallelism : (g_idx + 1) * parallelism]
            SIMT_Generator(
                group,
                g_idx,
                "DTCM",
                f,
                parallelism,
                task,
                min_vector,
                range_vector,
                n_classes,
                g_idx * parallelism,
            )
        for g_idx in simt_cache_groups:
            group = forest_levels[g_idx * parallelism : (g_idx + 1) * parallelism]
            SIMT_Generator(
                group,
                g_idx,
                "RAM_BIG",
                f,
                parallelism,
                task,
                min_vector,
                range_vector,
                n_classes,
                g_idx * parallelism,
            )

        for t_idx in dtrec_dtcm_trees:
            tpc = num_trees // n_classes if n_classes > 2 else num_trees
            c_idx = t_idx // tpc if n_classes > 2 else 0
            mv = min_vector[c_idx] if isinstance(min_vector, list) else min_vector
            rv = range_vector[c_idx] if isinstance(range_vector, list) else range_vector
            DTRec_Generator(trees[t_idx], t_idx, "DTCM", f, task, mv, rv)
        for t_idx in dtrec_sram_trees:
            tpc = num_trees // n_classes if n_classes > 2 else num_trees
            c_idx = t_idx // tpc if n_classes > 2 else 0
            mv = min_vector[c_idx] if isinstance(min_vector, list) else min_vector
            rv = range_vector[c_idx] if isinstance(range_vector, list) else range_vector
            DTRec_Generator(trees[t_idx], t_idx, "RAM_BIG", f, task, mv, rv)

        if fast_sram_trees:
            for t_idx in fast_sram_trees:
                FAST_Generator(forest_triangles[t_idx], t_idx, "RAM_BIG", f, fast_int16)

        f.write(
            generate_kernels(
                (simt_dtcm_groups or simt_cache_groups),
                (dtrec_dtcm_trees or dtrec_sram_trees),
                bool(fast_sram_trees),
            )
        )
        generate_inference(
            f,
            simt_dtcm_groups,
            simt_cache_groups,
            dtrec_dtcm_trees,
            dtrec_sram_trees,
            fast_sram_trees,
            parallelism,
            n_classes,
        )
        testset_gen(test_path, f, number_of_test_samples)
        f.write("#endif\n")

    print(f"Hybrid XGBoost code (L-Version) successfully generated in {header_path}")
