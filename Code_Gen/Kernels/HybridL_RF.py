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
from collections import deque
from pathlib import Path

import joblib
import numpy as np

import Code_Gen.Utils.trainRF as trainRF


def get_nodes_by_level(tree):
    children_left = tree.children_left
    children_right = tree.children_right
    levels = []
    queue = deque([(0, 0)])  # (node_id, level)
    while queue:
        node_id, level = queue.popleft()
        if level == len(levels):
            levels.append([])
        levels[level].append(node_id)
        if children_left[node_id] != -1:
            queue.append((children_left[node_id], level + 1))
        if children_right[node_id] != -1:
            queue.append((children_right[node_id], level + 1))
    return levels


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


def simulate_path(tree, tri_nodes, bitmask, root_to_id, current_tri_id, task):
    curr_idx_in_tri = 0
    while True:
        node_id = tri_nodes[curr_idx_in_tri]
        if node_id == -1:
            return 0
        if tree.children_left[node_id] == -1:
            if task == 0:
                val = int(np.argmax(tree.value[node_id][0]))
            else:
                val = int(tree.value[node_id][0][0])
            return -val
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
                if task == 0:
                    val = int(np.argmax(tree.value[target_node][0]))
                else:
                    val = int(tree.value[target_node][0][0])
                return -val
            else:
                if target_node in root_to_id:
                    return int(root_to_id[target_node] - current_tri_id)
                else:
                    return 0


def fast_initialization(model, task):
    DELTA = 32767
    all_trees_triangles = []
    for estimator in model.estimators_:
        tree = estimator.tree_
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
                    tree, tri_nodes, bitmask, root_to_id, i, task
                )
            tree_triangles.append(
                {"features": feats, "thresholds": threshs, "M": m_matrix}
            )
        all_trees_triangles.append(tree_triangles)
    return all_trees_triangles


def SIMT_Generator(group_levels, group_idx, section, fh, parallelism, task):
    DELTA = 32767
    max_depth = max(len(levels) for (_, levels) in group_levels)
    num_nodes = 0
    num_added_nodes = 0
    feature_list = []
    threshold_list = []
    childL_list = []
    childR_list = []
    for level_idx in range(max_depth):
        for tree_idx, (tree, levels) in enumerate(group_levels):
            if level_idx < len(levels):
                node_ids = levels[level_idx]
                for node_id in node_ids:
                    feature = tree.feature[node_id]
                    threshold = tree.threshold[node_id]
                    left = tree.children_left[node_id]
                    right = tree.children_right[node_id]
                    if left == right == -1:
                        class_idx = (
                            int(np.argmax(tree.value[node_id][0])) if task == 0 else 1
                        )
                        feature_list.append(np.uint16(0))
                        threshold_list.append(np.int16(DELTA))
                        childL_list.append(np.uint16(num_nodes))
                        childR_list.append(np.uint16(class_idx))
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
    fh.write(
        f"{section} uint16_t feature{suffix}{group_idx}[{len(feature_list)}] = {{{', '.join(map(str, feature_list))}}};\n"
    )
    fh.write(
        f"{section} int16_t threshold{suffix}{group_idx}[{len(threshold_list)}] = {{{', '.join(map(str, threshold_list))}}};\n"
    )
    fh.write(
        f"{section} uint16_t childL{suffix}{group_idx}[{len(childL_list)}] = {{{', '.join(map(str, childL_list))}}};\n"
    )
    fh.write(
        f"{section} uint16_t childR{suffix}{group_idx}[{len(childR_list)}] = {{{', '.join(map(str, childR_list))}}};\n\n"
    )


def DTRec_Generator(tree, tree_idx, section, fh, task):
    node_count = tree.node_count
    orig_to_internal = [-1] * node_count
    internal_idx = 0
    for node_id in range(node_count):
        if tree.children_left[node_id] != -1:
            orig_to_internal[node_id] = internal_idx
            internal_idx += 1
    num_internal = internal_idx
    struct_init_list = []
    suffix = "_dtcm" if section == "DTCM" else ""
    tree_name = f"tree{suffix}{tree_idx}"
    for node_id in range(node_count):
        if tree.children_left[node_id] == -1:
            continue
        feat = tree.feature[node_id]
        thresh = int(tree.threshold[node_id])
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        if tree.children_left[left_child] == -1:
            left_ptr = "&delta"
            left_class = int(np.argmax(tree.value[left_child][0])) if task == 0 else 1
        else:
            left_ptr = f"&{tree_name}[{orig_to_internal[left_child]}]"
            left_class = 0
        if tree.children_left[right_child] == -1:
            right_ptr = "&delta"
            right_class = int(np.argmax(tree.value[right_child][0])) if task == 0 else 1
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
    parallelism,
    sample_size,
    test_rows,
    has_simt,
    has_dtrec,
    has_fast,
    fast_int16,
):
    m_type = "int16_t" if fast_int16 else "int8_t"
    header = f"""\
#ifndef INC_HYBRID_H_
#define INC_HYBRID_H_

#include "arm_mve.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdbool.h"

#define TREES          {trees}
#define DELTA          32767
#define PARALLELISM    {parallelism}
#define SAMPLE_SIZE    {sample_size}
#define TEST_ROWS      {test_rows}

#define DTCM __attribute__((section(".dtcm_data"), aligned(32)))
#define RAM_BIG __attribute__((section(".big_data"), aligned(32)))

uint16_t final_results[TREES];

"""
    if has_simt:
        header += """
typedef struct{
    uint16_t class[PARALLELISM];
}classes_simt;
"""
    if has_dtrec:
        header += """
typedef struct Node_Rec{
	uint16_t feature;
	int16_t  threshold;
	struct Node_Rec *left;
	struct Node_Rec *right;
	int16_t left_class;
	int16_t right_class;
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
    fh, simt_dtcm, simt_cache, dtrec_dtcm, dtrec_sram, fast_sram, parallelism
):
    fh.write("static inline void inference(int16_t* sample){\n")
    fh.write("   uint16_t i = 0;\n")
    if simt_dtcm or simt_cache:
        fh.write("   classes_simt myclass;\n")
    for g_idx in simt_dtcm:
        fh.write(
            f"   myclass = kernel_simt(threshold_dtcm{g_idx}, feature_dtcm{g_idx}, childL_dtcm{g_idx}, childR_dtcm{g_idx}, sample);\n"
        )
        fh.write(
            f"   memcpy(&final_results[i], myclass.class, sizeof(uint16_t) * PARALLELISM);\n"
        )
        fh.write(f"   i += PARALLELISM;\n\n")
    for g_idx in simt_cache:
        fh.write(
            f"   myclass = kernel_simt(threshold{g_idx}, feature{g_idx}, childL{g_idx}, childR{g_idx}, sample);\n"
        )
        fh.write(
            f"   memcpy(&final_results[i], myclass.class, sizeof(uint16_t) * PARALLELISM);\n"
        )
        fh.write(f"   i += PARALLELISM;\n\n")
    for t_idx in dtrec_dtcm:
        fh.write(f"   final_results[i++] = dt_rec(tree_dtcm{t_idx}, sample);\n")
    for t_idx in dtrec_sram:
        fh.write(f"   final_results[i++] = dt_rec(tree{t_idx}, sample);\n")
    for t_idx in fast_sram:
        fh.write(f"   final_results[i++] = fast_v2(tree{t_idx}_triangles, sample);\n")
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


def generate_hybrid_rf(
    path, number_of_trees, max_depth, random_seed, parallelism, number_of_test_samples
):
    test_path, joblib_path, sample_size, task, accuracy, mse, mae, r2, csv_stem = (
        trainRF.training(
            path, number_of_trees, max_depth, random_seed, number_of_test_samples
        )
    )
    model = joblib.load(joblib_path)
    estimators = model.estimators_
    num_estimators = len(estimators)
    num_groups = num_estimators // parallelism
    DTCM_LIMIT = 128 * 1024
    CACHE_LIMIT = 32 * 1024
    DTREC_THRESHOLD = 128 * 1024
    simt_dtcm_groups, simt_cache_groups = [], []
    dtrec_dtcm_trees, dtrec_sram_trees = [], []
    fast_sram_trees = []
    current_dtcm, current_cache = 0, 0
    for g_idx in range(num_groups):
        group_trees = estimators[g_idx * parallelism : (g_idx + 1) * parallelism]
        group_size = sum(t.tree_.node_count for t in group_trees) * 8
        if current_dtcm + group_size <= DTCM_LIMIT:
            simt_dtcm_groups.append(g_idx)
            current_dtcm += group_size
        else:
            # Check if all remaining trees fit in Cache using SIMT
            remaining_simt_weight = (
                sum(t.tree_.node_count for t in estimators[g_idx * parallelism :]) * 8
            )
            if remaining_simt_weight <= CACHE_LIMIT:
                for g_idx2 in range(g_idx, num_groups):
                    simt_cache_groups.append(g_idx2)
                current_cache = remaining_simt_weight
            break
    simt_trees_count = (len(simt_dtcm_groups) + len(simt_cache_groups)) * parallelism
    remaining_trees_start = simt_trees_count
    if remaining_trees_start < num_estimators:
        rem_trees = estimators[remaining_trees_start:]
        # First, try to fit as many trees as possible into DTCM using DT-Rec
        for i, t in enumerate(rem_trees):
            t_idx = remaining_trees_start + i
            t_size = (t.tree_.node_count - t.tree_.n_leaves) * 16
            if current_dtcm + t_size <= DTCM_LIMIT:
                dtrec_dtcm_trees.append(t_idx)
                current_dtcm += t_size
            else:
                # The rest will go to SRAM. We decide between DT-Rec and FAST for these specific trees.
                sram_rem_trees = rem_trees[i:]
                dtrec_sram_total_size = sum(
                    (t2.tree_.node_count - t2.tree_.n_leaves) * 16
                    for t2 in sram_rem_trees
                )

                if dtrec_sram_total_size <= DTREC_THRESHOLD:
                    dtrec_sram_trees = list(range(t_idx, num_estimators))
                else:
                    fast_sram_trees = list(range(t_idx, num_estimators))
                break

    simt_dtcm_weight = (
        sum(
            sum(
                estimators[g * parallelism : (g + 1) * parallelism][i].tree_.node_count
                for i in range(parallelism)
            )
            for g in simt_dtcm_groups
        )
        * 8
    )
    simt_cache_weight = (
        sum(
            sum(
                estimators[g * parallelism : (g + 1) * parallelism][i].tree_.node_count
                for i in range(parallelism)
            )
            for g in simt_cache_groups
        )
        * 8
    )
    dtrec_dtcm_weight = sum(
        (estimators[t].tree_.node_count - estimators[t].tree_.n_leaves) * 16
        for t in dtrec_dtcm_trees
    )
    dtrec_sram_weight = sum(
        (estimators[t].tree_.node_count - estimators[t].tree_.n_leaves) * 16
        for t in dtrec_sram_trees
    )
    fast_weight, fast_int16 = 0, False
    if fast_sram_trees:
        forest_triangles = fast_initialization(model, task)
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
    models_path = Path(f"Models/{csv_stem}/hybrid_L_RF_T{number_of_trees}_D{max_depth}")
    os.makedirs(models_path, exist_ok=True)
    header_name = (
        f"{csv_stem}_RF_hybrid_T{number_of_trees}_D{max_depth}_RS{random_seed}.h"
    )
    header_path = models_path / header_name
    with open(header_path, "w") as f:
        f.write(
            generate_header(
                num_estimators,
                parallelism,
                sample_size,
                number_of_test_samples,
                (simt_dtcm_groups or simt_cache_groups),
                (dtrec_dtcm_trees or dtrec_sram_trees),
                bool(fast_sram_trees),
                fast_int16,
            )
        )
        forest_levels = []
        for estimator in estimators:
            levels = get_nodes_by_level(estimator.tree_)
            forest_levels.append((estimator.tree_, levels))
        for g_idx in simt_dtcm_groups:
            group = forest_levels[g_idx * parallelism : (g_idx + 1) * parallelism]
            SIMT_Generator(group, g_idx, "DTCM", f, parallelism, task)
        for g_idx in simt_cache_groups:
            group = forest_levels[g_idx * parallelism : (g_idx + 1) * parallelism]
            SIMT_Generator(group, g_idx, "RAM_BIG", f, parallelism, task)
        for t_idx in dtrec_dtcm_trees:
            DTRec_Generator(estimators[t_idx].tree_, t_idx, "DTCM", f, task)
        for t_idx in dtrec_sram_trees:
            DTRec_Generator(estimators[t_idx].tree_, t_idx, "RAM_BIG", f, task)
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
        )
        testset_gen(test_path, f, number_of_test_samples)
        f.write("#endif\n")
    print(f"Hybrid code (L-Version) successfully generated in {header_path}")
