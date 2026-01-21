# 
# Copyright (c) 2026 Lorenzo Abate.
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

import joblib
import csv
import numpy as np
import shutil
import os

from pathlib import Path
from collections import deque

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

def Parallel_generator(path, txt_name, num_group, fh, groups_hybrid, task=0):
    model = joblib.load(path)
    DELTA = 32767

    forest_levels = []
    
    for estimator in model.estimators_:
        tree = estimator.tree_
        levels = get_nodes_by_level(tree)
        forest_levels.append((tree, levels))

    groupindex = 0

    with open(txt_name, "w") as f:
        for i in range(0, len(forest_levels), num_group):
            group = forest_levels[i:i + num_group]
    
            max_depth = max(len(levels) for (_, levels) in group)
            
            num_nodes = 0
            num_added_nodes = 0
            
            feature_list = []
            threshold_list = []
            childL_list = []
            childR_list = []

            for level_idx in range(max_depth):
                #print(f"\n--- Livello {level_idx} del gruppo alberi {i}-{i+num_group-1} ---")
                for tree_idx, (tree, levels) in enumerate(group):
                    if level_idx < len(levels):
                        node_ids = levels[level_idx]
                        #print(f"  Albero {i + tree_idx}:")
                        for node_id in node_ids:
                            feature = tree.feature[node_id]
                            threshold = tree.threshold[node_id]
                            left = tree.children_left[node_id]
                            right = tree.children_right[node_id]
                            if(left == right == -1):
                                if task == 0:
                                    class_idx = int(np.argmax(tree.value[node_id][0]))
                                else:
                                    class_idx = 1
                                feature_list.append(np.uint16(0))
                                threshold_list.append(np.int16(DELTA))
                                childL_list.append(np.uint16(num_nodes))
                                childR_list.append(np.uint16(class_idx))
                                num_nodes += 1
                            else:
                                feature_list.append(np.uint16(feature))
                                threshold_list.append(np.int16(threshold))
                                childL_list.append(np.uint16((2*num_added_nodes) + num_group))
                                childR_list.append(np.uint16((2*num_added_nodes) + num_group + 1))
                                num_nodes += 1
                                num_added_nodes += 1

                            #print(f"Nodo {node_id}: feat={feature}, thresh={threshold:.4f}, sx={left}, dx={right}")
            
            f.write(f"Group {i}-{i+num_group-1}   Number of Nodes = {num_nodes}\n")
            feature_str = ", ".join(str(f) for f in feature_list)
            f.write(f"feature = {feature_str}\n")
            threshold_str = ", ".join(str(t) for t in threshold_list)
            f.write(f"threshold = {threshold_str}\n")
            childL_str = ", ".join(str(c) for c in childL_list)
            f.write(f"childL = {childL_str}\n")
            childR_str = ", ".join(str(c) for c in childR_list)
            f.write(f"childR = {childR_str}\n\n")

            if groups_hybrid == 0:     
                fh.write(f"RAM_BIG uint16_t feature{groupindex}[{len(feature_list)}] = {{{feature_str}}};\n")
                fh.write(f"RAM_BIG int16_t threshold{groupindex}[{len(threshold_list)}] = {{{threshold_str}}};\n")
                fh.write(f"RAM_BIG uint16_t childL{groupindex}[{len(childL_list)}] = {{{childL_str}}};\n")
                fh.write(f"RAM_BIG uint16_t childR{groupindex}[{len(childR_list)}] = {{{childR_str}}};\n")

                fh.write(f"DTCM uint16_t feature_dtcm{groupindex}[{len(feature_list)}] = {{{feature_str}}};\n")
                fh.write(f"DTCM int16_t threshold_dtcm{groupindex}[{len(threshold_list)}] = {{{threshold_str}}};\n")
                fh.write(f"DTCM uint16_t childL_dtcm{groupindex}[{len(childL_list)}] = {{{childL_str}}};\n")
                fh.write(f"DTCM uint16_t childR_dtcm{groupindex}[{len(childR_list)}] = {{{childR_str}}};\n\n")
                
                groupindex += 1
            elif groups_hybrid > 0:
                if(groupindex < groups_hybrid):
                    fh.write(f"DTCM uint16_t feature_dtcm{groupindex}[{len(feature_list)}] = {{{feature_str}}};\n")
                    fh.write(f"DTCM int16_t threshold_dtcm{groupindex}[{len(threshold_list)}] = {{{threshold_str}}};\n")
                    fh.write(f"DTCM uint16_t childL_dtcm{groupindex}[{len(childL_list)}] = {{{childL_str}}};\n")
                    fh.write(f"DTCM uint16_t childR_dtcm{groupindex}[{len(childR_list)}] = {{{childR_str}}};\n\n")

                    groupindex += 1

def DT_Rec_generator(path, num_internal_nodes, txt_name, fh, hybrid, simt_dtcm_trees, task=0):
    model = joblib.load(path)
    
    DELTA = 32767

    total_nodes = []

    index = 0
    with open(txt_name, "w") as f:
        for i, estimator in enumerate(model.estimators_):
            if i >= simt_dtcm_trees:
                tree = estimator.tree_

                features = tree.feature
                threshold = tree.threshold

                current_node = 0
                num_nodes = 0
                num_added_nodes = 0
                
                nodes = deque()
                nodes.append(current_node)

                feature_list = []
                threshold_list = []
                child_list = []

                while nodes:
                    current_node = nodes.popleft()
                    left_child = tree.children_left[current_node]
                    right_child = tree.children_right[current_node]

                    if(left_child == right_child == -1):
                        feature_list.append(np.uint16(0))
                        threshold_list.append(np.int16(DELTA))
                        if task == 0:
                            class_idx = int(np.argmax(tree.value[current_node][0]))
                        else:
                            class_idx = 1
                        child_list.append(np.int16(class_idx))  # child[2n] = class
                        child_list.append(np.int16(class_idx))  # child[2n+1] = placeholder

                        num_nodes += 1
                    else:
                        feature_list.append(np.uint16(features[current_node]))
                        threshold_list.append(np.int16(threshold[current_node]))
                        child_list.append(np.int16((2*num_added_nodes) + 1))
                        child_list.append(np.int16((2*num_added_nodes) + 2))

                        nodes.append(left_child)
                        nodes.append(right_child)

                        num_added_nodes += 1
                        num_nodes += 1

                f.write(f"Tree {i}   Number of Nodes = {num_nodes}\n")
                feature_str = ", ".join(str(f) for f in feature_list)
                f.write(f"feature = {feature_str}\n")
                threshold_str = ", ".join(str(t) for t in threshold_list)
                f.write(f"threshold = {threshold_str}\n")
                child_str = ", ".join(str(c) for c in child_list)
                f.write(f"child = {child_str}\n")

                fh.write(f"uint16_t Rec_feature{i}[{len(feature_list)}] = {{{feature_str}}};\n")
                fh.write(f"int16_t Rec_threshold{i}[{len(threshold_list)}] = {{{threshold_str}}};\n")
                fh.write(f"int16_t Rec_child{i}[{len(child_list)}] = {{{child_str}}};\n")

                if hybrid == 0:
                    fh.write(f"RAM_BIG Node_Rec tree{i}[{num_internal_nodes[i]}];\n")
                    fh.write(f"DTCM Node_Rec tree_dtcm{i}[{num_internal_nodes[i]}];\n\n")
                else:
                    if i >= hybrid:
                        fh.write(f"RAM_BIG Node_Rec tree{i}[{num_internal_nodes[index]}];\n\n")
                        total_nodes.append(len(feature_list))
                        index += 1
                    else:
                        fh.write(f"DTCM Node_Rec tree_dtcm{i}[{num_internal_nodes[index]}];\n\n")
                        total_nodes.append(len(feature_list))
                        index += 1

    return total_nodes         

def internal_nodes_calc(path, groups_hybrid, size_dtcm_simt):
    model = joblib.load(path)

    simt_trees = groups_hybrid * 8
    num_internal_nodes = []
    dtcm_dtrec_trees = 0
    size_bytes = 0
    size_dtcm_dtrec = 0
    size_ram_dtrec = 0
    ram_dtrec_trees = 0

    for i,estimator in enumerate(model):
        if i >= simt_trees:
            tree = estimator.tree_
            internal_nodes = tree.node_count - tree.n_leaves
            
            num_internal_nodes.append(internal_nodes)

    size_bytes = size_dtcm_simt
    if num_internal_nodes:
        for elem in num_internal_nodes:
            size_bytes += elem * 16

            if size_bytes <= 129000:
                size_dtcm_dtrec += elem * 16
                dtcm_dtrec_trees += 1
            elif size_bytes > 129000:
                size_ram_dtrec += elem * 16
                ram_dtrec_trees += 1

    #print(num_internal_nodes)

    print(f"  - Trees DTCM: {simt_trees} ({groups_hybrid} Groups) + {dtcm_dtrec_trees}")
    print(f"  - Trees RAM: {ram_dtrec_trees}")
    print(f"  - Dimension DTCM: {size_dtcm_simt} Bytes ({size_dtcm_simt/1024} KB) + {size_dtcm_dtrec} Bytes ({size_dtcm_dtrec/1024} KB)")
    print(f"  - Dimension RAM: {size_ram_dtrec} Bytes ({size_ram_dtrec/1024} KB)")
    print(f"  - Total Trees: {simt_trees + dtcm_dtrec_trees + ram_dtrec_trees}")
    print(f"  - Total Dimension: {size_bytes} Bytes ({size_bytes/1024} KB)")

    return num_internal_nodes, dtcm_dtrec_trees, ram_dtrec_trees

def simt_weight(path):
    model = joblib.load(path)

    size_bytes = 0
    size_dtcm_simt = 0
    counter = 0
    hybrid = 0

    for i,estimator in enumerate(model):
        tree = estimator.tree_
        nodes = tree.node_count

        size_bytes += nodes * 8
            
        if size_bytes > 129000 and hybrid == 0:
            hybrid = counter
            break
        if (i + 1) % 8 == 0:
            size_dtcm_simt = size_bytes


        counter += 1

    return hybrid, size_dtcm_simt


def generate_header(groups, trees, sample_size, parallelism, test_rows):
    return f"""\
#ifndef INC_FINAL_H_
#define INC_FINAL_H_    

#include "arm_mve.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdbool.h"

# define GROUPS         {int(groups)}
# define TREES          {trees}
# define DELTA          32767
# define PARALLELISM    {parallelism}
# define SAMPLE_SIZE    {sample_size}
# define TEST_ROWS      {test_rows}

#define DTCM __attribute__((section(".dtcm_data"), aligned(32)))
#define RAM_BIG __attribute__((section(".big_data"), aligned(32)))

typedef struct{{
    uint16_t class[PARALLELISM];
}}classes;

uint16_t final_results[GROUPS * PARALLELISM];
int16_t out[TREES];

typedef struct Node_Rec{{
	uint16_t feature;
	int16_t  threshold;

	struct Node_Rec *left;
	struct Node_Rec *right;

	int16_t left_class;
	int16_t right_class;
}}Node_Rec;

Node_Rec delta;

void init_struct_rec(Node_Rec *out_tree, int N, uint16_t *feature, int16_t *threshold, int16_t *child){{
	int *orig2int = (int *)malloc(N * sizeof(int));
	if (!orig2int) {{
		return;
	}}

	int internal_count = 0;
	for (int i = 0; i < N; ++i) {{
		if (threshold[i] != DELTA) {{
			orig2int[i] = internal_count;
			internal_count++;
		}} else {{
			orig2int[i] = -1;
		}}
	}}

	for (int i = 0; i < N; ++i) {{
		int idx = orig2int[i];
		if (idx < 0) {{
			continue; 
		}}

		out_tree[idx].feature     = feature[i];
		out_tree[idx].threshold   = threshold[i];

		out_tree[idx].left        = NULL;
		out_tree[idx].right       = NULL;
		out_tree[idx].left_class  = 0;
		out_tree[idx].right_class = 0;
	}}

	for (int i = 0; i < N; ++i) {{
		int idx = orig2int[i];
		if (idx < 0) {{
			continue; 
		}}

		int left_id  = child[2 * i];     
		int right_id = child[2 * i + 1]; 

		if (threshold[left_id] == DELTA) {{
			out_tree[idx].left = &delta;
			out_tree[idx].left_class = (int16_t)child[2 * left_id];
		}} else {{
			int child_idx = orig2int[left_id];
			out_tree[idx].left = &out_tree[child_idx];
		}}

		if (threshold[right_id] == DELTA) {{
			out_tree[idx].right = &delta;
			out_tree[idx].right_class = (int16_t)child[2 * right_id];
		}} else {{
			int child_idx = orig2int[right_id];
			out_tree[idx].right = &out_tree[child_idx];
		}}
	}}

	free(orig2int);
}}

"""

def generate_kernel_dtrec():
    return f"""\
int16_t dt_rec(Node_Rec *node, int16_t *sample){{
	if(sample[node->feature] <= node->threshold){{
		if(node->left != &delta){{
			return dt_rec(node->left, sample);
		}}else{{
			return node->left_class;
		}}
	}}else{{
		if(node->right != &delta){{
			return dt_rec(node->right, sample);
		}}else{{
			return node->right_class;
		}}
	}}
}}

"""

def generate_kernel_simt():
    return f"""\
bool all_active(mve_pred16_t mask){{
	return (mask == 0xFFFF);
}}

classes kernel_simt(int16_t* threshold, uint16_t* features, uint16_t* childL, uint16_t* childR, int16_t* sample){{
	uint16x8_t a = vld1q_u16(&childL[0]);
	uint16x8_t b = vld1q_u16(&childR[0]);

	int16x8_t thresh = vld1q_s16(&threshold[0]);
	uint16x8_t samp_offset = vld1q_u16(&features[0]);
	int16x8_t samp = vldrhq_gather_shifted_offset_s16(sample, samp_offset);

	mve_pred16_t mask = vcmpeqq_u16((uint16x8_t)vdupq_n_u16(0), (uint16x8_t)vdupq_n_u16(1));

	while(!all_active(mask)){{

		mve_pred16_t pred = vcmpleq_s16(samp, thresh);

		uint16x8_t dst = vpselq_u16(a, b, pred);

		thresh = vldrhq_gather_shifted_offset_s16(threshold, dst);

		mask = vcmpeqq_n_s16(thresh, DELTA);

		samp_offset = vldrhq_gather_shifted_offset_u16(features, dst);
		samp = vldrhq_gather_shifted_offset_s16(sample, samp_offset);
		a = vldrhq_gather_shifted_offset_u16(childL, dst);
		b = vldrhq_gather_shifted_offset_u16(childR, dst);
	}}

	classes result;
	vst1q_u16(result.class, b);

	return result;
}}

"""
        
def generate_inference_hybrid(fh, groups_hybrid, dtcm_simt_trees, dtcm_dtrec_trees, ram_dtrec_trees):
    
    total_dtrec_trees = dtcm_dtrec_trees + ram_dtrec_trees 
    
    fh.write("static inline void inference(int16_t* sample){\n")
    fh.write("   uint8_t i = 0;\n")
    fh.write("   classes myclass;\n")

    lines = []
    for i in range(groups_hybrid):
        line = f"   myclass = kernel_simt(threshold_dtcm{i}, feature_dtcm{i}, childL_dtcm{i}, childR_dtcm{i}, sample);\n"
        lines.append(line)
        line = "   memcpy(&final_results[i], myclass.class, sizeof(uint16_t) * PARALLELISM);\n"
        lines.append(line)
        line = "   i += PARALLELISM;\n\n"
        lines.append(line)
    fh.writelines(lines)

    if total_dtrec_trees != 0:
        index = dtcm_simt_trees
        for i in range(total_dtrec_trees):
            if i < dtcm_dtrec_trees:
                fh.write(f"   out[{i}] = dt_rec(tree_dtcm{index}, sample);\n")
                index += 1
            else:
                fh.write(f"   out[{i}] = dt_rec(tree{index}, sample);\n")
                index += 1    
    fh.write("}\n\n")

def generate_init(fh, dtrec_trees, dtcm_dtrec_trees, total_nodes, start):
    index = start

    if dtrec_trees != 0:
        fh.write("static inline void init_dtrec(){\n")
        for i in range(dtrec_trees):
            if i < dtcm_dtrec_trees and dtcm_dtrec_trees != 0:
                fh.write(f"   init_struct_rec(tree_dtcm{index}, {total_nodes[i]}, Rec_feature{index}, Rec_threshold{index}, Rec_child{index});\n")
                index += 1
            else:
                fh.write(f"   init_struct_rec(tree{index}, {total_nodes[i]}, Rec_feature{index}, Rec_threshold{index}, Rec_child{index});\n")
                index += 1
        fh.write("}\n\n") 

def generate_final(path, number_of_trees, max_depth, random_seed, parallelism, number_of_test_samples):
    test_path, joblib_path, sample_size, hybrid, csv_stem, task = trainRF.training(path, number_of_trees, max_depth, random_seed, number_of_test_samples, fast=0, hyb=1)
    #print(f"  - Total Trees in DTCM: {hybrid}")
    hybrid, size_dtcm_simt = simt_weight(joblib_path)
    #print(hybrid)
    if hybrid > 0:
        groups_hybrid = hybrid // parallelism
    else:
        groups_hybrid = number_of_trees/parallelism
    groups_hybrid = int(groups_hybrid)

    num_internal_nodes, dtcm_dtrec_trees, ram_dtrec_trees = internal_nodes_calc(joblib_path, groups_hybrid, size_dtcm_simt)

    models_path = Path(f"Models/{csv_stem}/final_RF_T{number_of_trees}_D{max_depth}")
    os.makedirs(models_path, exist_ok=True)

    header_name = f"{csv_stem}_RF_final_T{number_of_trees}_D{max_depth}_RS{random_seed}.h"
    txt_name = f"{csv_stem}_RF_final_T{number_of_trees}_D{max_depth}_RS{random_seed}.txt"

    header_path = models_path / header_name
    txt_path = models_path / txt_name

    dtrec_trees = dtcm_dtrec_trees + ram_dtrec_trees
    simt_dtcm_trees = groups_hybrid * 8
    header = generate_header(groups_hybrid, dtrec_trees, sample_size, parallelism, number_of_test_samples)
    simt = generate_kernel_simt()
    dtrec = generate_kernel_dtrec()

    total_nodes = 0

    with open(header_path, "w") as f:
        f.write(header)
        Parallel_generator(joblib_path, txt_path, parallelism, f, groups_hybrid, task)
        if dtrec_trees > 0:
            total_nodes = DT_Rec_generator(joblib_path, num_internal_nodes, txt_path, f, hybrid, simt_dtcm_trees, task)
        f.write(simt)
        f.write(dtrec)
        if total_nodes != 0:
            generate_init(f,dtrec_trees, dtcm_dtrec_trees, total_nodes, simt_dtcm_trees)
        generate_inference_hybrid(f, groups_hybrid, simt_dtcm_trees, dtcm_dtrec_trees, ram_dtrec_trees)
        testset_gen(test_path, f, number_of_test_samples)
        f.write("#endif\n")


