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
                if(groupindex >= groups_hybrid):
                    fh.write(f"RAM_BIG uint16_t feature{groupindex}[{len(feature_list)}] = {{{feature_str}}};\n")
                    fh.write(f"RAM_BIG int16_t threshold{groupindex}[{len(threshold_list)}] = {{{threshold_str}}};\n")
                    fh.write(f"RAM_BIG uint16_t childL{groupindex}[{len(childL_list)}] = {{{childL_str}}};\n")
                    fh.write(f"RAM_BIG uint16_t childR{groupindex}[{len(childR_list)}] = {{{childR_str}}};\n\n")

                    groupindex += 1
                else:
                    fh.write(f"DTCM uint16_t feature_dtcm{groupindex}[{len(feature_list)}] = {{{feature_str}}};\n")
                    fh.write(f"DTCM int16_t threshold_dtcm{groupindex}[{len(threshold_list)}] = {{{threshold_str}}};\n")
                    fh.write(f"DTCM uint16_t childL_dtcm{groupindex}[{len(childL_list)}] = {{{childL_str}}};\n")
                    fh.write(f"DTCM uint16_t childR_dtcm{groupindex}[{len(childR_list)}] = {{{childR_str}}};\n\n")

                    groupindex += 1



def generate_header(groups, sample_size, parallelism, test_rows):
    return f"""\
#ifndef INC_SIMT_H_
#define INC_SIMT_H_    

#include "arm_mve.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdbool.h"

# define GROUPS         {int(groups)}
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

"""

def generate_kernel():
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

def generate_inference_hybrid(fh, groups, groups_hybrid):
    fh.write("static inline void inference(int16_t* sample){\n")
    fh.write("   uint8_t i = 0;\n")
    fh.write("   classes myclass;\n")

    lines = []
    for i in range(groups):
        if i < groups_hybrid:
            line = f"   myclass = kernel_simt(threshold_dtcm{i}, feature_dtcm{i}, childL_dtcm{i}, childR_dtcm{i}, sample);\n"
            lines.append(line)
        else:
            line = f"   myclass = kernel_simt(threshold{i}, feature{i}, childL{i}, childR{i}, sample);\n"
            lines.append(line)
        line = "   memcpy(&final_results[i], myclass.class, sizeof(uint16_t) * PARALLELISM);\n"
        lines.append(line)
        line = "   i += PARALLELISM;\n\n"
        lines.append(line)
    fh.writelines(lines)
    fh.write("}\n\n")

def generate_inference(fh, groups): 
    fh.write("static inline void inference_ram(int16_t* sample){\n")
    fh.write("   uint8_t i = 0;\n")
    fh.write("   classes myclass;\n")
    
    lines = []
    for i in range(groups):
        line = f"   myclass = kernel_simt(threshold{i}, feature{i}, childL{i}, childR{i}, sample);\n"
        lines.append(line)
        line = "   memcpy(&final_results[i], myclass.class, sizeof(uint16_t) * PARALLELISM);\n"
        lines.append(line)
        line = "   i += PARALLELISM;\n\n"
        lines.append(line)
    
    fh.writelines(lines)
    fh.write("}\n\n")

    fh.write("static inline void inference_dtcm(int16_t* sample){\n")
    fh.write("   uint8_t i = 0;\n")
    fh.write("   classes myclass;\n")
    
    lines = []
    for i in range(groups):
        line = f"   myclass = kernel_simt(threshold_dtcm{i}, feature_dtcm{i}, childL_dtcm{i}, childR_dtcm{i}, sample);\n"
        lines.append(line)
        line = f"   memcpy(&final_results[i], myclass.class, sizeof(uint16_t) * PARALLELISM);\n"
        lines.append(line)
        line = f"   i += PARALLELISM;\n\n"
        lines.append(line)
    
    fh.writelines(lines)
    fh.write("}\n\n")

def testset_gen(csv_path, fh, n_rows):
    fh.write("int16_t testset[TEST_ROWS][SAMPLE_SIZE] = {\n")

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

def generate_simt(path, number_of_trees, max_depth, random_seed, parallelism, number_of_test_samples):
    test_path, joblib_path, sample_size, hybrid, csv_stem, task = trainRF.training(path, number_of_trees, max_depth, random_seed, number_of_test_samples)
    
    groups = number_of_trees/parallelism
    groups = int(groups)
    print(f"  - Groups: {groups}")

    models_path = Path(f"Models/{csv_stem}/simt_RF_G{groups}_D{max_depth}")
    os.makedirs(models_path, exist_ok=True)

    if(hybrid > 0):
        print("The model doesn't fit in DTCM. Proceeding with the hybrid approach...")
        groups_hybrid = hybrid // parallelism
        print(f"Only {hybrid} trees / {groups_hybrid} groups will fit in DTCM!")
        header_name = f"{csv_stem}_RF_simt_hybrid_G{groups}_D{max_depth}_RS{random_seed}.h"
        txt_name = f"{csv_stem}_RF_simt_hybrid_G{groups}_D{max_depth}_RS{random_seed}.txt"
        
        header_path = models_path / header_name
        txt_path = models_path / txt_name

        header = generate_header(groups, sample_size, parallelism, number_of_test_samples)
        kernel = generate_kernel()

        with open(header_path, "w") as f:
            f.write(header)
            Parallel_generator(joblib_path, txt_path, parallelism, f, groups_hybrid, task)
            f.write(kernel)
            generate_inference_hybrid(f, groups, groups_hybrid)
            testset_gen(test_path, f, number_of_test_samples)
            f.write("#endif\n")
    else:
        print("The model fits entirely in DTCM!")
        header_name = f"{csv_stem}_RF_simt_G{groups}_D{max_depth}_RS{random_seed}.h"
        txt_name = f"{csv_stem}_RF_simt_G{groups}_D{max_depth}_RS{random_seed}.txt"

        header_path = models_path / header_name
        txt_path = models_path / txt_name

        header = generate_header(groups, sample_size, parallelism, number_of_test_samples)
        kernel = generate_kernel()

        with open(header_path, "w") as f:
            f.write(header)
            Parallel_generator(joblib_path, txt_path, parallelism, f, 0, task)
            f.write(kernel)
            generate_inference(f, groups)
            testset_gen(test_path, f, number_of_test_samples)
            f.write("#endif\n")
 
