# 
# Copyright (c) 2026 Lorenzo Abate <lorenzoabate510@gmail.com>.
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

from joblib import dump, load
from xgboost import XGBClassifier
import re
import numpy as np
from collections import deque
from typing import List, Tuple, Dict
import os
import csv
from pathlib import Path
import shutil

import Code_Gen.Utils.trainXGB as trainXGB

DELTA = 32767

def round_and_class_text(i: int, num_class: int) -> Tuple[int, int]:
    return i // num_class, i % num_class

def group_dump_by_class_text(dump: List[str], num_class: int) -> Dict[int, List[Tuple[int, str]]]:
    grouped = {c: [] for c in range(num_class)}
    for i, tree_text in enumerate(dump):
        rnd, cls = i // num_class, i % num_class
        grouped[cls].append((rnd, tree_text))
    return grouped

def reorder_dump_by_class_text(dump: List[str], num_class: int) -> Tuple[List[str], List[int], List[int]]:
    grouped = group_dump_by_class_text(dump, num_class)
    new_dump = []
    new2old = []
    for cls in range(num_class):
        for rnd, tree_text in grouped[cls]:
            old_idx = rnd * num_class + cls
            new2old.append(old_idx)
            new_dump.append(tree_text)
    old2new = [None] * len(dump)
    for new_idx, old_idx in enumerate(new2old):
        old2new[old_idx] = new_idx
    return new_dump, old2new, new2old

def set_model(num_class, acc, path, task, mae, mse, r2):
    model = load(path)
    max_depth = model.get_params()["max_depth"]
    booster = model.get_booster()
    base_score = model.get_xgb_params()["base_score"]
    dump = booster.get_dump()

    re_split = re.compile(
        r'^(?P<nodeid>\d+):\[(?P<feature>[^<]+)<(?P<thresh>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\]\s+yes=(?P<yes>\d+),no=(?P<no>\d+),missing=(?P<missing>\d+)'
    )
    re_leaf = re.compile(
        r'^(?P<nodeid>\d+):leaf=(?P<leaf>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)'
    )
    
    if num_class > 2:
        print("Reordering Trees...")
        dump, old2new, new2old = reorder_dump_by_class_text(dump, num_class)
        #print(old2new)
        #print(new2old)

        total_trees = len(dump)
        trees_per_class = total_trees/num_class
        trees = 0
        
        maxv = float('-inf')
        minv = float('inf')
        max_vector = []
        min_vector = []
        range_vector = []
        n_nodes = []
        n_internal_nodes = []
        total_nodes = 0
        total_internal_nodes = 0
        size_bytes = 0
        size_kb = 0
        
        hybrid = 0
        counter = 0

        for txt in dump:
            n = 0
            n_i = 0
            for line in txt.splitlines():
                if not line.strip():
                    continue
                depth = 0
                while line.startswith('\t'):
                    depth += 1
                    line = line[1:]

                m_split = re_split.match(line)
                if m_split:
                    n += 1
                    n_i += 1
                    continue
                m_leaf = re_leaf.match(line)
                if m_leaf:
                    n+=1
                    d = m_leaf.groupdict()
                    #print(f"{int(d['nodeid'])}: {float(d['leaf'])}, {depth}")
                    if (float(d['leaf']) > maxv):
                        maxv = float(d['leaf'])
                    if (float(d['leaf']) < minv):
                        minv = float(d['leaf'])
                    continue
            trees += 1
            if(trees % trees_per_class == 0) and (trees != 0):
                max_vector.append(maxv)
                min_vector.append(minv)
                range_vector.append(maxv-minv)
                maxv = float('-inf')
                minv = float('inf')            
            n_nodes.append(int(n))
            n_internal_nodes.append(int(n_i))

        size_dtcm_simt = 0
        size_dtcm_dtrec = 0
        size_ram_dtrec = 0
        ram_dtrec_trees = 0
        dtcm_dtrec_trees = 0
        
        for i in range(len(n_nodes)):
            total_nodes += n_nodes[i]

        for i in range(len(n_nodes)):
            size_bytes += n_nodes[i] * 8

            if size_bytes > 129000 and hybrid == 0:
                hybrid = counter
                break

            if (i + 1) % 8 == 0:
                size_dtcm_simt = size_bytes

        if hybrid != 0:
            groups_hybrid = hybrid // 8
            simt_trees = groups_hybrid * 8

            size_bytes = size_dtcm_simt
            for i in range(len(n_nodes)):
                if i >= simt_trees:
                    size_bytes += n_internal_nodes[i] * 16

                    if size_bytes <= 129000:
                        size_dtcm_dtrec += n_internal_nodes[i] * 16
                        dtcm_dtrec_trees += 1
                    else:
                        size_ram_dtrec += n_internal_nodes[i] * 16
                        ram_dtrec_trees += 1
        else:
            simt_trees = len(n_nodes)
            groups_hybrid = simt_trees // 8

        print(f"Model Stats:")
        print(f"  - Total Nodes: {total_nodes}")
        print(f"  - Accuracy: {acc:.4f}")
        print(f"  - Classes: {num_class}")
        print(f"  - Max: {max_vector}")
        print(f"  - Min: {min_vector}")
        print(f"  - Trees DTCM: {simt_trees} ({groups_hybrid} Groups) + {dtcm_dtrec_trees}")
        print(f"  - Trees RAM: {ram_dtrec_trees}")
        print(f"  - Dimension DTCM: {size_dtcm_simt} Bytes ({size_dtcm_simt/1024} KB) + {size_dtcm_dtrec} Bytes ({size_dtcm_dtrec/1024} KB)")
        print(f"  - Dimension RAM: {size_ram_dtrec} Bytes ({size_ram_dtrec/1024} KB)")
        print(f"  - Total Trees: {simt_trees + dtcm_dtrec_trees + ram_dtrec_trees}")
        print(f"  - Total Dimension: {size_bytes} Bytes ({size_bytes/1024} KB)")
        
        thresholds = []
        features = []
        childL = []
        childR = []

        class_ = 0
        for i,txt in enumerate(dump):
            t = [0 for _ in range(int(n_nodes[i]))]
            f = [0 for _ in range(int(n_nodes[i]))]
            cl = [0 for _ in range(int(n_nodes[i]))]
            cr = [0 for _ in range(int(n_nodes[i]))]

            for line in txt.splitlines():
                if not line.strip():
                    continue
                depth = 0
                while line.startswith('\t'):
                    depth += 1
                    line = line[1:]

                m_split = re_split.match(line)
                if m_split:
                    d = m_split.groupdict()
                    node = int(d['nodeid'])
                    t[node] = int(d['thresh'])
                    f[node] = int(d['feature'])
                    cl[node] = int(d['yes'])
                    cr[node] = int(d['no'])
                    continue
                m_leaf = re_leaf.match(line)
                if m_leaf:
                    d = m_leaf.groupdict()
                    node = int(d['nodeid'])
                    t[node] = DELTA
                    f[node] = 0
                    
                    leave = float(d['leaf'])
                    q = np.round((leave - min_vector[class_]) / range_vector[class_] * 65535).astype(np.uint16)
                    cl[node] = q
                    cr[node] = q
                    continue

            thresholds.append(t)
            features.append(f)
            childL.append(cl)
            childR.append(cr)
            if((i+1) % trees_per_class == 0) and (i != 0):
                class_ += 1
                
        return thresholds, features, childL, childR, hybrid, len(dump), max_vector, min_vector, max_depth, n_internal_nodes,  groups_hybrid, dtcm_dtrec_trees, ram_dtrec_trees
    
    else:
        range_ = 0
        maxv = float('-inf')
        minv = float('inf')
        n_nodes = []
        n_internal_nodes = []
        total_nodes = 0
        total_internal_nodes = 0
        size_bytes = 0
        size_kb = 0
        
        hybrid = 0
        counter = 0

        for txt in dump:
            n = 0
            n_i = 0
            for line in txt.splitlines():
                if not line.strip():
                    continue
                depth = 0
                while line.startswith('\t'):
                    depth += 1
                    line = line[1:]

                m_split = re_split.match(line)
                if m_split:
                    n += 1
                    n_i += 1
                    continue
                m_leaf = re_leaf.match(line)
                if m_leaf:
                    n+=1
                    d = m_leaf.groupdict()
                    #print(f"{int(d['nodeid'])}: {float(d['leaf'])}, {depth}")
                    if (float(d['leaf']) > maxv):
                        maxv = float(d['leaf'])
                    if (float(d['leaf']) < minv):
                        minv = float(d['leaf'])
                    continue
            n_nodes.append(int(n))
            n_internal_nodes.append(int(n_i))

        size_dtcm_simt = 0
        size_dtcm_dtrec = 0
        size_ram_dtrec = 0
        ram_dtrec_trees = 0
        dtcm_dtrec_trees = 0
        
        range_ = maxv - minv

        for i in range(len(n_nodes)):
            total_nodes += n_nodes[i]

        for i in range(len(n_nodes)):
            size_bytes += n_nodes[i] * 8

            if size_bytes > 129000 and hybrid == 0:
                hybrid = counter
                break

            if (i + 1) % 8 == 0:
                size_dtcm_simt = size_bytes

        if hybrid != 0:
            groups_hybrid = hybrid // 8
            simt_trees = groups_hybrid * 8

            size_bytes = size_dtcm_simt
            for i in range(len(n_nodes)):
                if i >= simt_trees:
                    size_bytes += n_internal_nodes[i] * 16

                    if size_bytes <= 129000:
                        size_dtcm_dtrec += n_internal_nodes[i] * 16
                        dtcm_dtrec_trees += 1
                    else:
                        size_ram_dtrec += n_internal_nodes[i] * 16
                        ram_dtrec_trees += 1
        else:
            simt_trees = len(n_nodes)
            groups_hybrid = simt_trees // 8

        if task == 0:
            print(f"Model Stats:")
            print(f"  - Total Nodes: {total_nodes}")
            print(f"  - Accuracy: {acc:.4f}")
            print(f"  - Classes: {num_class}")
            print(f"  - Max: {maxv}")
            print(f"  - Min: {minv}")
            print(f"  - Trees DTCM: {simt_trees} ({groups_hybrid} Groups) + {dtcm_dtrec_trees}")
            print(f"  - Trees RAM: {ram_dtrec_trees}")
            print(f"  - Dimension DTCM: {size_dtcm_simt} Bytes ({size_dtcm_simt/1024} KB) + {size_dtcm_dtrec} Bytes ({size_dtcm_dtrec/1024} KB)")
            print(f"  - Dimension RAM: {size_ram_dtrec} Bytes ({size_ram_dtrec/1024} KB)")
            print(f"  - Total Trees: {simt_trees + dtcm_dtrec_trees + ram_dtrec_trees}")
            print(f"  - Total Dimension: {size_bytes} Bytes ({size_bytes/1024} KB)")
        else:
            print(f"Model Stats:")
            print(f"  - Total Nodes: {total_nodes}")
            print(f"  - MSE: {mse:.4f}")
            print(f"  - MAE: {mae:.4f}")
            print(f"  - R2:  {r2:.4f}")
            print(f"  - Classes: Regression")
            print(f"  - Max: {maxv}")
            print(f"  - Min: {minv}")
            print(f"  - Trees DTCM: {simt_trees} ({groups_hybrid} Groups) + {dtcm_dtrec_trees}")
            print(f"  - Trees RAM: {ram_dtrec_trees}")
            print(f"  - Dimension DTCM: {size_dtcm_simt} Bytes ({size_dtcm_simt/1024} KB) + {size_dtcm_dtrec} Bytes ({size_dtcm_dtrec/1024} KB)")
            print(f"  - Dimension RAM: {size_ram_dtrec} Bytes ({size_ram_dtrec/1024} KB)")
            print(f"  - Total Trees: {simt_trees + dtcm_dtrec_trees + ram_dtrec_trees}")
            print(f"  - Total Dimension: {size_bytes} Bytes ({size_bytes/1024} KB)")

        thresholds = []
        features = []
        childL = []
        childR = []

        for i,txt in enumerate(dump):
            t = [0 for _ in range(int(n_nodes[i]))]
            f = [0 for _ in range(int(n_nodes[i]))]
            cl = [0 for _ in range(int(n_nodes[i]))]
            cr = [0 for _ in range(int(n_nodes[i]))]

            for line in txt.splitlines():
                if not line.strip():
                    continue
                depth = 0
                while line.startswith('\t'):
                    depth += 1
                    line = line[1:]

                m_split = re_split.match(line)
                if m_split:
                    d = m_split.groupdict()
                    node = int(d['nodeid'])
                    t[node] = int(d['thresh'])
                    f[node] = int(d['feature'])
                    cl[node] = int(d['yes'])
                    cr[node] = int(d['no'])
                    continue
                m_leaf = re_leaf.match(line)
                if m_leaf:
                    d = m_leaf.groupdict()
                    node = int(d['nodeid'])
                    t[node] = DELTA
                    f[node] = 0
                    
                    leave = float(d['leaf'])
                    if task == 0:
                        q = np.round((leave - minv) / range_ * 65535).astype(np.uint16)
                    else:
                        q = 1
                    cl[node] = q
                    cr[node] = q
                    continue

            thresholds.append(t)
            features.append(f)
            childL.append(cl)
            childR.append(cr)
        return thresholds, features, childL, childR, hybrid, len(dump), maxv, minv, max_depth, n_internal_nodes, groups_hybrid, dtcm_dtrec_trees, ram_dtrec_trees

def generate_header_multi(groups, trees, n_classes, sample_size, parallelism, test_rows, maxv, minv):
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
# define CLASSES        {n_classes}
# define DELTA          32767
# define PARALLELISM    {parallelism}
# define SAMPLE_SIZE    {sample_size}
# define TEST_ROWS      {test_rows}

const float MAX[{len(maxv)}] = {{{", ".join(f"{m}f" for m in maxv)}}};
const float MIN[{len(minv)}] = {{{", ".join(f"{m}f" for m in minv)}}};

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

float dequantize_uint16(uint16_t q, float min, float max) {{
    float range = max - min;
    return ((float)q / 65535.0f) * range + min;
}}

void softmax(const float *logits, float *probs, uint16_t num_classes) {{
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

def generate_header_binary(groups, trees, sample_size, parallelism, test_rows, maxv, minv):
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

const float MAX = {maxv}f;
const float MIN = {minv}f;

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

float dequantize_uint16(uint16_t q, float min, float max) {{
    float range = max - min;
    return ((float)q / 65535.0f) * range + min;
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

def get_nodes_by_level(tree_index, childL, childR, threshold):
    levels = []
    queue = deque([(0, 0)])  # (node_id, level)

    while queue:
        node_id, level = queue.popleft()
        if level == len(levels):
            levels.append([])
        levels[level].append(node_id)
        if threshold[tree_index][node_id] != DELTA:
            queue.append((childL[tree_index][node_id], level + 1))
            queue.append((childR[tree_index][node_id], level + 1))
    return levels

def SIMT_XGBoost_gen_binary(thresholds, features, childL, childR, num_group, txt_name, groups_hybrid, fh):
    forest_levels = []

    for i in range(len(thresholds)):
        levels = get_nodes_by_level(i, childL, childR, thresholds)
        forest_levels.append((i, levels))

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
                for idx, (tree, levels) in enumerate(group):
                    if level_idx < len(levels):
                        node_ids = levels[level_idx]
                        #print(f"  Albero {i + tree_idx}:")
                        for node_id in node_ids:
                            feature = features[tree][node_id]
                            threshold = thresholds[tree][node_id]
                            left = childL[tree][node_id]
                            right = childR[tree][node_id]

                            if(threshold == DELTA):
                                feature_list.append(np.uint16(0))
                                threshold_list.append(np.int16(DELTA))
                                childL_list.append(np.uint16(num_nodes))
                                childR_list.append(np.uint16(right))
                                num_nodes += 1
                            else:
                                feature_list.append(np.uint16(feature))
                                threshold_list.append(np.int16(threshold))
                                childL_list.append(np.uint16((2*num_added_nodes) + num_group))
                                childR_list.append(np.uint16((2*num_added_nodes) + num_group + 1))
                                num_nodes += 1
                                num_added_nodes += 1
            
            f.write(f"Group {i}-{i+num_group-1}  Number of Nodes = {num_nodes}\n")
            feature_str = ", ".join(str(f) for f in feature_list)
            f.write(f"feature = {feature_str}\n")
            threshold_str = ", ".join(str(t) for t in threshold_list)
            f.write(f"threshold = {threshold_str}\n")
            childL_str = ", ".join(str(c) for c in childL_list)
            f.write(f"childL = {childL_str}\n")
            childR_str = ", ".join(str(c) for c in childR_list)
            f.write(f"childR = {childR_str}\n")

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

def DT_Rec_XGBoost(thresholds, features, childL, childR, num_internal_nodes, txt_name, fh, hybrid, simt_dtcm_trees, task = 0):
    DELTA = 32767

    num_nodes_list = [] 

    with open(txt_name, "w") as f:
        for tree in range(len(num_internal_nodes)):
            if tree >= simt_dtcm_trees:
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
                    feature = features[tree][current_node]
                    threshold = thresholds[tree][current_node]
                    left = childL[tree][current_node]
                    right = childR[tree][current_node]

                    if(threshold == DELTA):
                        feature_list.append(np.uint16(0))
                        threshold_list.append(np.int16(DELTA))

                        child_list.append(np.int16(left))  # child[2n] = class
                        child_list.append(np.int16(right))  # child[2n+1] = placeholder

                        num_nodes += 1
                    else:
                        feature_list.append(np.uint16(feature))
                        threshold_list.append(np.int16(threshold))
                        child_list.append(np.int16((2*num_added_nodes) + 1))
                        child_list.append(np.int16((2*num_added_nodes) + 2))

                        nodes.append(left)
                        nodes.append(right)

                        num_added_nodes += 1
                        num_nodes += 1

                f.write(f"Tree {tree}   Number of Nodes = {num_nodes}\n")
                feature_str = ", ".join(str(f) for f in feature_list)
                f.write(f"feature = {feature_str}\n")
                threshold_str = ", ".join(str(t) for t in threshold_list)
                f.write(f"threshold = {threshold_str}\n")
                child_str = ", ".join(str(c) for c in child_list)
                f.write(f"child = {child_str}\n")

                fh.write(f"uint16_t Rec_feature{tree}[{len(feature_list)}] = {{{feature_str}}};\n")
                fh.write(f"int16_t Rec_threshold{tree}[{len(threshold_list)}] = {{{threshold_str}}};\n")
                fh.write(f"int16_t Rec_child{tree}[{len(child_list)}] = {{{child_str}}};\n")

                if hybrid == 0:
                    fh.write(f"RAM_BIG Node_Rec tree{tree}[{num_internal_nodes[tree]}];\n")
                    fh.write(f"DTCM Node_Rec tree_dtcm{tree}[{num_internal_nodes[tree]}];\n\n")
                    num_nodes_list.append(len(feature_list))
                else:
                    if tree >= hybrid:
                        fh.write(f"RAM_BIG Node_Rec tree{tree}[{num_internal_nodes[tree]}];\n\n")
                        num_nodes_list.append(len(feature_list))
                    else:
                        fh.write(f"DTCM Node_Rec tree_dtcm{tree}[{num_internal_nodes[tree]}];\n\n")
                        num_nodes_list.append(len(feature_list))         
    return num_nodes_list

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

def generate_final(path, n_trees, random_seed, parallelism, number_of_test_samples):

    test_path, joblib_path, sample_size, acc, csv_stem, n_classes, task , mae, mse, r2= trainXGB.training(path, n_trees, random_seed, number_of_test_samples)
    thresholds, features, childL, childR, hybrid, number_of_trees, maxv, minv, max_depth, n_internal_nodes, groups_hybrid, dtcm_dtrec_trees, ram_dtrec_trees = set_model(n_classes, acc, joblib_path, task, mae, mse, r2)

    models_path = Path(f"Models/{csv_stem}/hybrid_XGB_T{number_of_trees}_D{max_depth}")
    os.makedirs(models_path, exist_ok=True)

    header_name = f"{csv_stem}_XGB_hybrid_T{number_of_trees}_D{max_depth}_RS{random_seed}.h"
    txt_name = f"{csv_stem}_XGB_hybrid_T{number_of_trees}_D{max_depth}_RS{random_seed}.txt"

    header_path = models_path / header_name
    txt_path = models_path / txt_name

    dtrec_trees = dtcm_dtrec_trees + ram_dtrec_trees
    simt_dtcm_trees = groups_hybrid * 8

    if n_classes > 2:
        header = generate_header_multi(groups_hybrid, dtrec_trees, n_classes, sample_size, parallelism, number_of_test_samples, maxv, minv)
    else:
        header = generate_header_binary(groups_hybrid, dtrec_trees, sample_size, parallelism, number_of_test_samples, maxv, minv)
    simt = generate_kernel_simt()
    dtrec = generate_kernel_dtrec()

    total_nodes = 0

    with open(header_path, "w") as f:
        f.write(header)
        SIMT_XGBoost_gen_binary(thresholds, features, childL, childR, parallelism, txt_path, groups_hybrid, f)
        if dtrec_trees > 0:
            total_nodes = DT_Rec_XGBoost(thresholds, features, childL, childR, n_internal_nodes, hybrid, simt_dtcm_trees, task)
        f.write(simt)
        f.write(dtrec)
        if total_nodes != 0:
            generate_init(f, dtrec_trees, dtcm_dtrec_trees, total_nodes, simt_dtcm_trees)
        generate_inference_hybrid(f, groups_hybrid, simt_dtcm_trees, dtcm_dtrec_trees, ram_dtrec_trees)
        testset_gen(test_path, f, number_of_test_samples)
        f.write("#endif\n")
