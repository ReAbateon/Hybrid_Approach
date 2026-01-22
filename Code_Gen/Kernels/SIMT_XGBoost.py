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

def generate_header(groups_per_class, n_classes, sample_size, parallelism, test_rows, maxv, minv):
    return f"""\
#ifndef INC_SIMT_H_
#define INC_SIMT_H_ 

#include "arm_mve.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdbool.h"
#include "math.h"
#include "stdint.h"

# define GROUPS         {int(groups_per_class)}
# define CLASSES        {int(n_classes)}
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

uint16_t final_results[CLASSES][GROUPS * PARALLELISM];

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

"""

def generate_header_binary(groups, sample_size, parallelism, test_rows, maxv, minv):
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

const float MAX = {maxv}f;
const float MIN = {minv}f;

#define DTCM __attribute__((section(".dtcm_data"), aligned(32)))
#define RAM_BIG __attribute__((section(".big_data"), aligned(32)))

typedef struct{{
    uint16_t class[PARALLELISM];
}}classes;

uint16_t final_results[GROUPS * PARALLELISM];

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
        total_nodes = 0
        size_bytes = 0
        size_kb = 0
        
        hybrid = 0
        counter = 0

        for txt in dump:
            n = 0
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

        for i in range(len(n_nodes)):
            total_nodes += n_nodes[i]
            size_bytes += n_nodes[i] * 8

            if size_bytes > 129000 and hybrid == 0:
                hybrid = counter

            counter += 1
        
        size_kb += size_bytes / 1024
        print(f"Model Stats:")
        print(f"  - Total Trees: {len(dump)}")
        print(f"  - Total Nodes: {total_nodes}")
        print(f"  - Dimension: {size_bytes} Bytes ({size_kb:.2f} KB)")
        print(f"  - Classes: {num_class}")
        print(f"  - Accuracy: {acc:.4f}")
        print(f"  - Max: {max_vector}")
        print(f"  - Min: {min_vector}")

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
                
        return thresholds, features, childL, childR, hybrid, len(dump), max_vector, min_vector, max_depth
    
    else:
        range_ = 0
        maxv = float('-inf')
        minv = float('inf')
        n_nodes = []
        total_nodes = 0
        size_bytes = 0
        size_kb = 0
        
        hybrid = 0
        counter = 0

        for txt in dump:
            n = 0
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

        for i in range(len(n_nodes)):
            total_nodes += n_nodes[i]
            size_bytes += n_nodes[i] * 8

            if size_bytes > 129000 and hybrid == 0:
                hybrid = counter

            counter += 1
        
        size_kb += size_bytes / 1024
        range_ = maxv - minv

        if task == 0:
            print(f"Model Stats:")
            print(f"  - Total Trees: {len(dump)}")
            print(f"  - Total Nodes: {total_nodes}")
            print(f"  - Dimension: {size_bytes} Bytes ({size_kb:.2f} KB)")
            print(f"  - Classes: {num_class}")
            print(f"  - Accuracy: {acc:.4f}")
            print(f"  - Max: {maxv}")
            print(f"  - Min: {minv}")
        else:
            print(f"Model Stats:")
            print(f"  - Total Trees: {len(dump)}")
            print(f"  - Total Nodes: {total_nodes}")
            print(f"  - Dimension: {size_bytes} Bytes ({size_kb:.2f} KB)")
            print(f"  - Classes: Regression")
            print(f"  - MSE: {mse:.4f}")
            print(f"  - MAE: {mae:.4f}")
            print(f"  - R2:  {r2:.4f}")
            print(f"  - Max: {maxv}")
            print(f"  - Min: {minv}")

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
                    q = np.round((leave - minv) / range_ * 65535).astype(np.uint16)
                    cl[node] = q
                    cr[node] = q
                    continue

            thresholds.append(t)
            features.append(f)
            childL.append(cl)
            childR.append(cr)
        return thresholds, features, childL, childR, hybrid, len(dump), maxv, minv, max_depth


def SIMT_XGBoost_gen(thresholds, features, childL, childR, num_group, txt_name, groups_hybrid, groups_per_class, fh):
    forest_levels = []

    for i in range(len(thresholds)):
        levels = get_nodes_by_level(i, childL, childR, thresholds)
        forest_levels.append((i, levels))

    groupindex = 0
    classindex = 0
    totalgroupindex = 0
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
            
            f.write(f"Group {i}-{i+num_group-1}   Number of Nodes = {num_nodes}\n")
            feature_str = ", ".join(str(f) for f in feature_list)
            f.write(f"feature = {feature_str}\n")
            threshold_str = ", ".join(str(t) for t in threshold_list)
            f.write(f"threshold = {threshold_str}\n")
            childL_str = ", ".join(str(c) for c in childL_list)
            f.write(f"childL = {childL_str}\n")
            childR_str = ", ".join(str(c) for c in childR_list)
            f.write(f"childR = {childR_str}\n")

            if groups_hybrid == 0:     
                fh.write(f"RAM_BIG uint16_t feature_{classindex}_{groupindex}[{len(feature_list)}] = {{{feature_str}}};\n")
                fh.write(f"RAM_BIG int16_t threshold_{classindex}_{groupindex}[{len(threshold_list)}] = {{{threshold_str}}};\n")
                fh.write(f"RAM_BIG uint16_t childL_{classindex}_{groupindex}[{len(childL_list)}] = {{{childL_str}}};\n")
                fh.write(f"RAM_BIG uint16_t childR_{classindex}_{groupindex}[{len(childR_list)}] = {{{childR_str}}};\n")

                fh.write(f"DTCM uint16_t feature_dtcm_{classindex}_{groupindex}[{len(feature_list)}] = {{{feature_str}}};\n")
                fh.write(f"DTCM int16_t threshold_dtcm_{classindex}_{groupindex}[{len(threshold_list)}] = {{{threshold_str}}};\n")
                fh.write(f"DTCM uint16_t childL_dtcm_{classindex}_{groupindex}[{len(childL_list)}] = {{{childL_str}}};\n")
                fh.write(f"DTCM uint16_t childR_dtcm_{classindex}_{groupindex}[{len(childR_list)}] = {{{childR_str}}};\n\n")
                
                groupindex += 1
                if(groupindex == groups_per_class):
                    groupindex = 0
                    classindex += 1
            elif groups_hybrid > 0:
                if(totalgroupindex >= groups_hybrid):
                    fh.write(f"RAM_BIG uint16_t feature_{classindex}_{groupindex}[{len(feature_list)}] = {{{feature_str}}};\n")
                    fh.write(f"RAM_BIG int16_t threshold_{classindex}_{groupindex}[{len(threshold_list)}] = {{{threshold_str}}};\n")
                    fh.write(f"RAM_BIG uint16_t childL_{classindex}_{groupindex}[{len(childL_list)}] = {{{childL_str}}};\n")
                    fh.write(f"RAM_BIG uint16_t childR_{classindex}_{groupindex}[{len(childR_list)}] = {{{childR_str}}};\n\n")

                    totalgroupindex += 1
                    groupindex += 1
                else:
                    fh.write(f"DTCM uint16_t feature_dtcm_{classindex}_{groupindex}[{len(feature_list)}] = {{{feature_str}}};\n")
                    fh.write(f"DTCM int16_t threshold_dtcm_{classindex}_{groupindex}[{len(threshold_list)}] = {{{threshold_str}}};\n")
                    fh.write(f"DTCM uint16_t childL_dtcm_{classindex}_{groupindex}[{len(childL_list)}] = {{{childL_str}}};\n")
                    fh.write(f"DTCM uint16_t childR_dtcm_{classindex}_{groupindex}[{len(childR_list)}] = {{{childR_str}}};\n\n")

                    totalgroupindex += 1
                    groupindex += 1
                if(groupindex == groups_per_class):
                    groupindex = 0
                    classindex += 1

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
            
            f.write(f"Group {i}-{i+num_group-1}   Number of Nodes = {num_nodes}\n")
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

def generate_inference(fh, n_classes, groups_per_class):
    fh.write("static inline void inference_ram(int16_t* sample){\n")
    fh.write("   uint8_t i = 0;\n")
    fh.write("   classes myclass;\n")

    lines = []
    for j in range(n_classes):
        for i in range(groups_per_class):
            line = f"   myclass = kernel_simt(threshold_{j}_{i}, feature_{j}_{i}, childL_{j}_{i}, childR_{j}_{i}, sample);\n"
            lines.append(line)
            line = f"   memcpy(&final_results[{j}][i], myclass.class, sizeof(uint16_t) * PARALLELISM);\n"
            lines.append(line)
            if i == groups_per_class - 1:
                line = "   i = 0;\n\n"
                lines.append(line)
            else:
                line = "   i += PARALLELISM;\n\n"
                lines.append(line)
                
    fh.writelines(lines)
    fh.write("}\n\n")

    fh.write("static inline void inference_dtcm(int16_t* sample){\n")
    fh.write("   uint8_t i = 0;\n")
    fh.write("   classes myclass;\n")
    
    lines = []
    for j in range(n_classes):
        for i in range(groups_per_class):
            line = f"   myclass = kernel_simt(threshold_dtcm_{j}_{i}, feature_dtcm_{j}_{i}, childL_dtcm_{j}_{i}, childR_dtcm_{j}_{i}, sample);\n"
            lines.append(line)
            line = f"   memcpy(&final_results[{j}][i], myclass.class, sizeof(uint16_t) * PARALLELISM);\n"
            lines.append(line)
            if i == groups_per_class - 1:
                line = "   i = 0;\n\n"
                lines.append(line)
            else:
                line = "   i += PARALLELISM;\n\n"
                lines.append(line)
    
    fh.writelines(lines)
    fh.write("}\n\n")

def generate_inference_hybrid(fh, n_classes, groups_per_class, groups_hybrid):
    fh.write("static inline void inference(int16_t* sample){\n")
    fh.write("   uint8_t i = 0;\n")
    fh.write("   classes myclass;\n")

    total_groups = 0
    lines = []
    for j in range(n_classes):
        for i in range(groups_per_class):
            if total_groups < groups_hybrid:
                line = f"   myclass = kernel_simt(threshold_dtcm_{j}_{i}, feature_dtcm_{j}_{i}, childL_dtcm_{j}_{i}, childR_dtcm_{j}_{i}, sample);\n"
                lines.append(line)
                total_groups += 1
            else:
                line = f"   myclass = kernel_simt(threshold_{j}_{i}, feature_{j}_{i}, childL_{j}_{i}, childR_{j}_{i}, sample);\n"
                lines.append(line)
                total_groups += 1
            line = "   memcpy(&final_results[i], myclass.class, sizeof(uint16_t) * PARALLELISM);\n"
            lines.append(line)
            line = "   i += PARALLELISM;\n\n"
            lines.append(line)
        fh.writelines(lines)
        fh.write("}\n\n")

def generate_inference_binary(fh, groups):
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



def generate_inference_binary_hybrid(fh, groups, groups_hybrid):
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


def generate_simt(path, n_trees, random_seed, parallelism, number_of_test_samples):
    test_path, joblib_path, sample_size, acc, csv_stem, n_classes, task , mae, mse, r2 = trainXGB.training(path, n_trees, random_seed, number_of_test_samples)
    thresholds, features, childL, childR, hybrid, number_of_trees, maxv, minv, max_depth = set_model(n_classes, acc, joblib_path, task, mae, mse, r2)
    
    models_path = Path(f"Models/{csv_stem}/simt_XGB_T{n_trees}_D{max_depth}")
    os.makedirs(models_path, exist_ok=True)

    if n_classes > 2:
        trees_per_class = number_of_trees/n_classes
        groups_per_class = trees_per_class/parallelism
        groups_per_class = int(groups_per_class)
        print(f"  - Groups_per_Class: {groups_per_class}")
        if(hybrid>0):
            print("The model doesn't fit in DTCM. Proceeding with the hybrid approach...")
            groups_hybrid = hybrid // parallelism
            print(f"Only {hybrid} trees / {groups_hybrid} groups will fit in DTCM!")
            
            header_name = f"{csv_stem}_XGB_simt_hybrid_GC{groups_per_class}_D{max_depth}_RS{random_seed}.h"
            txt_name = f"{csv_stem}_XGB_simt_hybrid_GC{groups_per_class}_D{max_depth}_RS{random_seed}.txt"

            header_path = models_path / header_name
            txt_path = models_path / txt_name

            header = generate_header(groups_per_class, n_classes, sample_size, parallelism, number_of_test_samples, maxv, minv)
            kernel = generate_kernel()

            with open(header_path, "w") as f:
                f.write(header)
                SIMT_XGBoost_gen(thresholds, features, childL, childR, parallelism, txt_path, groups_hybrid, groups_per_class, f)
                f.write(kernel)
                generate_inference_hybrid(f, n_classes, groups_per_class, groups_hybrid)
                testset_gen(test_path, f, number_of_test_samples)
                f.write("#endif\n")
        else:
            print("The model fits entirely in DTCM!")
            
            header_name = f"{csv_stem}_XGB_simt_GC{groups_per_class}_D{max_depth}_RS{random_seed}.h"
            txt_name = f"{csv_stem}_XGB_simt_GC{groups_per_class}_D{max_depth}_RS{random_seed}.txt"

            header_path = models_path / header_name
            txt_path = models_path / txt_name

            header = generate_header(groups_per_class, n_classes, sample_size, parallelism, number_of_test_samples, maxv, minv)
            kernel = generate_kernel()

            with open(header_path, "w") as f:
                f.write(header)
                SIMT_XGBoost_gen(thresholds, features, childL, childR, parallelism, txt_path, 0, groups_per_class, f)
                f.write(kernel)
                generate_inference(f, n_classes, groups_per_class)
                testset_gen(test_path, f, number_of_test_samples)
                f.write("#endif\n")

    else:
        groups = number_of_trees/parallelism
        groups = int(groups)
        print(f"  - Groups: {groups}")
        if(hybrid > 0):
            print("The model doesn't fit in DTCM. Proceeding with the hybrid approach...")
            groups_hybrid = hybrid // parallelism
            print(f"Only {hybrid} trees / {groups_hybrid} groups will fit in DTCM!")
            header_name = f"{csv_stem}_XGB_simt_hybrid_G{groups}_D{max_depth}_RS{random_seed}.h"
            txt_name = f"{csv_stem}_XGB_simt_hybrid_G{groups}_D{max_depth}_RS{random_seed}.txt"

            header_path = models_path / header_name
            txt_path = models_path / txt_name

            header = generate_header_binary(groups, sample_size, parallelism, number_of_test_samples, maxv, minv)
            kernel = generate_kernel()

            with open(header_path, "w") as f:
                f.write(header)
                SIMT_XGBoost_gen_binary(thresholds, features, childL, childR, parallelism, txt_path, groups_hybrid, f)
                f.write(kernel)
                generate_inference_binary_hybrid(f, groups, groups_hybrid)
                testset_gen(test_path, f, number_of_test_samples)
                f.write("#endif\n")
        else:
            print("The model fits entirely in DTCM!")
            header_name = f"{csv_stem}_XGB_simt_G{groups}_D{max_depth}_RS{random_seed}.h"
            txt_name = f"{csv_stem}_XGB_simt_G{groups}_D{max_depth}_RS{random_seed}.txt"

            header_path = models_path / header_name
            txt_path = models_path / txt_name

            header = generate_header_binary(groups, sample_size, parallelism, number_of_test_samples, maxv, minv)
            kernel = generate_kernel()

            with open(header_path, "w") as f:
                f.write(header)
                SIMT_XGBoost_gen_binary(thresholds, features, childL, childR, parallelism, txt_path, 0, f)
                f.write(kernel)
                generate_inference_binary(f, groups)
                testset_gen(test_path, f, number_of_test_samples)
                f.write("#endif\n")
