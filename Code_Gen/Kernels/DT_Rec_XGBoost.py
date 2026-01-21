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

        for i in range(len(n_internal_nodes)):
            total_nodes += n_nodes[i]
            total_internal_nodes += n_internal_nodes[i]
            size_bytes += n_internal_nodes[i] * 16

            if size_bytes > 129000 and hybrid == 0:
                hybrid = counter

            counter += 1
        
        size_kb += size_bytes / 1024
        print(f"Model Stats:")
        print(f"  - Total Trees: {len(dump)}")
        print(f"  - Total Nodes: {total_nodes}")
        print(f"  - Total Internal Nodes: {total_internal_nodes}")
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
                
        return thresholds, features, childL, childR, hybrid, len(dump), max_vector, min_vector, n_internal_nodes, max_depth
    
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

        for i in range(len(n_internal_nodes)):
            total_nodes += n_nodes[i]
            total_internal_nodes += n_internal_nodes[i]
            size_bytes += n_internal_nodes[i] * 16

            if size_bytes > 129000 and hybrid == 0:
                hybrid = counter

            counter += 1
        
        size_kb += size_bytes / 1024
        range_ = maxv - minv

        if task == 0:
            print(f"Model Stats:")
            print(f"  - Total Trees: {len(dump)}")
            print(f"  - Total Nodes: {total_nodes}")
            print(f"  - Total Internal Nodes: {total_internal_nodes}")
            print(f"  - Dimension: {size_bytes} Bytes ({size_kb:.2f} KB)")
            print(f"  - Classes: {num_class}")
            print(f"  - Accuracy: {acc:.4f}")
            print(f"  - Max: {maxv}")
            print(f"  - Min: {minv}")
        else:
            print(f"Model Stats:")
            print(f"  - Total Trees: {len(dump)}")
            print(f"  - Total Nodes: {total_nodes}")
            print(f"  - Total Internal Nodes: {total_internal_nodes}")
            print(f"  - Dimension: {size_bytes} Bytes ({size_kb:.2f} KB)")
            print(f"  - Classes: {num_class}")
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
        return thresholds, features, childL, childR, hybrid, len(dump), maxv, minv, n_internal_nodes, max_depth

def DT_Rec_XGBoost(thresholds, features, childL, childR, num_internal_nodes, txt_name, fh, hybrid):
    DELTA = 32767

    num_nodes_list = [] 

    with open(txt_name, "w") as f:
        for tree in range(len(num_internal_nodes)):
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

def generate_init(fh, trees, num_nodes_list):
    fh.write("static inline void init_ram(){\n") 
    for i in range(trees):
        fh.write(f"   init_struct_rec(tree{i}, {num_nodes_list[i]}, Rec_feature{i}, Rec_threshold{i}, Rec_child{i});\n")
    fh.write("}\n\n")

    fh.write("static inline void init_dtcm(){\n") 
    for i in range(trees):
        fh.write(f"   init_struct_rec(tree_dtcm{i}, {num_nodes_list[i]}, Rec_feature{i}, Rec_threshold{i}, Rec_child{i});\n")
    fh.write("}\n\n")

def generate_init_hybrid(fh, trees, num_nodes_list, hybrid):
    fh.write("static inline void init_dtrec(){\n")
    for i in range(trees):
        if i < hybrid:
            fh.write(f"   init_struct_rec(tree_dtcm{i}, {num_nodes_list[i]}, Rec_feature{i}, Rec_threshold{i}, Rec_child{i});\n")
        else:
            fh.write(f"   init_struct_rec(tree{i}, {num_nodes_list[i]}, Rec_feature{i}, Rec_threshold{i}, Rec_child{i});\n")
    fh.write("}\n\n")

def generate_inference(fh, trees):
    fh.write("static inline void inference_ram(int16_t* sample){\n")
    
    for i in range(trees):
        fh.write(f"   classes[{i}] = dt_rec(tree{i}, sample);\n")

    fh.write("}\n\n")

    fh.write("static inline void inference_dtcm(int16_t* sample){\n")

    for i in range(trees):
        fh.write(f"   classes[{i}] = dt_rec(tree_dtcm{i}, sample);\n")  
        
    fh.write("}\n\n")


def generate_inference_hybrid(fh, trees, hybrid):
    fh.write("static inline void inference(int16_t* sample){\n")
    
    for i in range(trees):
        if i < hybrid:
            fh.write(f"   classes[{i}] = dt_rec(tree_dtcm{i}, sample);\n")
        else:
            fh.write(f"   classes[{i}] = dt_rec(tree{i}, sample);\n")

    fh.write("}\n\n")

def generate_header(trees, n_classes, sample_size, test_rows, maxv, minv):
    return f"""\
#ifndef INC_DT_REC_H_
#define INC_DT_REC_H_

#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdbool.h"
#include "math.h"
#include "stdint.h"

# define TREES         {trees}
# define CLASSES        {int(n_classes)}
# define DELTA          32767
# define SAMPLE_SIZE    {sample_size}
# define TEST_ROWS      {test_rows}

const float MAX[{len(maxv)}] = {{{", ".join(f"{m}f" for m in maxv)}}};
const float MIN[{len(minv)}] = {{{", ".join(f"{m}f" for m in minv)}}};

#define DTCM __attribute__((section(".dtcm_data"), aligned(32)))
#define RAM_BIG __attribute__((section(".big_data"), aligned(32)))

int16_t classes[TREES];

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

def generate_header_binary(trees, sample_size, test_rows, maxv, minv):
    return f"""\
#ifndef INC_DT_REC_H_
#define INC_DT_REC_H_

#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdbool.h"
#include "math.h"
#include "stdint.h"

#define TREES          {trees}
#define DELTA          32767
#define SAMPLE_SIZE    {sample_size}
#define TEST_ROWS      {test_rows}

const float MAX = {maxv}f;
const float MIN = {minv}f;

#define DTCM __attribute__((section(".dtcm_data"), aligned(32)))
#define RAM_BIG __attribute__((section(".big_data"), aligned(32)))

int16_t classes[TREES] = {{0}};

typedef struct Node_Rec{{
	uint16_t feature;
	int16_t  threshold;

	struct Node_Rec *left;
	struct Node_Rec *right;

	uint16_t left_class;
	uint16_t right_class;
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

def generate_dtrec(path, n_trees, random_seed, number_of_test_samples):
    test_path, joblib_path, sample_size, acc, csv_stem, n_classes, task , mae, mse, r2= trainXGB.training(path, n_trees, random_seed, number_of_test_samples)
    thresholds, features, childL, childR, hybrid, number_of_trees, maxv, minv, number_of_internal_nodes, max_depth = set_model(n_classes, acc, joblib_path, task, mae, mse, r2)

    models_path = Path(f"Models/{csv_stem}/dtrec_XGB_T{n_trees}_D{max_depth}")
    os.makedirs(models_path, exist_ok=True)

    if n_classes > 2:
        trees_per_class = number_of_trees/n_classes
        if(hybrid>0):
            print("The model doesn't fit in DTCM. Proceeding with the hybrid approach...")
            print(f"Only {hybrid} trees will fit in DTCM!")

            header_name = f"{csv_stem}_XGB_dtrec_hybrid_TC{trees_per_class}_D{max_depth}_RS{random_seed}.h"
            txt_name = f"{csv_stem}_XGB_dtrec_hybrid_TC{trees_per_class}_D{max_depth}_RS{random_seed}.txt"

            header_path = models_path / header_name
            txt_path = models_path / txt_name

            header = generate_header(number_of_trees, n_classes, sample_size, number_of_test_samples, maxv, minv)
            kernel = generate_kernel()

            with open(header_path, "w") as f:
                f.write(header)
                num_nodes_list = DT_Rec_XGBoost(thresholds, features, childL, childR, number_of_internal_nodes, txt_path, f, hybrid)
                f.write(kernel)
                generate_init_hybrid(f, number_of_trees, num_nodes_list, hybrid)
                generate_inference_hybrid(f, number_of_trees, hybrid)
                testset_gen(test_path, f, number_of_test_samples)
                f.write("#endif\n")
        else:
            print("The model fits entirely in DTCM!")

            header_name = f"{csv_stem}_XGB_dtrec_TC{trees_per_class}_D{max_depth}_RS{random_seed}.h"
            txt_name = f"{csv_stem}_XGB_dtrec_TC{trees_per_class}_D{max_depth}_RS{random_seed}.txt"

            header_path = models_path / header_name
            txt_path = models_path / txt_name


            header = generate_header(number_of_trees, n_classes, sample_size, number_of_test_samples, maxv, minv)
            kernel = generate_kernel()

            with open(header_path, "w") as f:
                f.write(header)
                num_nodes_list = DT_Rec_XGBoost(thresholds, features, childL, childR, number_of_internal_nodes, txt_path, f, 0)
                f.write(kernel)
                generate_init(f, number_of_trees, num_nodes_list)
                generate_inference(f, number_of_trees)
                testset_gen(test_path, f, number_of_test_samples)
                f.write("#endif\n")
    else:
        if hybrid > 0:
            print("The model doesn't fit in DTCM. Proceeding with the hybrid approach...")
            print(f"Only {hybrid} trees will fit in DTCM!")

            header_name = f"{csv_stem}_XGB_dtrec_hybrid_T{number_of_trees}_D{max_depth}_RS{random_seed}.h"
            txt_name = f"{csv_stem}_XGB_dtrec_hybrid_T{number_of_trees}_D{max_depth}_RS{random_seed}.txt"

            header_path = models_path / header_name
            txt_path = models_path / txt_name

            header = generate_header_binary(number_of_trees, sample_size, number_of_test_samples, maxv, minv)
            kernel = generate_kernel()

            with open(header_path, "w") as f:
                f.write(header)
                num_nodes_list = DT_Rec_XGBoost(thresholds, features, childL, childR, number_of_internal_nodes, txt_path, f, hybrid)
                f.write(kernel)
                generate_init_hybrid(f, number_of_trees, num_nodes_list, hybrid)
                generate_inference_hybrid(f, number_of_trees, hybrid)
                testset_gen(test_path, f, number_of_test_samples)
                f.write("#endif\n")
        else:
            print("The model fits entirely in DTCM!")

            header_name = f"{csv_stem}_XGB_dtrec_T{number_of_trees}_D{max_depth}_RS{random_seed}.h"
            txt_name = f"{csv_stem}_XGB_dtrec_T{number_of_trees}_D{max_depth}_RS{random_seed}.txt"

            header_path = models_path / header_name
            txt_path = models_path / txt_name

            header = generate_header_binary(number_of_trees, sample_size, number_of_test_samples, maxv, minv)
            kernel = generate_kernel()

            with open(header_path, "w") as f:
                f.write(header)
                num_nodes_list = DT_Rec_XGBoost(thresholds, features, childL, childR, number_of_internal_nodes, txt_path, f, 0)
                f.write(kernel)
                generate_init(f, number_of_trees, num_nodes_list)
                generate_inference(f, number_of_trees)
                testset_gen(test_path, f, number_of_test_samples)
                f.write("#endif\n") 
