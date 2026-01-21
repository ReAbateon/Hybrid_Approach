import joblib
import csv
import numpy as np
import shutil
from collections import deque

import os

from pathlib import Path

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

def DT_Rec_generator(path, num_internal_nodes, txt_name, fh, hybrid, task):
    model = joblib.load(path)
    
    DELTA = 32767

    num_nodes_list = [] 

    with open(txt_name, "w") as f:
        for i, estimator in enumerate(model.estimators_):
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
                    child_list.append(np.int16(class_idx))          # child[2n+1] = placeholder

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
                num_nodes_list.append(len(feature_list))
            else:
                if i >= hybrid:
                    fh.write(f"RAM_BIG Node_Rec tree{i}[{num_internal_nodes[i]}];\n\n")
                    num_nodes_list.append(len(feature_list))
                else:
                    fh.write(f"DTCM Node_Rec tree_dtcm{i}[{num_internal_nodes[i]}];\n\n")
                    num_nodes_list.append(len(feature_list))         
    return num_nodes_list

def generate_header(trees, sample_size, test_rows):
    return f"""\
#ifndef INC_DT_REC_H_
#define INC_DT_REC_H_   

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

def internal_nodes_calc(path):
    model = joblib.load(path)

    num_internal_nodes = []
    total_nodes_list = []
    total_nodes = 0
    hybrid = 0
    counter = 0
    size_bytes = 0
    size_kb = 0

    for i,estimator  in enumerate(model):
        tree = estimator.tree_
        internal_nodes = tree.node_count - tree.n_leaves
        
        total_nodes += tree.node_count
        num_internal_nodes.append(internal_nodes)

    for elem in num_internal_nodes:
        size_bytes += elem * 16

        if size_bytes > 129000 and hybrid == 0:
            hybrid = counter
        
        counter += 1
    
    size_kb += size_bytes / 1024

    #print(num_internal_nodes)

    print(f"  - Total Nodes: {total_nodes}")
    print(f"  - Dimension: {size_bytes} Bytes ({size_kb:.2f} KB)")

    return num_internal_nodes, hybrid

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

def generate_dtrec(path, number_of_trees, max_depth, random_seed, number_of_test_samples):
    test_path, joblib_path, sample_size, hybrid, csv_stem, task = trainRF.training(path, number_of_trees, max_depth, random_seed, number_of_test_samples, fast=1)

    num_internal_nodes, hybrid = internal_nodes_calc(joblib_path)

    models_path = Path(f"Models/{csv_stem}/dtrec_RF_T{number_of_trees}_D{max_depth}")
    os.makedirs(models_path, exist_ok=True)

    if hybrid > 0:
        print("The model doesn't fit in DTCM. Proceeding with the hybrid approach...")
        print(f"Only {hybrid} trees will fit in DTCM!")

        header_name = f"{csv_stem}_RF_dtrec_hybrid_T{number_of_trees}_D{max_depth}_RS{random_seed}.h"
        txt_name = f"{csv_stem}_RF_dtrec_hybrid_T{number_of_trees}_D{max_depth}_RS{random_seed}.txt"

        header_path = models_path / header_name
        txt_path = models_path / txt_name

        header = generate_header(number_of_trees, sample_size, number_of_test_samples)
        kernel = generate_kernel()

        with open(header_path, "w") as f:
            f.write(header)
            num_nodes_list = DT_Rec_generator(joblib_path, num_internal_nodes, txt_path, f, hybrid, task)
            f.write(kernel)
            generate_init_hybrid(f, number_of_trees, num_nodes_list, hybrid)
            generate_inference_hybrid(f, number_of_trees, hybrid)
            testset_gen(test_path, f, number_of_test_samples)
            f.write("#endif\n")
    else:
        header_name = f"{csv_stem}_RF_dtrec_T{number_of_trees}_D{max_depth}_RS{random_seed}.h"
        txt_name = f"{csv_stem}_RF_dtrec_T{number_of_trees}_D{max_depth}_RS{random_seed}.txt"

        header_path = models_path / header_name
        txt_path = models_path / txt_name

        header = generate_header(number_of_trees, sample_size, number_of_test_samples)
        kernel = generate_kernel()

        with open(header_path, "w") as f:
            f.write(header)
            num_nodes_list = DT_Rec_generator(joblib_path, num_internal_nodes, txt_path, f, 0, task)
            f.write(kernel)
            generate_init(f, number_of_trees, num_nodes_list)
            generate_inference(f, number_of_trees)
            testset_gen(test_path, f, number_of_test_samples)
            f.write("#endif\n")
    