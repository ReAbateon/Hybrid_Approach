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


def DT_Rec_generator(path, num_internal_nodes, txt_name, fh, hybrid, task):
    model = joblib.load(path)
    DELTA = 32767
    num_nodes_list = []

    with open(txt_name, "w") as f:
        for i, estimator in enumerate(model.estimators_):
            tree = estimator.tree_

            # Map original node indices to internal node indices (the array indices)
            node_count = tree.node_count
            orig_to_internal = [-1] * node_count
            internal_idx = 0
            for node_id in range(node_count):
                if tree.children_left[node_id] != -1:  # Internal node
                    orig_to_internal[node_id] = internal_idx
                    internal_idx += 1

            struct_init_list = []
            for node_id in range(node_count):
                if tree.children_left[node_id] == -1:
                    continue

                feat = tree.feature[node_id]
                thresh = int(tree.threshold[node_id])

                left_child = tree.children_left[node_id]
                right_child = tree.children_right[node_id]

                left_ptr = "NULL"
                right_ptr = "NULL"
                left_class = 0
                right_class = 0

                # DT_Rec array names
                tree_name = (
                    f"tree_dtcm{i}" if (hybrid == -1 or i < hybrid) else f"tree{i}"
                )

                # Left side
                if tree.children_left[left_child] == -1:  # Leaf
                    left_ptr = "&delta"
                    if task == 0:
                        left_class = int(np.argmax(tree.value[left_child][0]))
                    else:
                        left_class = 1
                else:
                    left_ptr = f"&{tree_name}[{orig_to_internal[left_child]}]"

                # Right side
                if tree.children_left[right_child] == -1:  # Leaf
                    right_ptr = "&delta"
                    if task == 0:
                        right_class = int(np.argmax(tree.value[right_child][0]))
                    else:
                        right_class = 1
                else:
                    right_ptr = f"&{tree_name}[{orig_to_internal[right_child]}]"

                struct_init_list.append(
                    f"{{ {feat}, {thresh}, {left_ptr}, {right_ptr}, {left_class}, {right_class} }}"
                )

            f.write(f"Tree {i}   Number of Nodes = {node_count}\n")

            init_str = ", ".join(struct_init_list)
            section = "DTCM" if (hybrid == -1 or i < hybrid) else "RAM_BIG"
            tree_name = f"tree_dtcm{i}" if (hybrid == -1 or i < hybrid) else f"tree{i}"

            fh.write(
                f"{section} Node_Rec {tree_name}[{num_internal_nodes[i]}] = {{{init_str}}};\n\n"
            )
            num_nodes_list.append(num_internal_nodes[i])

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


def internal_nodes_calc(path, accuracy, mse, mae, r2, task):
    model = joblib.load(path)

    num_internal_nodes = []
    total_nodes = 0
    hybrid = -1
    counter = 0
    size_bytes = 0
    size_kb = 0

    for i, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        internal_nodes = tree.node_count - tree.n_leaves

        total_nodes += tree.node_count
        num_internal_nodes.append(internal_nodes)

        size_bytes += internal_nodes * 16  # 16 bytes per internal node for DT_Rec

        if size_bytes > 130000 and hybrid == -1:
            hybrid = counter

        counter += 1

    size_kb += size_bytes / 1024

    print(f"Model Stats:")
    print(f"  - Total Nodes: {total_nodes}")
    print(f"  - Dimension: {size_bytes} Bytes ({size_kb:.2f} KB)")
    if task == 0:
        print(f"  - Accuracy: {accuracy:.4f}")
    else:
        print(f"  - MSE: {mse:.4f}")
        print(f"  - MAE: {mae:.4f}")
        print(f"  - R2:  {r2:.4f}")

    return num_internal_nodes, hybrid


def generate_inference(fh, trees, hybrid):
    fh.write("static inline void inference(int16_t* sample){\n")
    for i in range(trees):
        tree_name = f"tree_dtcm{i}" if (hybrid == -1 or i < hybrid) else f"tree{i}"
        fh.write(f"   classes[{i}] = dt_rec({tree_name}, sample);\n")
    fh.write("}\n\n")


def generate_dtrec(
    path, number_of_trees, max_depth, random_seed, number_of_test_samples
):
    test_path, joblib_path, sample_size, task, accuracy, mse, mae, r2, csv_stem = (
        trainRF.training(
            path, number_of_trees, max_depth, random_seed, number_of_test_samples
        )
    )

    num_internal_nodes, hybrid = internal_nodes_calc(
        joblib_path, accuracy, mse, mae, r2, task
    )

    models_path = Path(f"Models/{csv_stem}/dtrec_RF_T{number_of_trees}_D{max_depth}")
    os.makedirs(models_path, exist_ok=True)

    if hybrid != -1:
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
            num_nodes_list = DT_Rec_generator(
                joblib_path, num_internal_nodes, txt_path, f, hybrid, task
            )
            f.write(kernel)
            generate_inference(f, number_of_trees, hybrid)
            testset_gen(test_path, f, number_of_test_samples)
            f.write("#endif\n")
    else:
        print("The model fits entirely in DTCM!")
        header_name = (
            f"{csv_stem}_RF_dtrec_T{number_of_trees}_D{max_depth}_RS{random_seed}.h"
        )
        txt_name = (
            f"{csv_stem}_RF_dtrec_T{number_of_trees}_D{max_depth}_RS{random_seed}.txt"
        )

        header_path = models_path / header_name
        txt_path = models_path / txt_name

        header = generate_header(number_of_trees, sample_size, number_of_test_samples)
        kernel = generate_kernel()

        with open(header_path, "w") as f:
            f.write(header)
            num_nodes_list = DT_Rec_generator(
                joblib_path, num_internal_nodes, txt_path, f, -1, task
            )
            f.write(kernel)
            generate_inference(f, number_of_trees, -1)
            testset_gen(test_path, f, number_of_test_samples)
            f.write("#endif\n")
