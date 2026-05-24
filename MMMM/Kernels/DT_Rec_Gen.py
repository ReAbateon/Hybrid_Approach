import os
import re
from collections import deque
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

DELTA = 32767


def parse_xgb_tree(tree_text):
    nodes = {}
    re_split = re.compile(
        r"^(?P<id>\d+):\[(?P<feat>.+)<(?P<thresh>.+)\] yes=(?P<yes>\d+),no=(?P<no>\d+)"
    )
    re_leaf = re.compile(r"^(?P<id>\d+):leaf=(?P<val>.+)")

    for line in tree_text.splitlines():
        line = line.strip()
        m_split = re_split.match(line)
        if m_split:
            d = m_split.groupdict()
            nodes[int(d["id"])] = {
                "feat": int(d["feat"]),
                "thresh": float(d["thresh"]),
                "yes": int(d["yes"]),
                "no": int(d["no"]),
                "is_leaf": False,
            }
        else:
            m_leaf = re_leaf.match(line)
            if m_leaf:
                d = m_leaf.groupdict()
                nodes[int(d["id"])] = {"val": float(d["val"]), "is_leaf": True}
    return nodes


def DT_Rec_generator(model, txt_name, fh, hybrid_trees, task):
    forest_data = []
    if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        for estimator in model.estimators_:
            forest_data.append({"type": "RF", "tree": estimator.tree_})
    else:
        booster = model.get_booster()
        dump = booster.get_dump()
        for tree_text in dump:
            nodes = parse_xgb_tree(tree_text)
            forest_data.append({"type": "XGB", "nodes": nodes})

    with open(txt_name, "w") as f:
        for i, g in enumerate(forest_data):
            internal_nodes = []
            node_map = {}

            queue = deque([0])
            while queue:
                curr = queue.popleft()
                if g["type"] == "RF":
                    tree = g["tree"]
                    left, right = tree.children_left[curr], tree.children_right[curr]
                    if left != -1:
                        node_map[curr] = len(internal_nodes)
                        internal_nodes.append(curr)
                        queue.extend([left, right])
                else:
                    node = g["nodes"][curr]
                    if not node["is_leaf"]:
                        node_map[curr] = len(internal_nodes)
                        internal_nodes.append(curr)
                        queue.extend([node["yes"], node["no"]])

            attr = "DTCM" if i < hybrid_trees else "RAM_BIG"
            suffix = "_dtcm" if attr == "DTCM" else ""
            tree_name = f"tree{suffix}{i}"

            fh.write(f"{attr} Node_Rec {tree_name}[] = {{\n")
            for idx, old_id in enumerate(internal_nodes):
                if g["type"] == "RF":
                    tree = g["tree"]
                    feat = tree.feature[old_id]
                    thresh = tree.threshold[old_id]
                    left_id, right_id = (
                        tree.children_left[old_id],
                        tree.children_right[old_id],
                    )

                    if tree.children_left[left_id] == -1:
                        left_ptr = "&delta"
                        left_class = (
                            int(np.argmax(tree.value[left_id][0])) if task == 0 else 1
                        )
                    else:
                        left_ptr = f"&{tree_name}[{node_map[left_id]}]"
                        left_class = 0

                    if tree.children_left[right_id] == -1:
                        right_ptr = "&delta"
                        right_class = (
                            int(np.argmax(tree.value[right_id][0])) if task == 0 else 1
                        )
                    else:
                        right_ptr = f"&{tree_name}[{node_map[right_id]}]"
                        right_class = 0
                else:
                    node = g["nodes"][old_id]
                    feat, thresh = node["feat"], node["thresh"]
                    left_id, right_id = node["yes"], node["no"]

                    if g["nodes"][left_id]["is_leaf"]:
                        left_ptr = "&delta"
                        left_class = 1
                    else:
                        left_ptr = f"&{tree_name}[{node_map[left_id]}]"
                        left_class = 0

                    if g["nodes"][right_id]["is_leaf"]:
                        right_ptr = "&delta"
                        right_class = 1
                    else:
                        right_ptr = f"&{tree_name}[{node_map[right_id]}]"
                        right_class = 0

                comma = "," if idx < len(internal_nodes) - 1 else ""
                fh.write(
                    f"    {{{feat}, {int(thresh)}, {left_ptr}, {right_ptr}, {left_class}, {right_class}}}{comma}\n"
                )
            fh.write("};\n\n")


def generate_header(trees, sample_size, test_rows):
    return f"""#ifndef INC_DT_REC_H_
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
int16_t model_classes[TREES] = {{0}};
typedef struct Node_Rec{{
	uint16_t feature; int16_t threshold;
	struct Node_Rec *left; struct Node_Rec *right;
	int16_t left_class; int16_t right_class;
}}Node_Rec;
Node_Rec delta;
"""


def generate_kernel():
    return """
int16_t dt_rec(Node_Rec *node, int16_t *sample){
	if(sample[node->feature] <= node->threshold){
		if(node->left != &delta) return dt_rec(node->left, sample);
		else return node->left_class;
	}else{
		if(node->right != &delta) return dt_rec(node->right, sample);
		else return node->right_class;
	}
}
"""


def generate_inference(fh, trees, hybrid_trees):
    fh.write("static inline void inference(int16_t* sample){\n")
    for i in range(trees):
        suffix = "_dtcm" if i < hybrid_trees else ""
        fh.write(f"   model_classes[{i}] = dt_rec(&tree{suffix}{i}[0], sample);\n")
    fh.write("}\n")


def testset_gen(X_test, fh):
    fh.write(f"RAM_BIG int16_t testset[TEST_ROWS][SAMPLE_SIZE] = {{\n")
    for i in range(len(X_test)):
        row_str = ", ".join(str(int(v)) for v in X_test.iloc[i].values)
        fh.write(f"  {{{row_str}}},\n")
    fh.write("};\n")


def generate_dtrec(model, task, csv_path, X_test):
    csv_path = Path(csv_path)
    output_dir = csv_path.parent / "DT_Rec_Output"
    output_dir.mkdir(exist_ok=True)

    n_trees = (
        len(model.estimators_) if hasattr(model, "estimators_") else model.n_estimators
    )
    sample_size = X_test.shape[1]
    test_rows = X_test.shape[0]

    total_internal_nodes = 0
    if hasattr(model, "estimators_"):
        for estimator in model.estimators_:
            total_internal_nodes += (
                estimator.tree_.node_count - estimator.tree_.n_leaves
            )
    else:
        booster = model.get_booster()
        dump = booster.get_dump()
        re_split = re.compile(r"^\d+:\[.*\]")
        for t in dump:
            for line in t.splitlines():
                if re_split.match(line.strip()):
                    total_internal_nodes += 1

    weight_bytes = total_internal_nodes * 16
    if weight_bytes > 128 * 1024:
        avg_internal = total_internal_nodes / n_trees
        hybrid_trees = int((128 * 1024) // (avg_internal * 16))
    else:
        hybrid_trees = n_trees

    header_path = output_dir / f"{csv_path.stem}_dtrec.h"
    txt_path = output_dir / f"{csv_path.stem}_dtrec.txt"

    with open(header_path, "w") as fh:
        fh.write(generate_header(n_trees, sample_size, test_rows))
        DT_Rec_generator(model, txt_path, fh, hybrid_trees, task)
        fh.write(generate_kernel())
        generate_inference(fh, n_trees, hybrid_trees)
        testset_gen(X_test, fh)
        fh.write("#endif\n")

    return weight_bytes
