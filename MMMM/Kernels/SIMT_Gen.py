import os
import re
from collections import deque
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

DELTA = 32767


def get_nodes_by_level(tree):
    children_left = tree.children_left
    children_right = tree.children_right
    levels = []
    queue = deque([(0, 0)])
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

    levels = []
    queue = deque([(0, 0)])
    while queue:
        node_id, level = queue.popleft()
        if level == len(levels):
            levels.append([])
        levels[level].append(node_id)
        if not nodes[node_id]["is_leaf"]:
            queue.append((nodes[node_id]["yes"], level + 1))
            queue.append((nodes[node_id]["no"], level + 1))

    return nodes, levels


def parallel_generator(model, txt_name, num_group, fh, groups_hybrid, task=0):
    forest_data = []
    if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        for estimator in model.estimators_:
            tree = estimator.tree_
            levels = get_nodes_by_level(tree)
            forest_data.append({"type": "RF", "tree": tree, "levels": levels})
    else:
        booster = model.get_booster()
        dump = booster.get_dump()
        for tree_text in dump:
            nodes, levels = parse_xgb_tree(tree_text)
            forest_data.append({"type": "XGB", "nodes": nodes, "levels": levels})

    groupindex = 0
    with open(txt_name, "w") as f:
        for i in range(0, len(forest_data), num_group):
            group = forest_data[i : i + num_group]
            max_depth = max(len(g["levels"]) for g in group)
            num_nodes, num_added_nodes = 0, 0
            feature_list, threshold_list, childL_list, childR_list = [], [], [], []

            for level_idx in range(max_depth):
                for g in group:
                    if level_idx < len(g["levels"]):
                        node_ids = g["levels"][level_idx]
                        for node_id in node_ids:
                            if g["type"] == "RF":
                                tree = g["tree"]
                                is_leaf = tree.children_left[node_id] == -1
                                if is_leaf:
                                    feat, thresh = 0, DELTA
                                    cl = num_nodes
                                    cr = (
                                        int(np.argmax(tree.value[node_id][0]))
                                        if task == 0
                                        else 1
                                    )
                                else:
                                    feat, thresh = (
                                        tree.feature[node_id],
                                        tree.threshold[node_id],
                                    )
                                    cl, cr = (
                                        2 * num_added_nodes + num_group,
                                        2 * num_added_nodes + num_group + 1,
                                    )
                                    num_added_nodes += 1
                            else:
                                node = g["nodes"][node_id]
                                if node["is_leaf"]:
                                    feat, thresh = 0, DELTA
                                    cl = num_nodes
                                    cr = 1
                                else:
                                    feat, thresh = node["feat"], node["thresh"]
                                    cl, cr = (
                                        2 * num_added_nodes + num_group,
                                        2 * num_added_nodes + num_group + 1,
                                    )
                                    num_added_nodes += 1

                            feature_list.append(np.uint16(feat))
                            threshold_list.append(np.int16(thresh))
                            childL_list.append(np.uint16(cl))
                            childR_list.append(np.uint16(cr))
                            num_nodes += 1

            feature_str = ", ".join(str(f) for f in feature_list)
            threshold_str = ", ".join(str(t) for t in threshold_list)
            childL_str = ", ".join(str(c) for c in childL_list)
            childR_str = ", ".join(str(c) for c in childR_list)

            attr = "DTCM" if groupindex < groups_hybrid else "RAM_BIG"
            prefix = "_dtcm" if attr == "DTCM" else ""
            fh.write(
                f"{attr} uint16_t feature{prefix}{groupindex}[{len(feature_list)}] = {{{feature_str}}};\n"
            )
            fh.write(
                f"{attr} int16_t threshold{prefix}{groupindex}[{len(threshold_list)}] = {{{threshold_str}}};\n"
            )
            fh.write(
                f"{attr} uint16_t childL{prefix}{groupindex}[{len(childL_list)}] = {{{childL_str}}};\n"
            )
            fh.write(
                f"{attr} uint16_t childR{prefix}{groupindex}[{len(childR_list)}] = {{{childR_str}}};\n\n"
            )
            groupindex += 1


def generate_header(groups, sample_size, parallelism, test_rows):
    return f"""#ifndef INC_SIMT_H_
#define INC_SIMT_H_
#include "arm_mve.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdbool.h"
#define GROUPS         {int(groups)}
#define DELTA          32767
#define PARALLELISM    {parallelism}
#define SAMPLE_SIZE    {sample_size}
#define TEST_ROWS      {test_rows}
#define DTCM __attribute__((section(".dtcm_data"), aligned(32)))
#define RAM_BIG __attribute__((section(".big_data"), aligned(32)))
typedef struct{{ uint16_t class[PARALLELISM]; }}classes;
uint16_t final_results[GROUPS * PARALLELISM];
"""


def generate_kernel():
    return """
bool all_active(mve_pred16_t mask){ return (mask == 0xFFFF); }
classes kernel_simt(int16_t* threshold, uint16_t* features, uint16_t* childL, uint16_t* childR, int16_t* sample){
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
	classes result; vst1q_u16(result.class, b);
	return result;
}
"""


def generate_inference(fh, groups, groups_hybrid):
    fh.write(
        "static inline void inference(int16_t* sample){\n   uint8_t i = 0;\n   classes myclass;\n"
    )
    for j in range(groups):
        prefix = "_dtcm" if j < groups_hybrid else ""
        fh.write(
            f"   myclass = kernel_simt(threshold{prefix}{j}, feature{prefix}{j}, childL{prefix}{j}, childR{prefix}{j}, sample);\n"
        )
        fh.write(
            f"   memcpy(&final_results[i], myclass.class, sizeof(uint16_t) * PARALLELISM);\n   i += PARALLELISM;\n"
        )
    fh.write("}\n")


def testset_gen(X_test, fh):
    fh.write(f"RAM_BIG int16_t testset[TEST_ROWS][SAMPLE_SIZE] = {{\n")
    for i in range(len(X_test)):
        row_str = ", ".join(str(int(v)) for v in X_test.iloc[i].values)
        fh.write(f"  {{{row_str}}},\n")
    fh.write("};\n")


def generate_simt(model, task, csv_path, X_test, parallelism=8):
    csv_path = Path(csv_path)
    output_dir = csv_path.parent / "SIMT_Output"
    output_dir.mkdir(exist_ok=True)

    n_trees = (
        len(model.estimators_) if hasattr(model, "estimators_") else model.n_estimators
    )
    groups = n_trees // parallelism
    sample_size = X_test.shape[1]
    test_rows = X_test.shape[0]

    total_nodes = 0
    if hasattr(model, "estimators_"):
        for estimator in model.estimators_:
            total_nodes += estimator.tree_.node_count
    else:
        booster = model.get_booster()
        dump = booster.get_dump()
        for t in dump:
            for line in t.splitlines():
                if line.strip() and ":" in line:
                    total_nodes += 1

    weight_bytes = total_nodes * 8
    if weight_bytes > 128 * 1024:
        avg_nodes = total_nodes / n_trees
        hybrid_trees = int((128 * 1024) // (avg_nodes * 8))
        groups_hybrid = hybrid_trees // parallelism
    else:
        groups_hybrid = groups

    header_path = output_dir / f"{csv_path.stem}_simt.h"
    txt_path = output_dir / f"{csv_path.stem}_simt.txt"

    with open(header_path, "w") as fh:
        fh.write(generate_header(groups, sample_size, parallelism, test_rows))
        parallel_generator(model, txt_path, parallelism, fh, groups_hybrid, task)
        fh.write(generate_kernel())
        generate_inference(fh, groups, groups_hybrid)
        testset_gen(X_test, fh)
        fh.write("#endif\n")

    return weight_bytes
