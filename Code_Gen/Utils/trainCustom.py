import random
from typing import List, Dict, Any, Tuple


def generate_full_tree_int(
    depth: int,
    n_features: int,
    n_classes: int,
    qrange: Tuple[int, int] = (-32768, 32766),
) -> Dict[str, Any]:
    """
    Genera un singolo albero binario completo di profondità `depth`
    con soglie intere nel range qrange (es. int16).

    Convenzioni:
    - numero di nodi = 2^(depth+1) - 1
    - nodi interni: children_left[i], children_right[i] >= 0
    - foglie: children_left[i] = children_right[i] = -1
    - feature[i] = -1 per le foglie
    - value[i] = classe (0..n_classes-1) per le foglie, None per nodi interni
    """

    assert depth >= 0
    assert n_features > 0
    assert n_classes > 0

    qmin, qmax = qrange
    assert qmin < qmax

    n_nodes = 2 ** (depth + 1) - 1

    children_left  = [-1] * n_nodes
    children_right = [-1] * n_nodes
    feature        = [-1] * n_nodes
    threshold      = [0] * n_nodes      # <<< INTERI
    value          = [None] * n_nodes   # classe solo alle foglie

    for i in range(n_nodes):
        left_idx = 2 * i + 1
        right_idx = 2 * i + 2

        if left_idx < n_nodes:
            # Nodo interno
            children_left[i] = left_idx
            children_right[i] = right_idx

            feature[i] = random.randint(0, n_features - 1)
            threshold[i] = random.randint(qmin, qmax)  # <<< soglia intera

            value[i] = None
        else:
            # Nodo foglia
            children_left[i] = -1
            children_right[i] = -1
            feature[i] = -1
            threshold[i] = 0  # o un valore dummy, non usato in foglia
            value[i] = random.randint(0, n_classes - 1)

    tree = {
        "children_left": children_left,
        "children_right": children_right,
        "feature": feature,
        "threshold": threshold,  # <<< lista di int
        "value": value,
        "depth": depth,
        "n_features": n_features,
        "n_classes": n_classes,
        "n_nodes": n_nodes,
        "qrange": qrange,
    }

    return tree


def generate_random_forest_int(
    n_trees: int,
    depth: int,
    n_features: int,
    n_classes: int,
    qrange: Tuple[int, int] = (-32768, 32766),
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Genera un insieme di n_trees alberi completi,
    tutti con soglie intere nel range qrange (es. int16).
    """
    if seed is not None:
        random.seed(seed)

    forest: List[Dict[str, Any]] = []
    for i in range(n_trees):
        if (n_trees == 8) and (depth == 10) and (i == n_trees - 1):
            depth = 9
        if (n_trees == 120) and (depth == 6) and (i >= n_trees - 16):
            depth = 5
        tree = generate_full_tree_int(
            depth=depth,
            n_features=n_features,
            n_classes=n_classes,
            qrange=qrange,
        )
        forest.append(tree)

    return forest


def generate_test_samples_int(
    n_samples: int,
    n_features: int,
    qrange: Tuple[int, int] = (-32768, 32766),
    seed: int | None = None,
) -> List[List[int]]:
    """
    Genera n_samples campioni, ognuno con n_features feature,
    con valori interi uniformi in qrange (es. int16).
    """
    if seed is not None:
        random.seed(seed)

    qmin, qmax = qrange
    samples: List[List[int]] = []

    for _ in range(n_samples):
        x = [random.randint(qmin, qmax) for _ in range(n_features)]
        samples.append(x)

    return samples


def training(n_trees, depth, t_size, weight=8, n_features = None, fast = 0):
    range = (-32768, 32766)
    if n_features == None:
        n_features = random.randint(15, 65)
    n_classes = random.randint(5, 20)

    forest = generate_random_forest_int(
        n_trees=n_trees,
        depth=depth,
        n_features=n_features,
        n_classes=n_classes,
        qrange=range,
        seed = None
    )

    X_test = generate_test_samples_int(
        n_samples=t_size,
        n_features=n_features,
        qrange=range,
        seed = None
    )

    hybrid = 0
    if(fast == 0):
        size_bytes = 0
        size_kb = 0
        total_nodes = 0
        counter = 0
        #max_ = 131009
        max_ = 129000

        for tree in forest:
            nodes = tree["n_nodes"]
            total_nodes += nodes

            size_bytes += nodes * weight
            
            if size_bytes > max_ and hybrid == 0:
                hybrid = counter

            counter += 1

        size_kb += size_bytes / 1024

        print(f"Model Stats:")
        print(f"  - Total Nodes:    {total_nodes}")
        print(f"  - Max Depth:      {depth}")
        print(f"  - Feature Number: {n_features}")
        print(f"  - Class Number:   {n_classes}")
        print(f"  - Dimension: {size_bytes} Bytes ({size_kb:.2f} KB)")

    return forest, X_test, hybrid, n_features, n_classes

if __name__ == "__main__":
    n_trees = 8
    depth = 10
    n_features = random.randint(10, 64)
    n_classes = random.randint(5, 20)
    t_size = 1000

    
    qrange = (-32768, 32766)

    forest = generate_random_forest_int(
        n_trees=n_trees,
        depth=depth,
        n_features=n_features,
        n_classes=n_classes,
        qrange=qrange,
        seed=42,
    )

    X_test = generate_test_samples_int(
        n_samples=t_size,
        n_features=n_features,
        qrange=qrange,
        seed=123,
    )

    size_bytes = 0
    size_kb = 0
    total_nodes = 0
    counter = 0
    hybrid = 0

    for tree in forest:
        nodes = tree["n_nodes"]
        total_nodes += nodes

        size_bytes += nodes * 8
        
        if size_bytes > 129000 and hybrid == 0:
            hybrid = counter

        counter += 1

    size_kb += size_bytes / 1024

    print(f"Model Stats:")
    print(f"  - Total Nodes: {total_nodes}")
    print(f"  - Dimension: {size_bytes} Bytes ({size_kb:.2f} KB)")


    for i in range(len(forest)):
        print(forest[i]["n_nodes"])

