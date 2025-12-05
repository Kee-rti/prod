import json
import os


def node_to_js(node):
    # If leaf
    if 'leaf' in node:
        return repr(float(node['leaf']))

    # split key like 'f0' -> index 0
    split = node.get('split') or node.get('feature')
    cond = node.get('split_condition') or node.get('threshold')
    if split is None or cond is None:
        # fallback
        return '0.0'

    idx = ''.join(ch for ch in split if ch.isdigit())
    if idx == '':
        idx = '0'

    yes = node.get('yes')
    no = node.get('no')
    missing = node.get('missing', yes)

    # find child nodes
    children = {c['nodeid']: c for c in node.get('children', [])}

    yes_node = children.get(yes)
    no_node = children.get(no)
    missing_node = children.get(missing) or yes_node

    yes_expr = node_to_js(yes_node) if yes_node is not None else '0.0'
    no_expr = node_to_js(no_node) if no_node is not None else '0.0'

    # Use ternary: (features[idx] < cond ? yes_expr : no_expr)
    return f"((features[{idx}] < {repr(float(cond))}) ? ({yes_expr}) : ({no_expr}))"


def compile_forest(tree_jsons):
    # tree_jsons: list of parsed JSON trees (root nodes)
    tree_exprs = []
    for tree in tree_jsons:
        expr = node_to_js(tree)
        tree_exprs.append(expr)
    # Sum of tree outputs
    sum_expr = ' + '.join(f'({e})' for e in tree_exprs) or '0.0'
    # compute prob via sigmoid
    js = (
        "function predictXGB(features, threshold=0.86){\n"
        "  const score = " + sum_expr + ";\n"
        "  const prob = 1/(1+Math.exp(-score));\n"
        "  return {\n    prob: prob,\n    label: prob > threshold ? 1 : 0\n  };\n}\n\n// expose globally\nglobalThis.predictXGB = predictXGB;\n"
    )
    return js


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, '../extension/model')
    trees_path = os.path.join(model_dir, 'xgb_trees.json')
    out_js = os.path.join(script_dir, '../extension/lib/xgb_compiled.js')

    if not os.path.exists(trees_path):
        print('xgb_trees.json not found. Run threshold_sweep.py first to export trees.')
        return

    with open(trees_path, 'r') as f:
        tree_strs = json.load(f)

    tree_jsons = [json.loads(s) for s in tree_strs]

    js = compile_forest(tree_jsons)

    with open(out_js, 'w') as f:
        f.write('// Auto-generated XGBoost compiled predictor\n')
        f.write(js)

    print('Wrote compiled JS predictor to', out_js)


if __name__ == '__main__':
    main()
