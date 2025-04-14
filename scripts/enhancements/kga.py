import re
import pandas as pd
import networkx as nx
from pathlib import Path
from scripts.util.savers import save_dataset

def logical_form_to_graph(logical_form: str) -> nx.DiGraph:
    """
    Convert a logical form string into a directed knowledge graph.
    """
    graph = nx.DiGraph()
    triples = re.findall(r"(\w+)\s*\(\s*([^,]+)\s*,\s*([^)]+)\)", logical_form)

    for predicate, subject, obj in triples:
        graph.add_node(subject)
        graph.add_node(obj)
        graph.add_edge(subject, obj, label=predicate)

    return graph

def graph_to_string(graph: nx.DiGraph) -> str:
    """
    Serialize a graph as a string of triples like: x1 -[pred]-> x2
    """
    return "; ".join(f"{u} -[{d['label']}]-> {v}" for u, v, d in graph.edges(data=True))

def augment_with_kga(input_path: str, output_path: str):
    """
    Add serialized graph representation to each logical form.
    """
    df = pd.read_csv(input_path, sep="\t", header=None)
    if df.shape[1] == 3:
        df.columns = ["input", "output", "split"]
    elif df.shape[1] == 2:
        df.columns = ["input", "output"]
    else:
        raise ValueError(f"Unexpected number of columns in {input_path}: {df.shape[1]}")

    df["graph"] = df["output"].apply(lambda lf: graph_to_string(logical_form_to_graph(lf)))
    df = df[["input", "output", "graph"]]
    save_dataset(df, enhancement="kga", split=Path(output_path).stem)
