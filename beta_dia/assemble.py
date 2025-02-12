import numpy as np
import pandas as pd
import networkx as nx

from beta_dia.log import Logger

try:
    profile
except NameError:
    profile = lambda x: x

logger = Logger.get_logger()

def plot_graph(graph):
    U = {node for node, bipartite in graph.nodes(data='bipartite') if
         bipartite == 0}
    pos = nx.bipartite_layout(graph, U)
    nx.draw(graph, pos, with_labels=True, node_color="skyblue", node_size=3000,
            edge_color="gray", font_size=12, font_weight="bold")
    import matplotlib.pyplot as plt
    plt.show()


def greedy_bipartite_vertex_cover(graph):
    graph = nx.freeze(graph)
    graph = nx.Graph(graph)
    left_nodes = [node for node, data in graph.nodes(data=True) if
                  data["bipartite"] == 0]
    right_nodes = [node for node, data in graph.nodes(data=True) if
                   data["bipartite"] == 1]
    protein_v, peptide_v = [], []

    while right_nodes:
        # select nodes with most edges, if in tie, select the best edge's nodes
        # after removing, some nodes no edges, max return 0
        df = [
            [node,
             len(graph[node]),
             max((edge['weight'] for edge in graph[node].values()), default=0)
             # sum((edge['weight'] for edge in graph[node].values()))
        ] for node in left_nodes
        ]
        df = pd.DataFrame(df, columns=['Node', 'Degree', 'CScore'])
        df = df.sort_values(by=['Degree', 'CScore'], ascending=[False, False])
        df = df.reset_index(drop=True)
        node = df.loc[0, 'Node']

        neighbors = list(graph.neighbors(node))
        protein_v.append(node)
        peptide_v.append(neighbors)

        graph.remove_nodes_from([node] + neighbors)
        left_nodes = [node for node, data in graph.nodes(data=True) if
                      data["bipartite"] == 0]
        right_nodes = [node for node, data in graph.nodes(data=True) if
                       data["bipartite"] == 1]
    return protein_v, peptide_v


def assemble_to_pg(df_input, q_cut_infer, run_or_global):
    col_q_pr = 'q_pr_' + run_or_global
    col_cscore_pr = 'cscore_pr_' + run_or_global

    if 'strip_seq' not in df_input.columns:
        if 'simple_seq' not in df_input.columns:
            df_input['simple_seq'] = df_input['pr_id'].str[:-1].replace(
                ['C\(UniMod:4\)', 'M\(UniMod:35\)'], ['c', 'm'], regex=True
            )
        df_input['strip_seq'] = df_input['simple_seq'].str.upper()

    df = df_input[df_input[col_q_pr] < q_cut_infer]

    df = df[['protein_id', 'strip_seq', col_cscore_pr]]
    df['protein_id'] = df['protein_id'].str.split(';')
    proteins = df['protein_id'].explode().values
    protein_num = df['protein_id'].apply(len)

    df = df.loc[np.repeat(df.index, protein_num)]
    df = df.reset_index(drop=True)
    df['protein_id'] = proteins

    # protein meta
    df_protein = df.groupby('protein_id', sort=False)['strip_seq'].agg(set)
    df_protein = df_protein.reset_index()
    df_protein['strip_seq'] = df_protein['strip_seq'].apply(tuple)
    df_protein = df_protein.groupby('strip_seq', sort=False)[
        'protein_id'].agg(set)
    df_protein = df_protein.reset_index()

    # corresponding
    df_protein['Protein.Meta'] = df_protein['protein_id'].str.join(';')
    proteins = df_protein['protein_id'].explode().values
    protein_num = df_protein['protein_id'].apply(len)
    df_protein = df_protein.loc[
        np.repeat(df_protein.index, protein_num)].reset_index(drop=True)
    df_protein['Protein'] = proteins
    df_protein = df_protein[['Protein', 'Protein.Meta']]

    df_protein.set_index('Protein', inplace=True)

    # from 1 vs. 1 to meta vs. meta
    df['Protein.Meta'] = df_protein.loc[df['protein_id']][
        'Protein.Meta'].values
    df['Peptide.Meta'] = df['strip_seq'] # no need to make peptide.meta
    df = df[['Protein.Meta', 'Peptide.Meta', col_cscore_pr]]
    df = df.sort_values(col_cscore_pr, ascending=False)
    df = df.drop_duplicates(
        subset=['Protein.Meta', 'Peptide.Meta'], keep='first'
    ).reset_index(drop=True)

    # graph
    graph = nx.Graph()
    graph.add_nodes_from(df['Protein.Meta'], bipartite=0)
    graph.add_nodes_from(df['Peptide.Meta'], bipartite=1)
    edges = [
        (row['Protein.Meta'], row['Peptide.Meta'], row[col_cscore_pr])
             for _, row in df.iterrows()
    ]
    graph.add_weighted_edges_from(edges)

    # assign
    protein_v, peptide_v = [], []
    subgraphs = list(nx.connected_components(graph))
    for subgraph in subgraphs:
        subgraph = graph.subgraph(subgraph)
        proteins, peptides = greedy_bipartite_vertex_cover(subgraph)
        protein_v.extend(proteins)
        peptide_v.extend(peptides)

    df = pd.DataFrame({'strip_seq': peptide_v, 'protein_group': protein_v})
    pep_num = df['strip_seq'].apply(len).values
    peptide_v = df['strip_seq'].explode().tolist()
    df = df.loc[np.repeat(df.index, pep_num)]
    df = df.reset_index(drop=True)
    df['strip_seq'] = peptide_v

    # result
    df = df_input.merge(df, on='strip_seq', how='left').reset_index(drop=True)
    not_in_range = df['protein_group'].isna()
    df.loc[not_in_range, 'protein_group'] = df.loc[not_in_range, 'protein_id']
    return df
