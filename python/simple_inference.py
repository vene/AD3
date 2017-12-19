import itertools
import numpy as np

from . import factor_graph as fg


def simple_grid(unaries, pairwise, verbose=1):
    height, width, n_states = unaries.shape

    graph = fg.PFactorGraph()

    multi_variables = []
    for i in range(height):
        multi_variables.append([])
        for j in range(width):
            new_variable = graph.create_multi_variable(n_states)
            for state in range(n_states):
                new_variable.set_log_potential(state, unaries[i, j, state])
            multi_variables[i].append(new_variable)

    for i, j in itertools.product(range(height), range(width)):
        if j > 0:
            # horizontal edge
            edge_variables = [multi_variables[i][j - 1], multi_variables[i][j]]
            graph.create_factor_dense(edge_variables, pairwise.ravel())

        if i > 0:
            # vertical edge
            edge_variables = [multi_variables[i - 1][j], multi_variables[i][j]]
            graph.create_factor_dense(edge_variables, pairwise.ravel())

    value, marginals, edge_marginals, status = graph.solve(verbose=verbose)
    marginals = np.array(marginals).reshape(unaries.shape)
    edge_marginals = np.array(edge_marginals).reshape(-1, n_states ** 2)

    return marginals, edge_marginals, value, status


def general_graph(unaries, edges, edge_weights, verbose=1, n_iterations=1000,
                  eta=.1, exact=False):
    if isinstance(unaries, list):
        return _general_graph_multitype(unaries, edges, edge_weights, verbose,
                                        n_iterations, eta, exact)
    else:
        return _general_graph(unaries, edges, edge_weights, verbose,
                              n_iterations, eta, exact)


def _general_graph(unaries, edges, edge_weights, verbose=1, n_iterations=1000,
                   eta=.1, exact=False):
    # The original code of AD3
    if unaries.shape[1] != edge_weights.shape[1]:
        raise ValueError("incompatible shapes of unaries"
                         " and edge_weights.")
    if edge_weights.shape[1] != edge_weights.shape[2]:
        raise ValueError("Edge weights need to be of shape "
                         "(n_edges, n_states, n_states)!")
    if edge_weights.shape[0] != edges.shape[0]:
        raise ValueError("Number of edge weights different from number of"
                         "edges")

    factor_graph = fg.PFactorGraph()
    n_states = unaries.shape[-1]

    multi_variables = []
    for u in unaries:
        new_variable = factor_graph.create_multi_variable(n_states)
        for state, cost in enumerate(u):
            new_variable.set_log_potential(state, cost)
        multi_variables.append(new_variable)

    for i, e in enumerate(edges):
            edge_variables = [multi_variables[e[0]], multi_variables[e[1]]]
            factor_graph.create_factor_dense(edge_variables,
                                             edge_weights[i].ravel())

    value, marginals, edge_marginals, solver_status = factor_graph.solve(
        eta=eta,
        adapt=True,
        max_iter=n_iterations,
        branch_and_bound=exact,
        verbose=verbose)

    marginals = np.array(marginals).reshape(unaries.shape)

    edge_marginals = np.array(edge_marginals).reshape(-1, n_states ** 2)

    return marginals, edge_marginals, value, solver_status


def _general_graph_multitype(l_unaries, l_edges, l_edge_weights, verbose=1,
                             n_iterations=1000, eta=.1, exact=False):
    """
    General graph in presence of multiple node types
    Developed  for the EU project READ. The READ project has received funding
    from the European Union's Horizon 2020 research and innovation programme
    under grant agreement No 674943.

    See CAp 2017 paper (also at arXiv:1708.07644)

    JL Meunier - Feb 2017
    """
    # number of nodes and of states per type
    l_n_nodes, l_n_states = zip(*[unary.shape for unary in l_unaries])

    # BASIC CHECKING
    assert len(l_unaries)**2 == len(l_edges)
    assert len(l_edges) == len(l_edge_weights)

    # when  making a variable, index of 1st variable given a type
    a_variable_startindex_by_type = np.cumsum([0]+list(l_n_nodes))

    factor_graph = fg.PFactorGraph()
    multi_variables = []

    for n_states, unaries in zip(l_n_states, l_unaries):
        for u in unaries:
            new_variable = factor_graph.create_multi_variable(n_states)
            for state, cost in enumerate(u):
                new_variable.set_log_potential(state, cost)
            multi_variables.append(new_variable)

    i_typ_typ = 0
    for typ_i, n_states_i in enumerate(l_n_states):
        var_start_i = a_variable_startindex_by_type[typ_i]
        for typ_j, n_states_j in enumerate(l_n_states):
            var_start_j = a_variable_startindex_by_type[typ_j]

            edges = l_edges[i_typ_typ]
            edge_weights = l_edge_weights[i_typ_typ]
            i_typ_typ += 1

            for i, e in enumerate(edges):
                edge_variables = [multi_variables[var_start_i+e[0]],
                                  multi_variables[var_start_j+e[1]]]
                factor_graph.create_factor_dense(edge_variables,
                                                 edge_weights[i].ravel())

    value, marginals, edge_marginals, solver_status = factor_graph.solve(
        eta=eta,
        adapt=True,
        max_iter=n_iterations,
        branch_and_bound=exact,
        verbose=verbose)

    # NODE MARGINALS
    marginals = np.asarray(marginals)
    ret_node_marginals = list()
    k = 0
    for (_n_nodes, _n_states) in zip(l_n_nodes, l_n_states):
        # iteration by type
        _n_marg = _n_nodes * _n_states
        ret_node_marginals.append(marginals[k:k+_n_marg].reshape(_n_nodes,
                                                                 _n_states))
        k += _n_marg
    assert k == len(marginals)

    # EDGE MARGINALS
    edge_marginals = np.asarray(edge_marginals)
    ret_edge_marginals = list()
    i_typ_typ = 0
    i_marg_start = 0
    for typ_i, n_states_i in enumerate(l_n_states):
        for typ_j, n_states_j in enumerate(l_n_states):
            edges = l_edges[i_typ_typ]  # pop would modify the list..
            i_typ_typ += 1

            _n_edges = len(edges)
            _n_edge_states = n_states_i * n_states_j
            _n_marg = _n_edges * _n_edge_states

            if _n_edges:
                ret_edge_marginals.append(
                    edge_marginals[i_marg_start:
                                   i_marg_start+_n_marg].reshape((
                                                                _n_edges,
                                                                _n_edge_states)
                                                                 ))
            else:
                ret_edge_marginals.append(np.zeros((0, _n_edge_states)))
            i_marg_start += _n_marg
    assert i_marg_start == len(edge_marginals)

    return ret_node_marginals, ret_edge_marginals, value, solver_status
