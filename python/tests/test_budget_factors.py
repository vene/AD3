import pytest
from numpy.testing import assert_array_equal

from .. import factor_graph as fg


def test_knapsack_wrong_cost_size():
    graph = fg.PFactorGraph()
    n_vars = 50
    variables = [graph.create_binary_variable() for _ in range(n_vars)]
    negated = [False for _ in variables]
    costs = [17]
    budget = 1

    with pytest.raises(IndexError):
        graph.create_factor_knapsack(variables, negated, costs, budget)


def test_budget():
    graph = fg.PFactorGraph()

    potentials = [100, 1, 100, 1, 100]

    for val in potentials:
        var = graph.create_binary_variable()
        var.set_log_potential(val)

    _, assign, _, _ = graph.solve()
    print(assign)
    assert sum(assign) == 5

    budget = 3

    graph = fg.PFactorGraph()

    variables = []
    negated = []
    for val in potentials:
        var = graph.create_binary_variable()
        var.set_log_potential(val)
        variables.append(var)
        negated.append(False)

    graph.create_factor_budget(variables, negated, budget=budget)
    _, assign, _, status = graph.solve()
    assert_array_equal(assign, [1, 0, 1, 0, 1])


def test_knapsack():
    graph = fg.PFactorGraph()

    potentials = [100, 1, 100, 1, 100]
    costs = [3, 5, 5, 5, 2]

    for val in potentials:
        var = graph.create_binary_variable()
        var.set_log_potential(val)

    _, assign, _, _ = graph.solve()
    print(assign)
    assert sum(assign) == 5

    budget = 5

    graph = fg.PFactorGraph()

    variables = []
    negated = []
    for val in potentials:
        var = graph.create_binary_variable()
        var.set_log_potential(val)
        variables.append(var)
        negated.append(False)

    graph.create_factor_knapsack(variables, negated, costs, budget)
    _, assign, _, status = graph.solve(branch_and_bound=True)
    assert_array_equal(assign, [1, 0, 0, 0, 1])
