# matching (linear assignment)

import numpy as np
from ad3 import factor_graph as fg

rows = 5
cols = 3

weights = np.random.RandomState(0).randn(rows, cols)

g = fg.PFactorGraph()
binary_vars = []

for i in range(rows):
    for j in range(cols):
        var = g.create_binary_variable()
        var.set_log_potential(weights[i, j])
        binary_vars.append(var)

fm = fg.PFactorMatching()
g.declare_factor(fm, binary_vars)
fm.initialize(rows, cols)

val, posteriors, _, _ = g.solve()

assignment = np.array(posteriors).reshape(rows, cols)

print("Best assignment score:", val)
print("Best assignment:\n", assignment)
print("Sanity check score:", np.dot(assignment.ravel(), weights.ravel()))
