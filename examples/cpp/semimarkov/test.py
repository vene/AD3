from ad3 import factor_graph as fg

n = 5

unaries_dict = {
    (0, 0): -0.01,
    (1, 1): -0.02,
    (2, 2): 0.4,
    (3, 3): 0.4,
    (4, 4): -0.16,

    (2, 3): 1
}
unaries = [unaries_dict.get((start, end), 0)
           for start in range(n)
           for end in range(start, n)]

transition = [-0.0001
              for start in range(1, n)
              for end in range(start, n)]

g = fg.PFactorGraph()
binary_vars = [g.create_binary_variable()
               for start in range(n)
               for end in range(start, n)]
fbs = fg.PFactorBinarySegmentation()
g.declare_factor(fbs, binary_vars)
fbs.initialize(n)
fbs.set_additional_log_potentials(transition)

u, add, aset, vbar, invA = fbs.solve_qp(unaries, transition)
print(u)
print(add)
print(vbar)
print(invA)


print("---")
print(aset)
