from libcpp.vector cimport vector
from libcpp cimport bool

cimport cython

# get the classes from the c++ headers

cdef extern from "../ad3/Factor.h" namespace "AD3":
    cdef cppclass BinaryVariable:
        BinaryVariable()
        double GetLogPotential()
        void SetLogPotential(double log_potential)
        int GetId()
        int Degree()

    cdef cppclass Factor:
        Factor()
        vector[double] GetAdditionalLogPotentials()
        void SetAdditionalLogPotentials(vector[double] additional_log_potentials)
        int Degree()
        int GetLinkId(int i)
        BinaryVariable *GetVariable(int i)
        void SolveMAP(vector[double] variable_log_potentials,
                      vector[double] additional_log_potentials,
                      vector[double] *variable_posteriors,
                      vector[double] *additional_posteriors,
                      double *value)


cdef extern from "../ad3/MultiVariable.h" namespace "AD3":
    cdef cppclass MultiVariable:
        int GetNumStates()
        BinaryVariable *GetState(int i)
        double GetLogPotential(int i)
        void SetLogPotential(int i, double log_potential)


cdef extern from "../ad3/FactorGraph.h" namespace "AD3":
    cdef cppclass FactorGraph:
        FactorGraph()
        void SetVerbosity(int verbosity)
        void SetEtaPSDD(double eta)
        void SetMaxIterationsPSDD(int max_iterations)
        int SolveLPMAPWithPSDD(vector[double]* posteriors,
                               vector[double]* additional_posteriors,
                               double* value)
        void SetEtaAD3(double eta)
        void AdaptEtaAD3(bool adapt)
        void SetMaxIterationsAD3(int max_iterations)
        void SetResidualThresholdAD3(double threshold)
        void FixMultiVariablesWithoutFactors()
        int SolveLPMAPWithAD3(vector[double]* posteriors,
                              vector[double]* additional_posteriors,
                              double* value)
        int SolveExactMAPWithAD3(vector[double]* posteriors,
                                 vector[double]* additional_posteriors,
                                 double* value)

        vector[double] GetDualVariables()
        vector[double] GetLocalPrimalVariables()
        vector[double] GetGlobalPrimalVariables()

        BinaryVariable *CreateBinaryVariable()
        MultiVariable *CreateMultiVariable(int num_states)
        Factor *CreateFactorDense(vector[MultiVariable*] multi_variables,
                                  vector[double] additional_log_potentials,
                                  bool owned_by_graph)
        Factor *CreateFactorXOR(vector[BinaryVariable*] variables,
                                vector[bool] negated,
                                bool owned_by_graph)
        Factor *CreateFactorXOROUT(vector[BinaryVariable*] variables,
                                   vector[bool] negated,
                                   bool owned_by_graph)
        Factor *CreateFactorAtMostOne(vector[BinaryVariable*] variables,
                                      vector[bool] negated,
                                      bool owned_by_graph)
        Factor *CreateFactorOR(vector[BinaryVariable*] variables,
                               vector[bool] negated,
                               bool owned_by_graph)
        Factor *CreateFactorOROUT(vector[BinaryVariable*] variables,
                                  vector[bool] negated,
                                  bool owned_by_graph)
        Factor *CreateFactorANDOUT(vector[BinaryVariable*] variables,
                                   vector[bool] negated,
                                   bool owned_by_graph)
        Factor *CreateFactorIMPLY(vector[BinaryVariable*] variables,
                                  vector[bool] negated,
                                  bool owned_by_graph)
        Factor *CreateFactorPAIR(vector[BinaryVariable*] variables,
                                 double edge_log_potential,
                                 bool owned_by_graph)
        Factor *CreateFactorBUDGET(vector[BinaryVariable*] variables,
                                   vector[bool] negated,
                                   int budget,
                                   bool owned_by_graph)
        Factor *CreateFactorKNAPSACK(vector[BinaryVariable*] variables,
                                     vector[bool] negated,
                                     vector[double] costs,
                                     double budget,
                                     bool owned_by_graph)
        void DeclareFactor(Factor *factor,
                           vector[BinaryVariable*] variables,
                           bool owned_by_graph)


cdef extern from "../examples/cpp/dense/FactorSequence.h" namespace "AD3":
    cdef cppclass FactorSequence(Factor):
        FactorSequence()
        void Initialize(vector[int] num_states)


cdef extern from "../examples/cpp/summarization/FactorSequenceCompressor.h" namespace "AD3":
    cdef cppclass FactorSequenceCompressor(Factor):
        FactorSequenceCompressor()
        void Initialize(int length, vector[int] left_positions,
                        vector[int] right_positions)


cdef extern from "../examples/cpp/summarization/FactorCompressionBudget.h" namespace "AD3":
    cdef cppclass FactorCompressionBudget(Factor):
        FactorCompressionBudget()
        void Initialize(int length, int budget,
                        vector[bool] counts_for_budget,
                        vector[int] bigram_positions)


cdef extern from "../examples/cpp/summarization/FactorBinaryTree.h" namespace "AD3":
    cdef cppclass FactorBinaryTree(Factor):
        FactorBinaryTree()
        void Initialize(vector[int] parents)


cdef extern from "../examples/cpp/summarization/FactorBinaryTreeCounts.h" namespace "AD3":
    cdef cppclass FactorBinaryTreeCounts(Factor):
        FactorBinaryTreeCounts()
        void Initialize(vector[int] parents, vector[bool] counts_for_budget)
        void Initialize(vector[int] parents, vector[bool] counts_for_budget,
                        vector[bool] has_count_scores)
        void Initialize(vector[int] parents, vector[bool] counts_for_budget,
                        vector[bool] has_count_scores, int max_num_bins)


cdef extern from "../examples/cpp/summarization/FactorGeneralTree.h" namespace "AD3":
    cdef cppclass FactorGeneralTree(Factor):
        FactorGeneralTree()
        void Initialize(vector[int] parents, vector[int] num_states)


cdef extern from "../examples/cpp/summarization/FactorGeneralTreeCounts.h" namespace "AD3":
    cdef cppclass FactorGeneralTreeCounts(Factor):
        FactorGeneralTreeCounts()
        void Initialize(vector[int] parents, vector[int] num_states)


cdef extern from "../examples/cpp/parsing/FactorTree.h" namespace "AD3":
    cdef cppclass Arc:
        Arc(int, int)

    cdef cppclass FactorTree(Factor):
        FactorTree()
        void Initialize(int, vector[Arc *])
        int RunCLE(vector[double]&, vector[int] *v, double *d)


# wrap them into python extension types
cdef class PBinaryVariable:
    cdef BinaryVariable *thisptr
    cdef bool allocate
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
            self.thisptr = new BinaryVariable()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def get_log_potential(self):
        return self.thisptr.GetLogPotential()

    def set_log_potential(self, double log_potential):
        self.thisptr.SetLogPotential(log_potential)

    def get_id(self):
        return self.thisptr.GetId()

    def get_degree(self):
        return self.thisptr.Degree()


cdef class PMultiVariable:
    cdef MultiVariable *thisptr
    cdef bool allocate

    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
            self.thisptr = new MultiVariable()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    cdef int _get_n_states(self):
        return self.thisptr.GetNumStates()

    def __len__(self):
        return self._get_n_states()

    def get_state(self, int i, bool validate=True):

        if validate and not 0 <= i < self._get_n_states():
            raise IndexError("State {:d} is out of bounds.".format(i))

        cdef BinaryVariable *variable = self.thisptr.GetState(i)
        pvariable = PBinaryVariable(allocate=False)
        pvariable.thisptr = variable
        return pvariable

    def __getitem__(self, int i):

        if not 0 <= i < self._get_n_states():
            raise IndexError("State {:d} is out of bounds.".format(i))

        return self.get_log_potential(i)

    def __setitem__(self, int i, double log_potential):
        if not 0 <= i < len(self):
            raise IndexError("State {:d} is out of bounds.".format(i))
        self.set_log_potential(i, log_potential)

    def get_log_potential(self, int i):
        return self.thisptr.GetLogPotential(i)

    def set_log_potential(self, int i, double log_potential):
        self.thisptr.SetLogPotential(i, log_potential)

    @cython.boundscheck(False)
    def set_log_potentials(self, double[:] log_potentials, bool validate=True):
        cdef Py_ssize_t n_states = self.thisptr.GetNumStates()
        cdef Py_ssize_t i

        if validate and len(log_potentials) != n_states:
            raise IndexError("Expected buffer of length {}".format(n_states))

        for i in range(n_states):
            self.thisptr.SetLogPotential(i, log_potentials[i])


cdef class PFactor:
    cdef Factor* thisptr
    cdef bool allocate
    # This is a virtual class, so don't allocate/deallocate.
    def __cinit__(self):
        self.allocate = False
        pass

    def __dealloc__(self):
        pass

    def set_allocate(self, allocate):
        self.allocate = allocate

    def get_additional_log_potentials(self):
        cdef vector[double] additional_log_potentials
        additional_log_potentials = self.thisptr.GetAdditionalLogPotentials()
        p_additional_log_potentials = []
        cdef size_t i
        for i in xrange(additional_log_potentials.size()):
            p_additional_log_potentials.append(additional_log_potentials[i])
        return p_additional_log_potentials

    def set_additional_log_potentials(self, vector[double] additional_log_potentials):
        self.thisptr.SetAdditionalLogPotentials(additional_log_potentials)

    def get_degree(self):
        return self.thisptr.Degree()

    def get_link_id(self, int i):
        return self.thisptr.GetLinkId(i)

    def get_variable(self, int i):
        cdef BinaryVariable *variable = self.thisptr.GetVariable(i)
        pvariable = PBinaryVariable(allocate=False)
        pvariable.thisptr = variable
        return pvariable

    def solve_map(self, vector[double] variable_log_potentials,
                  vector[double] additional_log_potentials):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        self.thisptr.SolveMAP(variable_log_potentials, additional_log_potentials,
                              &posteriors, &additional_posteriors,
                              &value)
        p_posteriors, p_additional_posteriors = [], []
        cdef size_t i
        for i in range(posteriors.size()):
            p_posteriors.append(posteriors[i])
        for i in range(additional_posteriors.size()):
            p_additional_posteriors.append(additional_posteriors[i])

        return value, p_posteriors, p_additional_posteriors


cdef class PFactorSequence(PFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorSequence()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] num_states):
        (<FactorSequence*>self.thisptr).Initialize(num_states)


cdef class PFactorSequenceCompressor(PFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorSequenceCompressor()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int length, vector[int] left_positions,
                   vector[int] right_positions):
        (<FactorSequenceCompressor*>self.thisptr).Initialize(length,
                                                             left_positions,
                                                             right_positions)


cdef class PFactorCompressionBudget(PFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorCompressionBudget()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int length, int budget,
                   pcounts_for_budget,
                   vector[int] bigram_positions):
        cdef vector[bool] counts_for_budget
        for counts in pcounts_for_budget:
            counts_for_budget.push_back(counts)
        (<FactorCompressionBudget*>self.thisptr).Initialize(length, budget,
                                                            counts_for_budget,
                                                            bigram_positions)


cdef class PFactorBinaryTree(PFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorBinaryTree()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] parents):
        (<FactorBinaryTree*>self.thisptr).Initialize(parents)


cdef class PFactorBinaryTreeCounts(PFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorBinaryTreeCounts()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] parents,
                   pcounts_for_budget,
                   phas_count_scores=None,
                   max_num_bins=None):
        cdef vector[bool] counts_for_budget
        cdef vector[bool] has_count_scores
        for counts in pcounts_for_budget:
            counts_for_budget.push_back(counts)
        if phas_count_scores is not None:
            for has_count in phas_count_scores:
                has_count_scores.push_back(has_count)
            if max_num_bins is not None:
                (<FactorBinaryTreeCounts*>self.thisptr).Initialize(
                    parents, counts_for_budget, has_count_scores, max_num_bins)

            else:
                (<FactorBinaryTreeCounts*>self.thisptr).Initialize(
                    parents, counts_for_budget, has_count_scores)

        else:
            (<FactorBinaryTreeCounts*>self.thisptr).Initialize(
                parents, counts_for_budget)


cdef class PFactorGeneralTree(PFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorGeneralTree()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] parents, vector[int] num_states):
        (<FactorGeneralTree*>self.thisptr).Initialize(parents, num_states)


cdef class PFactorGeneralTreeCounts(PFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorGeneralTreeCounts()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] parents, vector[int] num_states):
        (<FactorGeneralTreeCounts*>self.thisptr).Initialize(parents,
                                                            num_states)


cdef class PFactorTree(PFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorTree()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int length, list arcs):
        cdef vector[Arc *] arcs_v
        cdef int head, modifier

        for arc in arcs:
            head, modifier = arc
            arcs_v.push_back(new Arc(head, modifier))

        (<FactorTree*>self.thisptr).Initialize(length, arcs_v)

        for arcp in arcs_v:
            del arcp


cdef class PFactorGraph:
    cdef FactorGraph *thisptr
    def __cinit__(self):
        self.thisptr = new FactorGraph()

    def __dealloc__(self):
        del self.thisptr

    def set_verbosity(self, int verbosity):
        self.thisptr.SetVerbosity(verbosity)

    def create_binary_variable(self):
        cdef BinaryVariable * variable = self.thisptr.CreateBinaryVariable()
        pvariable = PBinaryVariable(allocate=False)
        pvariable.thisptr = variable
        return pvariable

    def create_multi_variable(self, int num_states):
        cdef MultiVariable * mv = self.thisptr.CreateMultiVariable(num_states)
        pmult = PMultiVariable(allocate=False)
        pmult.thisptr = mv
        return pmult

    def create_factor_logic(self, factor_type, p_variables, p_negated,
                            bool owned_by_graph=True):
        cdef vector[BinaryVariable*] variables
        cdef vector[bool] negated
        for i, var in enumerate(p_variables):
            variables.push_back((<PBinaryVariable>var).thisptr)

            negated.push_back(p_negated[i])
        if factor_type == 'XOR':
            self.thisptr.CreateFactorXOR(variables, negated, owned_by_graph)
        elif factor_type == 'XOROUT':
            self.thisptr.CreateFactorXOROUT(variables, negated, owned_by_graph)
        elif factor_type == 'ATMOSTONE':
            self.thisptr.CreateFactorAtMostOne(variables, negated, owned_by_graph)
        elif factor_type == 'OR':
            self.thisptr.CreateFactorOR(variables, negated, owned_by_graph)
        elif factor_type == 'OROUT':
            self.thisptr.CreateFactorOROUT(variables, negated, owned_by_graph)
        elif factor_type == 'ANDOUT':
            self.thisptr.CreateFactorANDOUT(variables, negated, owned_by_graph)
        elif factor_type == 'IMPLY':
            self.thisptr.CreateFactorIMPLY(variables, negated, owned_by_graph)
        else:
            raise NotImplementedError(
                'Unknown factor type: {}'.format(factor_type))

    def create_factor_pair(self, p_variables, double edge_log_potential,
                           bool owned_by_graph=True):
        cdef vector[BinaryVariable*] variables
        for var in p_variables:
            variables.push_back((<PBinaryVariable>var).thisptr)
        self.thisptr.CreateFactorPAIR(variables, edge_log_potential,
                                      owned_by_graph)

    def create_factor_budget(self, p_variables, p_negated, int budget,
                             bool owned_by_graph=True):
        cdef vector[BinaryVariable*] variables
        cdef vector[bool] negated
        for i, var in enumerate(p_variables):
            variables.push_back((<PBinaryVariable>var).thisptr)
            negated.push_back(p_negated[i])
        self.thisptr.CreateFactorBUDGET(variables, negated, budget,
                                        owned_by_graph)

    def create_factor_knapsack(self, p_variables, p_negated, p_costs,
                               double budget, bool owned_by_graph=True):
        cdef vector[BinaryVariable*] variables
        cdef vector[bool] negated
        cdef vector[double] costs
        for i, var in enumerate(p_variables):
            variables.push_back((<PBinaryVariable>var).thisptr)
            negated.push_back(p_negated[i])
            costs.push_back(p_costs[i])
        self.thisptr.CreateFactorKNAPSACK(variables, negated, costs, budget,
                                          owned_by_graph)

    def create_factor_dense(self,  p_multi_variables,
                            p_additional_log_potentials,
                            bool owned_by_graph=True):
        cdef vector[MultiVariable*] multi_variables
        cdef PMultiVariable blub
        for var in p_multi_variables:
            blub = var
            multi_variables.push_back(<MultiVariable*>blub.thisptr)

        cdef vector[double] additional_log_potentials
        for potential in p_additional_log_potentials:
            additional_log_potentials.push_back(potential)
        self.thisptr.CreateFactorDense(multi_variables,
                                       additional_log_potentials,
                                       owned_by_graph)

    def declare_factor(self, p_factor, p_variables, bool owned_by_graph=False):
        cdef vector[BinaryVariable*] variables
        cdef Factor *factor
        for var in p_variables:
            variables.push_back((<PBinaryVariable>var).thisptr)
        if owned_by_graph:
            p_factor.set_allocate(False)
        factor = (<PFactor>p_factor).thisptr
        self.thisptr.DeclareFactor(factor, variables, owned_by_graph)

    def fix_multi_variables_without_factors(self):
        self.thisptr.FixMultiVariablesWithoutFactors()

    def set_eta_psdd(self, double eta):
        self.thisptr.SetEtaPSDD(eta)

    def set_max_iterations_psdd(self, int max_iterations):
        self.thisptr.SetMaxIterationsPSDD(max_iterations)

    def solve_lp_map_psdd(self):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        self.thisptr.SolveLPMAPWithPSDD(&posteriors, &additional_posteriors,
                                        &value)
        p_posteriors, p_additional_posteriors = [], []
        cdef size_t i
        for i in range(posteriors.size()):
            p_posteriors.append(posteriors[i])
        for i in range(additional_posteriors.size()):
            p_additional_posteriors.append(additional_posteriors[i])

        return value, p_posteriors, p_additional_posteriors

    def set_eta_ad3(self, double eta):
        self.thisptr.SetEtaAD3(eta)

    def adapt_eta_ad3(self, bool adapt):
        self.thisptr.AdaptEtaAD3(adapt)

    def set_max_iterations_ad3(self, int max_iterations):
        self.thisptr.SetMaxIterationsAD3(max_iterations)

    def set_residual_threshold_ad3(self, double threshold):
        self.thisptr.SetResidualThresholdAD3(threshold)

    def solve_lp_map_ad3(self):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        cdef int solver_status
        solver_status = self.thisptr.SolveLPMAPWithAD3(&posteriors,
                                                       &additional_posteriors,
                                                       &value)
        p_posteriors, p_additional_posteriors = [], []
        cdef size_t i
        for i in range(posteriors.size()):
            p_posteriors.append(posteriors[i])
        for i in range(additional_posteriors.size()):
            p_additional_posteriors.append(additional_posteriors[i])

        return value, p_posteriors, p_additional_posteriors, solver_status

    def solve_exact_map_ad3(self):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        cdef int solver_status
        solver_status = self.thisptr.SolveExactMAPWithAD3(&posteriors,
                                                          &additional_posteriors,
                                                          &value)
        p_posteriors, p_additional_posteriors = [], []
        cdef size_t i
        for i in range(posteriors.size()):
            p_posteriors.append(posteriors[i])
        for i in range(additional_posteriors.size()):
            p_additional_posteriors.append(additional_posteriors[i])

        return value, p_posteriors, p_additional_posteriors, solver_status

    def get_dual_variables(self):
        cdef vector[double] dual_variables = self.thisptr.GetDualVariables()
        p_dual_variables = []
        for i in xrange(dual_variables.size()):
            p_dual_variables.append(dual_variables[i])
        return p_dual_variables

    def get_local_primal_variables(self):
        cdef vector[double] local_primal_variables
        local_primal_variables = self.thisptr.GetLocalPrimalVariables()
        p_local_primal_variables = []
        for i in xrange(local_primal_variables.size()):
            p_local_primal_variables.append(local_primal_variables[i])
        return p_local_primal_variables

    def get_global_primal_variables(self):
        cdef vector[double] global_primal_variables
        global_primal_variables = self.thisptr.GetGlobalPrimalVariables()
        p_global_primal_variables = []
        for i in xrange(global_primal_variables.size()):
            p_global_primal_variables.append(global_primal_variables[i])
        return p_global_primal_variables

    def solve(self, eta=0.1, adapt=True, max_iter=1000, tol=1e-6,
              verbose=False, branch_and_bound=False):
        """Solve the MAP inference problem associated with the factor graph.

        Parameters
        ---------

        eta : float, default: 0.1
            Value of the penalty constant. If adapt_eta is true, this is the
            initial penalty, otherwise every iteration will apply this amount
            of penalty.

        adapt_eta : boolean, default: True
            If true, adapt the penalty constant using the strategy in [2].

        max_iter : int, default: 1000
            Maximum number of iterations to perform.

        tol : double, default: 1e-6
            Theshold for the primal and dual residuals in AD3. The algorithm
            ends early when both residuals are below this threshold.

        branch_and_bound : boolean, default: False
            If true, apply a branch-and-bound procedure for obtaining the exact
            MAP (note: this can be slow if the relaxation is "too fractional").

        Returns
        -------

        value : double
            The total score (negative energy) of the solution.

        posteriors : list
            The MAP assignment of each binarized variable in the graph,
            in the order in which they were created. Multi-valued variables
            are represented using a value for each state.  If solution is
            approximate, the values may be fractional.

        additional_posteriors : list
            Additional posteriors for each log-potential in the factors.

        status : string, (integral|fractional|infeasible|unsolved)
            Inference status.
        """


        self.set_eta_ad3(eta)
        self.adapt_eta_ad3(adapt)
        self.set_max_iterations_ad3(max_iter)
        self.set_residual_threshold_ad3(tol)
        self.set_verbosity(verbose)

        if branch_and_bound:
            result = self.solve_exact_map_ad3()
        else:
            result = self.solve_lp_map_ad3()

        value, marginals, edge_marginals, solver_status = result

        solver_string = ["integral", "fractional", "infeasible", "unsolved"]
        return value, marginals, edge_marginals, solver_string[solver_status]
