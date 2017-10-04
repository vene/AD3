from libcpp.vector cimport vector
from libcpp cimport bool

cimport cython
from base cimport Factor
from base cimport BinaryVariable
from base cimport MultiVariable


cdef class PBinaryVariable:

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
    # This is a virtual class, so don't allocate/deallocate.
    def __cinit__(self):
        self.allocate = False
        pass

    def __dealloc__(self):
        pass

    def set_allocate(self, allocate):
        self.allocate = allocate

    def get_additional_log_potentials(self):
        return self.thisptr.GetAdditionalLogPotentials()

    def set_additional_log_potentials(self,
                                      vector[double] additional_log_potentials):
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
        self.thisptr.SolveMAP(variable_log_potentials,
                              additional_log_potentials,
                              &posteriors,
                              &additional_posteriors,
                              &value)

        return value, posteriors, additional_posteriors

    def solve_qp(self, vector[double] variable_log_potentials,
                 vector[double] additional_log_potentials,
                 int max_iter=10):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors

        self.thisptr.SolveQP(variable_log_potentials,
                             additional_log_potentials,
                             &posteriors,
                             &additional_posteriors)

        return posteriors, additional_posteriors, None


cdef class PGenericFactor(PFactor):
    """Factor which uses the active set algorithm to solve its QP."""

    cdef cast_configuration(self, Configuration cfg):
        return (<vector[int]*> cfg)[0]

    def solve_qp(self, vector[double] variable_log_potentials,
                 vector[double] additional_log_potentials,
                 int max_iter=10, bool return_array=True):

        cdef:
            GenericFactor* gf
            vector[Configuration] active_set_c
            vector[double] posteriors, additional_posteriors

            # below will be converted to cython arrays
            vector[double] distribution
            vector[double] inverse_A
            vector[double] M, Madd


        gf = <GenericFactor*?> self.thisptr

        gf.SetQPMaxIter(max_iter)

        (posteriors, additional_posteriors, _) = \
            super(PGenericFactor, self).solve_qp(
            variable_log_potentials, additional_log_potentials)

        active_set_c = gf.GetQPActiveSet()
        distribution = gf.GetQPDistribution()
        inverse_A = gf.GetQPInvA()
        gf.GetCorrespondence(&M, &Madd)

        active_set_py = [self.cast_configuration(x) for x in active_set_c]

        solver_data = {
            'active_set': active_set_py,
            'distribution': distribution,
            'inverse_A': inverse_A,
            'M': M,
            'Madd': Madd
        }

        return posteriors, additional_posteriors, solver_data

