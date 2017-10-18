from libcpp.vector cimport vector
from libcpp cimport bool

cimport cython

from base cimport Factor
from base cimport BinaryVariable
from base cimport MultiVariable
from base cimport FactorGraph
from base cimport Configuration
from base cimport PBinaryVariable, PMultiVariable, PFactor, PGenericFactor


cdef extern from "../examples/cpp/dense/FactorSequence.h" namespace "AD3":
    cdef cppclass FactorSequence(Factor):
        FactorSequence()
        void Initialize(vector[int] num_states)


cdef extern from "../examples/cpp/dense/FactorSequenceDistance.h" namespace "AD3":
    cdef cppclass FactorSequenceDistance(Factor):
        FactorSequenceDistance()
        void Initialize(int n_nodes, int n_states, int bandwidth)


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


cdef extern from "../examples/cpp/parsing/FactorHeadAutomaton.h" namespace "AD3":
    cdef cppclass Sibling:
        Sibling(int, int, int)

    cdef cppclass FactorHeadAutomaton(Factor):
        FactorHeadAutomaton()
        void Initialize(int, vector[Sibling *])


cdef extern from "../examples/cpp/semimarkov/FactorBinarySegmentation.h" namespace "AD3":
    cdef struct Segment:
        int start
        int end
        bool tag

    cdef cppclass FactorBinarySegmentation(Factor):
        FactorBinarySegmentation()
        void Initialize(int)


cdef extern from "../examples/cpp/matching/FactorMatching.h" namespace "AD3":
    cdef cppclass FactorMatching(Factor):
        FactorMatching()
        void Initialize(int, int)



cdef class PFactorSequence(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorSequence()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] num_states):
        (<FactorSequence*>self.thisptr).Initialize(num_states)


cdef class PFactorSequenceDistance(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorSequenceDistance()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int n_nodes, int n_states, int bandwidth):
        (<FactorSequenceDistance*>self.thisptr).Initialize(n_nodes,
                                                           n_states,
                                                           bandwidth)


cdef class PFactorSequenceCompressor(PGenericFactor):
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


cdef class PFactorCompressionBudget(PGenericFactor):
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


cdef class PFactorBinaryTree(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorBinaryTree()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] parents):
        (<FactorBinaryTree*>self.thisptr).Initialize(parents)


cdef class PFactorBinaryTreeCounts(PGenericFactor):
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


cdef class PFactorGeneralTree(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorGeneralTree()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] parents, vector[int] num_states):
        (<FactorGeneralTree*>self.thisptr).Initialize(parents, num_states)


cdef class PFactorGeneralTreeCounts(PGenericFactor):
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


cdef class PFactorTree(PGenericFactor):
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

        cdef tuple arc
        for arc in arcs:
            head = arc[0]
            modifier = arc[1]

            if not 0 <= head < length:
                raise ValueError("Invalid arc: head must be in [0, length)")
            if not 1 <= modifier < length:
                raise ValueError("Invalid arc: modifier must be in [1, length)")
            if not head != modifier:
                raise ValueError("Invalid arc: head cannot be the same as the "
                                 " modifier")
            arcs_v.push_back(new Arc(head, modifier))

        # if arcs_v.size() != <Py_ssize_t> self.thisptr.Degree():
        #    raise ValueError("Number of arcs differs from number of bound "
        #                     "variables.")
        (<FactorTree*>self.thisptr).Initialize(length, arcs_v)

        for arcp in arcs_v:
            del arcp


cdef class PFactorBinarySegmentation(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorBinarySegmentation()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int length):
        (<FactorBinarySegmentation*>self.thisptr).Initialize(length)

    cdef cast_configuration(self, Configuration cfg):
        # cdef Segment segment
        cdef vector[Segment] segments = (<vector[Segment]*> cfg)[0]
        res = [(segment.start, segment.end, segment.tag)
               for segment in segments]

        return res


cdef class PFactorMatching(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
            self.thisptr = new FactorMatching()

    def __dealloc(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int rows, int cols):
        (<FactorMatching*>self.thisptr).Initialize(rows, cols)


cdef class PFactorHeadAutomaton(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorHeadAutomaton()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int length, list siblings):
        # length = max(s - h) for (h, m, s) in siblings I think

        cdef vector[Sibling *] siblings_v

        cdef tuple sibling
        for sibling in siblings:
            siblings_v.push_back(new Sibling(sibling[0],
                                             sibling[1],
                                             sibling[2]))

        if siblings_v.size() != length * (1 + length) / 2:
            raise ValueError("wrong length?")

        if length != self.thisptr.Degree() + 1:
            raise ValueError("Number of variables doesn't match")

        (<FactorHeadAutomaton*>self.thisptr).Initialize(length, siblings_v)

        for sibp in siblings_v:
            del sibp
