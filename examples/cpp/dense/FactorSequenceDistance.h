#ifndef FACTOR_SEQUENCE_DISTANCE
#define FACTOR_SEQUENCE_DISTANCE

#include <algorithm>
#include "FactorSequence.h"


/*
 * A simplified sequence factor
 *
 * - All nodes have the same number of states.
 * - Transition matrix is stationary and diagonal-banded:
 *   score[i][j] = a_[j - i]
 *
 *   (In practice we take max(-k, min(j - i, k))
 *    where k is a "range" parameter)
 */
namespace AD3 {
class FactorSequenceDistance : public FactorSequence {
    public:
    void Initialize(int length, int n_states, int range) {
        num_states_.assign(length, n_states);
        offset_states_.resize(length);
        for (int i = 0; i < length; ++i)
            offset_states_[i] = i * n_states;

        /* index_edges[position][prev_state][next_state]
         * indexes into additional_log_potentials_ which is size 2 * range + 1
         */
        index_edges_.resize(length + 1);
        int index;
        for (int i = 0; i <= length; ++i) {
            // If i == 0, the previous state is the start symbol.
            int n_prev_states = (i > 0) ? num_states_[i - 1] : 1;
            // If i == length, the previous state is the final symbol.
            int n_curr_states = (i < length) ? num_states_[i] : 1;
            index_edges_[i].resize(n_prev_states);
            for (int prev = 0; prev < n_prev_states; ++prev) {
                index_edges_[i][prev].resize(n_curr_states);
                for (int curr = 0; curr < n_curr_states; ++curr) {

                    if (i == 0)
                        // at the beginning it's as if the prev state were -1
                        index = curr - (-1);
                    else if (i == length)
                        // at the end it's as if the next state is n + 1
                        index = (n_states + 1) - prev;
                    else
                        // the relative distance gives the index
                        index = curr - prev;

                    // clamp to (-range, range)
                    index = std::max(-range, std::min(index, range));

                    // shift to [0, 2 * range] to have valid array indices
                    index += range;
                    index_edges_[i][prev][curr] = index;

                    /*
                    cout << i << ", " << prev << ", " << curr;
                    cout << " -> " << index;
                    cout << " = " << additional_log_potentials_[index];
                    cout << endl;
                    */
                }
            }
        }
    }
};
}

#endif
