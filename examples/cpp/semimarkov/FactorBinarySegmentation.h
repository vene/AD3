#ifndef FACTOR_BINARY_SEGMENTATION
#define FACTOR_BINARY_SEGMENTATION

#include <limits>
#include <algorithm>
#include <tuple>
#include "ad3/GenericFactor.h"

using std::tuple;
using std::pair;
using std::tie;

namespace AD3 {

    typedef tuple<int, int, bool> Segment; // tie(start, end, tag)
    typedef pair<int, bool> Backptr;  // j, tag;

    class FactorBinarySegmentation : public GenericFactor {

        protected:

        /*
         * score(j, i, true, false) = unary(j, i)
         *
         * score(j, i, true, true) = unary(j, i) + consecutive_score(j, i)
         *   where consecutive_score(j, i) = 0 if j == 0 or additionals[idx]
         *
         * */
        double SegmentScore(int start, int end, bool curr_tag, bool prev_tag,
                            const vector<double> &variable_log_potentials,
                            const vector<double> &additional_log_potentials) {

            if (!curr_tag)
                return 0;

            int i;

            // cout << "SegmentScore(" << start << ", " << end << ")=";

            i = index_segment_[start][end];
            double res = variable_log_potentials[i];

            if (start > 0 && prev_tag) {
                i = index_segment_prev_[start][end];
                res += additional_log_potentials[i];
            }

            // cout << res << endl;
            return res;
        }

        public:
        FactorBinarySegmentation () {}
        virtual ~FactorBinarySegmentation() { ClearActiveSet(); }

        void Evaluate(const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials,
                      const Configuration configuration,
                      double *value) {
            const vector<Segment>* segments =
                static_cast<vector<Segment>* >(configuration);
            *value = 0;

            bool prev_tag = false;
            bool curr_tag;
            int start, end;

            for (auto const& segment: *segments) {
                tie(start, end, curr_tag) = segment;
                (*value) += SegmentScore(start, end, curr_tag, prev_tag,
                                         variable_log_potentials,
                                         additional_log_potentials);
                prev_tag = curr_tag;
            }
        }

        void Maximize(const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials,
                      Configuration &configuration,
                      double *value) {

            vector<bool> tags = { false, true };
            vector<vector<double> > values(2);
            vector<vector<Backptr> > backptr(2);

            // recursion initial conditions
            for (auto const& tag: tags) {
                values[tag].resize(length_ + 1);
                backptr[tag].resize(length_ + 1);

                values[tag][0] = 0;
                backptr[tag][0] = Backptr(-1, false);
            }

            double val;
            const double INF = std::numeric_limits<double>::infinity();

            for (int i = 1; i <= length_; ++i) {
                for (auto const& curr_tag: tags) {
                    values[curr_tag][i] = -INF;
                    for (int j = 0; j < i; ++j) {
                        for (auto const& prev_tag: tags) {
                            val = values[prev_tag][j] +
                                  SegmentScore(j, i - 1, curr_tag, prev_tag,
                                               variable_log_potentials,
                                               additional_log_potentials);
                            if (val > values[curr_tag][i]) {
                                values[curr_tag][i] = val;
                                backptr[curr_tag][i] = Backptr(j, prev_tag);
                            }
                        }
                    }
                }
            }

            // backtrack: first find best value
            bool best_tag = false;
            *value = -std::numeric_limits<double>::infinity();
            for (auto const& curr_tag: tags) {
                if (values[curr_tag][length_] > *value) {
                    *value = values[curr_tag][length_];
                    best_tag = curr_tag;
                }
            }

            // then, trace back the steps
            int start;
            int end = length_;
            bool prev_best_tag;

            vector<Segment>* segments =
                static_cast<vector<Segment>*>(configuration);

            while (end > 0) {
                tie(start, prev_best_tag) = backptr[best_tag][end];
                segments->insert(segments->begin(),
                                 Segment(start, end - 1, best_tag));
                end = start;
                best_tag = prev_best_tag;
            }


        }

        void UpdateMarginalsFromConfiguration(
                const Configuration &configuration,
                double weight,
                vector<double> *variable_posteriors,
                vector<double> *additional_posteriors) {

            const vector<Segment> *segments =
                static_cast<vector<Segment>*>(configuration);

            int start, end, i;
            bool tag;
            bool prev_tag = false;
            for (auto const& segment: *segments) {
                tie(start, end, tag) = segment;
                if (!tag)
                    continue;

                i = index_segment_[start][end];
                (*variable_posteriors)[i] += weight;

                if (start > 0 && prev_tag){
                    i = index_segment_prev_[start][end];
                    (*additional_posteriors)[i] += weight;
                }

                prev_tag = tag;
            }
        }

        int CountCommonValues(const Configuration &configuration1,
                              const Configuration &configuration2) {
            vector<double> unary, extra;
            unary.assign(length_ * (length_ + 1) / 2, 0);
            extra.assign(length_ * (length_ - 1) / 2, 0);

            UpdateMarginalsFromConfiguration(configuration1, 1, &unary, &extra);
            UpdateMarginalsFromConfiguration(configuration2, 1, &unary, &extra);

            int count = 0;
            for (int val : unary)
                if (val > 1)
                    count += 1;
            return count;
        }

        bool SameConfiguration(const Configuration &configuration1,
                               const Configuration &configuration2) {
            const vector<Segment>* segments1 =
                static_cast<vector<Segment>* >(configuration1);
            const vector<Segment>* segments2 =
                static_cast<vector<Segment>* >(configuration2);

            if (segments1->size() != segments2->size()) return false;

            for (int k = 0; k < segments1->size(); ++k)
                if (!((*segments1)[k] == (*segments2)[k]))
                    return false;

            return true;
        }

        void DeleteConfiguration(Configuration configuration) {
            vector<Segment>* segments =
                static_cast<vector<Segment>* >(configuration);
            delete segments;
        }

        Configuration CreateConfiguration() {
            vector<Segment>* config = new vector<Segment>;
            return static_cast<Configuration>(config);
        }

        /* refactor me with maps? */
        void Initialize(int length) {
            length_ = length;

            // index every span in order
            int index = 0;
            index_segment_.resize(length_);
            for (int start = 0; start < length_; ++start) {
                index_segment_[start].resize(length_);
                for (int end = start; end < length; ++end) {
                    index_segment_[start][end] = index;
                    index += 1;
                }
            }

            // index every non-initial span
            index = 0;
            index_segment_prev_.resize(length_);
            for (int start = 1; start < length_; ++start) {
                index_segment_prev_[start].resize(length_);
                for (int end = start; end < length; ++end) {
                    index_segment_prev_[start][end] = index;
                    index += 1;
                }
            }
        }

        vector<vector<int> > ConfigToVector(Configuration configuration) {
            int start;
            int end;
            bool tag;
            vector<vector<int> > result;
            vector<int> tmp;
            vector<Segment>* segments =
                static_cast<vector<Segment>* >(configuration);

            for (auto const& segment: *segments) {
                tie(start, end, tag) = segment;
                tmp = vector<int>();
                tmp.push_back(start);
                tmp.push_back(end);
                tmp.push_back(tag);
                result.push_back(tmp);
            }
            return result;
        }

        private:
        int length_;
        vector<vector<int> > index_segment_;
        vector<vector<int> > index_segment_prev_;

    };
} // namespace AD3

#endif
