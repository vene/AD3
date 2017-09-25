#ifndef FACTOR_BINARY_SEGMENTATION
#define FACTOR_BINARY_SEGMENTATION

#include <limits>
#include "ad3/GenericFactor.h"

namespace AD3 {

    class Segment {
        public:
        Segment() {}
        Segment(int start, int end, bool tag) : start_(start), end_(end), tag_(tag) {}
        ~Segment() {}

        int start() const {return start_;}
        int end() const {return end_;}
        bool tag() const {return tag_;}
        bool operator==(Segment other) const {
            return (start_ == other.start() &&
                    end_ == other.end() &&
                    tag_ == other.tag());
        }

        private:
        int start_;
        int end_;
        bool tag_;
    };

    class FactorBinarySegmentation : public GenericFactor {

        protected:
        double SegmentScore(int j, int i, bool curr_tag, bool prev_tag,
                            const vector<double> &variable_log_potentials,
                            const vector<double> &additional_log_potentials) {

            if (!curr_tag)
                return 0;

            double res;
            if (j == i)
                res = variable_log_potentials[i];
            else
                res = additional_log_potentials[index_segment_[j][i]];
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
            for (auto const& s: *segments) {
                (*value) += SegmentScore(s.start(), s.end(), s.tag(), prev_tag,
                                         variable_log_potentials,
                                         additional_log_potentials);
                prev_tag = s.tag();
            }
        }

        void Maximize(const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials,
                      Configuration &configuration,
                      double *value) {

            vector<bool> tags = { false, true };
            vector<vector<double> > values(2);
            vector<vector<pair<int, bool> > > backptr(2);

            // recursion initial conditions
            for (auto const& tag: tags) {
                values[tag].resize(length_ + 1);
                backptr[tag].resize(length_ + 1);

                values[tag][0] = 0;
                backptr[tag][0] = make_pair<int, bool>(-1, false);
            }

            double val;

            for (int i = 1; i <= length_; ++i) {
                for (auto const& curr_tag: tags) {
                    values[curr_tag][i] = -std::numeric_limits<double>::infinity();
                    for (int j = 0; j < i; ++j) {
                        for (auto const& prev_tag: tags) {
                            val = values[prev_tag][j] +
                                  SegmentScore(j, i - 1, curr_tag, prev_tag,
                                               variable_log_potentials,
                                               additional_log_potentials);
                            if (val > values[curr_tag][i]) {
                                values[curr_tag][i] = val;
                                backptr[curr_tag][i] = pair<int, bool>(j, prev_tag);
                            }
                        }
                    }
                }
            }
            for (int i = 0; i <= length_; ++i) {
                cout << i << "  ";
                for (auto const& curr_tag: tags)
                    cout << values[curr_tag][i] << " ";
                cout << endl;
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
            int end = length_;
            pair<int, bool> ptr;
            vector<Segment>* cfg_vec = static_cast<vector<Segment>*>(configuration);

            while (end > 0) {
                ptr = backptr[best_tag][end];
                cfg_vec->push_back(Segment(ptr.first, end - 1, best_tag));
                best_tag = ptr.second;
                end = ptr.first;
            }

        }

        void UpdateMarginalsFromConfiguration(
                const Configuration &configuration,
                double weight,
                vector<double> *variable_posteriors,
                vector<double> *additional_posteriors) {

            vector<Segment>* segments = static_cast<vector<Segment>*>(configuration);
            for (auto const& segment: segments) {
                if (segment.tag()) {
                    if (segment.start( )== segment.end())
                        (*variable_posteriors)[segment.start()] += weight;
                    else:
                        (*additional_posteriors)[segment.start()][segment.end()] += weight;
                }
            }
        }

        int CountCommonValues(const Configuration &configuration1,
                              const Configuration &configuration2) {
            const vector<Segment>* segments1 =
                static_cast<vector<Segment>* >(configuration1);
            const vector<Segment>* segments2 =
                static_cast<vector<Segment>* >(configuration2);
            int count = 0;
            int j = 0;
            for (int i = 0; i < segments1->size(); ++i) {
                for (; j < segments2->size(); ++j) {
                    if ((*segments2)[j].start() >= (*segments1)[i].start())
                        break;
                }
                if (j < segments2->size() &&
                        (*segments2)[j] == (*segments1)[i]) {
                    ++count;
                    ++j;
                }
            }
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

        void Initialize(int length) {
            length_ = length;
            int index = 0;
            index_segment_.resize(length_ - 1);
            for (int j = 0; j < length_ - 1; ++j) {
                index_segment_[j].resize(length_ - j - 1);
                for (int i = j + 1; i < length; ++i) {
                    index_segment_[j][i] = index;
                    index += 1;
                }
            }
        }

        private:
        int length_;
        vector<vector<int> > index_segment_;

    };
} // namespace AD3

#endif
