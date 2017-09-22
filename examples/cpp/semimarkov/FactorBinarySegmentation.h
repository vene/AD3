#ifndef FACTOR_BINARY_SEGMENTATION
#define FACTOR_BINARY_SEGMENTATION

#include "ad3/GenericFactor.h"

namespace AD3 {

    class FactorBinarySegmentation : public GenericFactor {

        protected:
        // double SegmentScore(int j, int i) {

        public:
        FactorBinarySegmentation () {}
        virtual ~FactorBinarySegmentation() { ClearActiveSet(); }

        void Evaluate(const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials,
                      const Configuration configuration,
                      double *value) {
            *value = 0;
        }

        void Maximize(const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials,
                      Configuration &configuration,
                      double *value) {

        }

        void UpdateMarginalsFromConfiguration(
                const Configuration &configuration,
                double weight,
                vector<double> *variable_posteriors,
                vector<double> *additional_posteriors) {
        }

        int CountCommonValues(const Configuration &configuration1,
                              const Configuration &configuration2) {
            return 0;
        }

        bool SameConfiguration(const Configuration &configuration1,
                               const Configuration &configuration2) {
            return false;
        }

        void DeleteConfiguration(Configuration configuration) {
        }

        Configuration CreateConfiguration() {
            vector<int>* config = new vector<int>;
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

            for (int i = 0; i < length_; ++i) {
                for (int j = 0; j < i; ++j) {
                    cout << j << ", " << i << ":" << index_segment_[j][i] << endl;
                }
            }

        }

        private:
        int length_;
        vector<vector<int> > index_segment_;

    };
} // namespace AD3

#endif


