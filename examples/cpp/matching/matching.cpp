#include <random>

#include "ad3/FactorGraph.h"
#include "FactorMatching.h"

using namespace AD3;

void solve_qp(vector<double> unaries, int n, int m) {
    vector<double> additionals(0);

    Factor *f = new FactorMatching;
    FactorMatching *fm =
        static_cast<FactorMatching*>(f);
    fm->SetClearCache(false);
    fm->SetVerbosity(100);

    cout << "Solving QP" << endl;
    FactorGraph g;
    vector<BinaryVariable*> binary_vars;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            binary_vars.push_back(g.CreateBinaryVariable());

    g.DeclareFactor(f, binary_vars, true);
    fm->Initialize(n, m);

    vector<double> unary_post, additional_post;
    f->SolveQP(unaries, additionals, &unary_post, &additional_post);


    cout << "Unary posteriors" << endl;
    int k = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cout << unary_post[k] << " ";
            k += 1;
        }
        cout << endl;
    }
    cout << endl;

    vector<Configuration> active_set = fm->GetQPActiveSet();
    vector<double> distribution = fm->GetQPDistribution();
    vector<double> inverse_a = fm->GetQPInvA();
    vector<int>* cfg_vec;

    cout << "Configurations in active set" << endl;
    for (int i = 0; i < distribution.size(); ++i) {
        cout << distribution[i] << ": ";
        cfg_vec = static_cast<vector<int>*>(active_set[i]);
        for (auto const& i: *cfg_vec)
            cout << i << ", ";
        cout << endl;
    }
    cout << endl;

}

int main(int argc, char **argv) {

    random_device rd;
    mt19937 gen(rd());
    gen.seed(43);
    normal_distribution<> nrm;

    int n = 4;
    int m = 3;
    int k = 0;

    vector<double> unaries(n * m);

    cout << std::fixed << std::setprecision(2) << std::setw(3) << std::showpos;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            unaries[k] = nrm(gen);
            cout << unaries[k] << " ";
            ++k;
        }
        cout << endl;
    }

    solve_qp(unaries, n, m);
    cout << "Now, transposing." << endl;

    vector<double> unaries_tr(n * m);
    k = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            unaries_tr[k] = unaries[m * j + i];
            cout << unaries_tr[k] << " ";
            k += 1;
        }
        cout << endl;
    }

    solve_qp(unaries_tr, m, n);

    return 0;
}
