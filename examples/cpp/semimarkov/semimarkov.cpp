#include "ad3/FactorGraph.h"
#include "FactorBinarySegmentation.h"

using namespace AD3;

int main(int argc, char **argv) {

    int n = 5;

    vector<double> unaries;
    vector<double> additionals;

    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            if (i == 0 && j == 0)
                unaries.push_back(-.01);
            else if (i == 1 && j == 1)
                unaries.push_back(-.02);
            else if (i == 2 && j == 2)
                unaries.push_back(.4);
            else if (i == 3 && j == 3)
                unaries.push_back(.4);
            else if (i == 4 && j == 4)
                unaries.push_back(-.16);
            else if (i == 3 && j == 2)
                unaries.push_back(1);
            else
                unaries.push_back(0);
        }
    }

    // additionals: score for consecutive positive spans
    for (int j = 1; j < n; ++j)
        for (int i = j; i < n; ++i)
            // additionals.push_back(0);
            // additionals.push_back(.3);
            additionals.push_back(-0.0001);

    Factor *f = new FactorBinarySegmentation;
    FactorBinarySegmentation *fbs =
        static_cast<FactorBinarySegmentation*>(f);

    fbs->Initialize(n);

    Configuration cfg = fbs->CreateConfiguration();
    vector<Segment>* cfg_vec = static_cast<vector<Segment>*>(cfg);
    cfg_vec->push_back(Segment(0, 1, false));
    cfg_vec->push_back(Segment(2, 2, true));
    cfg_vec->push_back(Segment(3, 3, true));
    cfg_vec->push_back(Segment(4, 4, false));

    Configuration cfg2 = fbs->CreateConfiguration();
    cfg_vec = static_cast<vector<Segment>*>(cfg2);
    cfg_vec->push_back(Segment(0, 0, false));
    cfg_vec->push_back(Segment(1, 1, false));
    cfg_vec->push_back(Segment(2, 3, true));
    cfg_vec->push_back(Segment(4, 4, false));

    double value;
    cout << "Evaluating: ";
    fbs->Evaluate(unaries, additionals, cfg, &value);
    cout << value << endl;
    cout << "Evaluating: ";
    fbs->Evaluate(unaries, additionals, cfg2, &value);
    cout << value << endl;

    cout << "Same config with itself: ";
    cout << fbs->SameConfiguration(cfg, cfg) << endl;
    cout << "Same config with another: ";
    cout << fbs->SameConfiguration(cfg, cfg2) << endl;
    cout << "Common values:" << endl;
    cout << fbs->CountCommonValues(cfg, cfg) << endl;
    cout << fbs->CountCommonValues(cfg, cfg2) << endl;
    cout << fbs->CountCommonValues(cfg2, cfg) << endl;


    cout << "Maximizing!" << endl;
    Configuration out = fbs->CreateConfiguration();
    fbs->Maximize(unaries, additionals, out, &value);
    cout << value << endl;
    fbs->Evaluate(unaries, additionals, out, &value);
    cout << value << endl;


    cout << "Solving MAP" << endl;
    vector<double> unary_post, additional_post;
    f->SolveMAP(unaries, additionals, &unary_post, &additional_post, &value);
    for (const auto x: unary_post)
        cout << x << " ";
    cout << endl;
    for (const auto x: additional_post)
        cout << x << " ";
    cout << endl;

    cout << "Solving QP" << endl;
    FactorGraph g;
    vector<BinaryVariable*> binary_vars;

    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i)
            binary_vars.push_back(g.CreateBinaryVariable());

    g.DeclareFactor(f, binary_vars, true);
    fbs->Initialize(n);
    // f->SetAdditionalLogPotentials(additionals);
    f->SolveQP(unaries, additionals, &unary_post, &additional_post);
    for (const auto x: unary_post)
        cout << x << " ";
    cout << endl;
    for (const auto x: additional_post)
        cout << x << " ";
    cout << endl;

    vector<Configuration> active_set = fbs->GetQPActiveSet();
    vector<double> distribution = fbs->GetQPDistribution();
    vector<double> inverse_a = fbs->GetQPInvA();

    int start, end;
    bool tag;
    for (int i = 0; i < distribution.size(); ++i) {
        cout << distribution[i] << ": ";
        cfg_vec = static_cast<vector<Segment>*>(active_set[i]);
        for (auto const& segment: *cfg_vec) {
            tie(start, end, tag) = segment;
            cout << "(" << start << ", " << end << ": " << tag << "), ";
        }
        cout << endl;
    }

}
