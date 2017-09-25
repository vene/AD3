#include "ad3/FactorGraph.h"
#include "FactorBinarySegmentation.h"

using namespace AD3;

int main(int argc, char **argv) {

    int n = 5;

    vector<double> unaries = {-.01, .02, .04, .08, -.16};
    vector<double> additionals;
    for (int j = 0; j < n - 1; ++j) {
        for (int i = j + 1; i < n; ++i) {
            if (i == 3 && j == 2)
                additionals.push_back(.32);
            else
                additionals.push_back(0);
        }
    }

    Factor *f = new FactorBinarySegmentation;
    FactorBinarySegmentation *fbs =
        static_cast<FactorBinarySegmentation*>(f);

    fbs->Initialize(n);

    Configuration cfg = fbs->CreateConfiguration();
    vector<Segment>* cfg_vec = static_cast<vector<Segment>*>(cfg);
    cfg_vec->push_back(Segment(0, 1, false));
    cfg_vec->push_back(Segment(2, 3, true));
    cfg_vec->push_back(Segment(4, 4, true));

    Configuration cfg2 = fbs->CreateConfiguration();
    cfg_vec = static_cast<vector<Segment>*>(cfg2);
    cfg_vec->push_back(Segment(0, 0, false));
    cfg_vec->push_back(Segment(1, 1, false));
    cfg_vec->push_back(Segment(2, 3, true));
    cfg_vec->push_back(Segment(4, 4, false));

    double value;
    fbs->Evaluate(unaries, additionals, cfg, &value);
    cout << value << endl;
    fbs->Evaluate(unaries, additionals, cfg2, &value);
    cout << value << endl;
    cout << fbs->SameConfiguration(cfg, cfg) << endl;
    cout << fbs->SameConfiguration(cfg, cfg2) << endl;
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
    // unary_post.resize(n);
    //additional_post.resize(additionals.size());
    f->SolveMAP(unaries, additionals, &unary_post, &additional_post, &value);
    for (const auto x: unary_post)
        cout << x << " ";
    cout << endl;
    for (const auto x: additional_post)
        cout << x << " ";
    cout << endl;

    cout << "Solving QP" << endl;
    FactorGraph g;
    vector<BinaryVariable*> binary_vars(n);
    for (int i = 0; i < n; ++i)
        binary_vars[i] = g.CreateBinaryVariable();
    g.DeclareFactor(f, binary_vars, true);
    fbs->Initialize(n);
    f->SetAdditionalLogPotentials(additionals);
    f->SolveQP(unaries, additionals, &unary_post, &additional_post);
    for (const auto x: unary_post)
        cout << x << " ";
    cout << endl;
    for (const auto x: additional_post)
        cout << x << " ";
    cout << endl;
}
