#include "FactorBinarySegmentation.h"

using namespace AD3;

int main(int argc, char **argv) {

    int n = 5;

    Factor *f = new FactorBinarySegmentation;
    static_cast<FactorBinarySegmentation*>(f)->Initialize(n);
}
