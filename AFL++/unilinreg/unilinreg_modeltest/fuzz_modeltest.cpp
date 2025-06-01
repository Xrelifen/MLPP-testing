#include "UniLinReg/UniLinReg.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

int main() {
    // 固定訓練：x={1,2,3}, y={2,4,6}
    std::vector<double> x{1,2,3}, y{2,4,6};
    MLPP::UniLinReg model(x, y);

    unsigned char buf[8];
    if (fread(buf,1,8,stdin) < 8) return 0;
    uint64_t v; memcpy(&v, buf, 8);
    double inp; memcpy(&inp, &v, 8);

    volatile double out = model.modelTest(inp);
    (void)out;
    return 0;
}
