// term_vec/fuzz_regTerm.cpp
#include <string>        // ← 一定要先 include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "Regularization/Reg.hpp"

int main() {
    unsigned char buf[512];
    size_t len = fread(buf, 1, sizeof(buf), stdin);
    if (len < 17) return 0;

    // 1) 选择 reg 类型 (0=Ridge,1=Lasso,2=ElasticNet,3=None)
    int r = buf[0] % 4;
    const char* regs[] = {"Ridge", "Lasso", "ElasticNet", "None"};
    std::string regType(regs[r]);

    // 2) 解析 lambda, alpha
    double lambda, alpha;
    memcpy(&lambda, buf + 1, 8);
    memcpy(&alpha,  buf + 9, 8);

    // 3) 向量长度 n
    int n = (buf[17] % 8) + 1;   // 1..8
    if (len < size_t(18 + 8 * n)) return 0;

    std::vector<double> w(n);
    size_t off = 18;
    for (int i = 0; i < n; i++) {
        uint64_t v;
        memcpy(&v, buf + off, 8);
        memcpy(&w[i], &v, 8);
        off += 8;
    }

    // 4) 构造对象并调用成员函数
    MLPP::Reg regObj;
    volatile double t = regObj.regTerm(w, lambda, alpha, regType);
    (void)t;

    return 0;
}
