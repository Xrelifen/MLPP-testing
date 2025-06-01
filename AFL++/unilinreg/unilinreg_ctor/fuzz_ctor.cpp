#include "UniLinReg/UniLinReg.hpp"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
    unsigned char buf[512];
    size_t len = fread(buf, 1, sizeof(buf), stdin);
    if (len < 2) return 0;

    int nx = buf[0] % 8;      // x 向量長度
    int ny = buf[1] % 8;      // y 向量長度
    size_t need = 2 + 8*(nx + ny);
    if (len < need) return 0;

    std::vector<double> x(nx), y(ny);
    size_t off = 2;
    for (int i = 0; i < nx; i++) {
        uint64_t v; memcpy(&v, buf+off, 8); off += 8;
        memcpy(&x[i], &v, 8);
    }
    for (int i = 0; i < ny; i++) {
        uint64_t v; memcpy(&v, buf+off, 8); off += 8;
        memcpy(&y[i], &v, 8);
    }

    try {
        MLPP::UniLinReg model(x, y);
    } catch (...) { }
    return 0;
}
