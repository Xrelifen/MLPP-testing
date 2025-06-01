// fuzz_inverse.cpp
#include "LinAlg/LinAlg.hpp"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
    unsigned char buf[256];
    size_t len = fread(buf,1,sizeof(buf),stdin);
    if (len < 1) return 0;
    int d = (buf[0] % 4) + 1;
    size_t need = 1 + 8 * d * d;
    if (len < need) return 0;

    // 构造矩阵 A
    std::vector<std::vector<double>> A(d, std::vector<double>(d));
    size_t off = 1;
    for (int i = 0; i < d; i++) {
      for (int j = 0; j < d; j++) {
        uint64_t v;
        memcpy(&v, buf + off, 8);
        off += 8;
        memcpy(&A[i][j], &v, 8);
      }
    }

    MLPP::LinAlg linalg;
    try {
        // 不要把 B 定义成 volatile
        auto B = linalg.inverse(A);
        // 如果想防止编译器优化掉 B 的使用，可以这样：
        volatile size_t dummy = B.size();
        (void)dummy;
    } catch (...) {
    }
    return 0;
}
