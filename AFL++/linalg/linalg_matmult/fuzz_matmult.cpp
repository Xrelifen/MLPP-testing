// fuzz_matmult.cpp
#include "LinAlg/LinAlg.hpp"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
    unsigned char buf[512];
    size_t len = fread(buf,1,sizeof(buf),stdin);
    if (len < 3) return 0;
    int n = (buf[0]%3)+1, k = (buf[1]%3)+1, m = (buf[2]%3)+1;
    size_t need = 3 + 8*(n*k + k*m);
    if (len < need) return 0;

    // build A (n×k)
    std::vector<std::vector<double>> A(n, std::vector<double>(k));
    size_t off = 3;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < k; j++) {
        uint64_t v; memcpy(&v, buf + off, 8); off += 8;
        memcpy(&A[i][j], &v, 8);
      }

    // build B (k×m)
    std::vector<std::vector<double>> B(k, std::vector<double>(m));
    for (int i = 0; i < k; i++)
      for (int j = 0; j < m; j++) {
        uint64_t v; memcpy(&v, buf + off, 8); off += 8;
        memcpy(&B[i][j], &v, 8);
      }

    // 先建物件，再呼叫
    MLPP::LinAlg linalg;
    auto C = linalg.matmult(A, B);
    volatile size_t dummy = C.size();
    (void)dummy;

    return 0;
}
