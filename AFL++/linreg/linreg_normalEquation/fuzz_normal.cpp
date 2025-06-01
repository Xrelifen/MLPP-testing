// fuzz_normaleq.cpp
#include "LinReg/LinReg.hpp"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>

int main() {
    // 讀入 raw bytes
    unsigned char buf[512];
    size_t len = fread(buf,1,sizeof(buf),stdin);
    if (len < 2) return 0;

    // 前 2 bytes 決定 n, k
    int n = (buf[0] % 4) + 1;   // 1..4
    int k = (buf[1] % 4) + 1;   // 1..4
    size_t need = 2 + size_t(n)*k*8;
    if (len < need) return 0;

    // 建設矩陣 X
    std::vector<std::vector<double>> X;
    X.reserve(n);
    size_t off = 2;
    for(int i=0; i<n; i++){
      std::vector<double> row;
      row.reserve(k);
      for(int j=0; j<k; j++){
        uint64_t v; memcpy(&v, buf+off, 8);
        double d; memcpy(&d, &v, 8);
        row.push_back(d);
        off += 8;
      }
      X.push_back(row);
    }
    // y vector（全 0）
    std::vector<double> y(n, 0.0);

    // train model
    MLPP::LinReg model(X, y, "None", 0.0, 0.0);

    // 呼叫 normalEquation（可能 throw）
    try {
      model.normalEquation();
    } catch (const std::exception&) {
    } catch (...) {
    }
    return 0;
}
