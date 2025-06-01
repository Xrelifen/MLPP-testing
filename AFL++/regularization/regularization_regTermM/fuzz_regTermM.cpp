// term_mat/fuzz_term_mat.cpp
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "Regularization/Reg.hpp"

int main() {
    unsigned char buf[512];
    size_t len = fread(buf,1,sizeof(buf),stdin);
    if (len < 19) return 0;

    int r = buf[0] % 3;
    const char* regs[] = {"Ridge","Lasso","ElasticNet"};
    std::string reg(regs[r]);

    double lambda, alpha;
    memcpy(&lambda, buf+1,8);
    memcpy(&alpha,  buf+9,8);

    int rows = buf[17]%4 +1, cols = buf[18]%4 +1;
    size_t need = 19 + 8*rows*cols;
    if (len < need) return 0;

    std::vector<std::vector<double>> W(rows, std::vector<double>(cols));
    size_t off=19;
    for(int i=0;i<rows;i++)
      for(int j=0;j<cols;j++){
        uint64_t v; memcpy(&v, buf+off,8); off+=8;
        memcpy(&W[i][j], &v, 8);
      }

    // ← 先建立物件，再呼叫成員函式
    MLPP::Reg regObj;
    volatile double t = regObj.regTerm(W, lambda, alpha, reg);
    (void)t;

    return 0;
}
