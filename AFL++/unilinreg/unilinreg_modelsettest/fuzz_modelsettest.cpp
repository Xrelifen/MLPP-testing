#include "UniLinReg/UniLinReg.hpp"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
    // 固定訓練資料
    std::vector<double> x0{1,2,3}, y{2,4,6};
    MLPP::UniLinReg model(x0, y);

    unsigned char buf[256];
    size_t len=fread(buf,1,sizeof(buf),stdin);
    if(len<1) return 0;
    int n = buf[0] % 8;
    if(len < 1 + 8*n) return 0;

    std::vector<double> xin(n);
    size_t off=1;
    for(int i=0;i<n;i++){
      uint64_t v; memcpy(&v, buf+off,8); off+=8;
      memcpy(&xin[i], &v, 8);
    }
    auto out = model.modelSetTest(xin);
    volatile size_t s = out.size();
    (void)s;
    return 0;
}
