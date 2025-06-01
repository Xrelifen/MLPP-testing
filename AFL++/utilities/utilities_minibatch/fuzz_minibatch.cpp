#include "Utilities/Utilities.hpp"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main(){
    unsigned char buf[2];
    size_t len=fread(buf,1,2,stdin);
    if(len<2) return 0;
    int n = buf[0] % 8;      // 0..7 samples
    int b = buf[1] % 8;      // 0..7 batches
    if(b < 1) return 0;
    // 讓每筆樣本維度固定為 1
    std::vector<std::vector<double>> inputSet(n, std::vector<double>(1,0.0));
    auto batches = MLPP::Utilities::createMiniBatches(inputSet, b);
    volatile size_t nb = batches.size();
    (void)nb;
    return 0;
}
