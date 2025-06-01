#include "Utilities/Utilities.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main(){
    unsigned char buf[3];
    if (fread(buf,1,3,stdin) < 3) return 0;
    int n = buf[0] % 8 + 1;    // 1..8
    int m = buf[1] % 8 + 1;    // 1..8
    int t = buf[2] % 8;        // 0..7
    const char* types[] = {
      "XavierNormal","XavierUniform",
      "HeNormal","HeUniform",
      "LeCunNormal","LeCunUniform",
      "Uniform","Other"
    };
    std::string type(types[t]);
    auto W = MLPP::Utilities::weightInitialization(n,m,type);
    volatile size_t s = W.size() * (W.empty()?0:W[0].size());
    (void)s;
    return 0;
}
