#include "Utilities/Utilities.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main(){
    unsigned char buf[2];
    if (fread(buf,1,2,stdin) < 2) return 0;
    int n = buf[0] % 16;      // 0..15
    int t = buf[1] % 8;       // 0..7
    const char* types[] = {
      "XavierNormal","XavierUniform",
      "HeNormal","HeUniform",
      "LeCunNormal","LeCunUniform",
      "Uniform","Other"
    };
    std::string type(types[t]);
    auto w = MLPP::Utilities::weightInitialization(n, type);
    volatile size_t s = w.size();
    (void)s;
    return 0;
}
