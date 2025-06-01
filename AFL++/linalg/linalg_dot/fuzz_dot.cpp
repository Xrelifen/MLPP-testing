#include "LinAlg/LinAlg.hpp"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
    unsigned char buf[256];
    size_t len = fread(buf,1,sizeof(buf),stdin);
    if (len < 2) return 0;
    int n = (buf[0]%5)+1;
    size_t need = 1 + 8*(2*n);
    if (len < need) return 0;

    std::vector<double> a(n), b(n);
    size_t off=1;
    for(int i=0;i<n;i++){
        uint64_t v; memcpy(&v, buf+off,8); off+=8; memcpy(&a[i],&v,8);
    }
    for(int i=0;i<n;i++){
        uint64_t v; memcpy(&v, buf+off,8); off+=8; memcpy(&b[i],&v,8);
    }

    MLPP::LinAlg linalg;
    volatile double c = linalg.dot(a, b);
    (void)c;
    return 0;
}
