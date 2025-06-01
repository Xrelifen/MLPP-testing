// weights_vec/fuzz_regWeights.cpp
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "Regularization/Reg.hpp"

int main(){
  unsigned char buf[512];
  size_t len = fread(buf,1,sizeof(buf),stdin);
  if(len < 17) return 0;

  int r = buf[0] % 2;  // 0 = WeightClipping, 1 = Ridge
  std::string reg = (r==0 ? "WeightClipping" : "Ridge");

  double lambda, alpha;
  memcpy(&lambda, buf+1, 8);
  memcpy(&alpha,  buf+9, 8);

  int n = buf[17] % 8 + 1;
  if(len < size_t(18 + 8*n)) return 0;

  std::vector<double> w(n);
  size_t off = 18;
  for(int i = 0; i < n; i++){
    uint64_t v;
    memcpy(&v, buf+off, 8);
    memcpy(&w[i], &v, 8);
    off += 8;
  }

  // ← 先造物件，再呼叫
  MLPP::Reg regObj;
  auto out = regObj.regWeights(w, lambda, alpha, reg);

  size_t s = out.size();
  (void)s;
  return 0;
}
