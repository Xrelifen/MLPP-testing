#include "LinReg/LinReg.hpp"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
    // 1) 固定小規模訓練資料，初始化一支 LinReg model
    std::vector<std::vector<double>> trainX = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}
    };
    std::vector<double> trainY = {3.0, 5.0, 7.0, 9.0};
    MLPP::LinReg model(trainX, trainY, "None", 0.0, 0.0);

    // 2) 從 stdin 讀取最多 64 bytes，作為 candidate 特徵
    unsigned char buf[64];
    size_t len = fread(buf, 1, sizeof(buf), stdin);
    if (len < 8) {
        // 太短無法構成至少一個 double 就跳過
        return 0;
    }

    // 3) 將每 8 bytes 轉成 double，存到 vector<double>
    std::vector<double> x;
    for (size_t i = 0; i + 7 < len; i += 8) {
        uint64_t v = 0;
        memcpy(&v, buf + i, 8);
        double d;
        memcpy(&d, &v, 8);
        x.push_back(d);
    }

    // 4) 呼叫要 fuzz 的 API：modelTest()
    //    可以測試不同向量長度、NaN、overflow…等 edge case
    volatile double y = model.modelTest(x);  // :contentReference[oaicite:0]{index=0}
    (void)y;

    return 0;
}
