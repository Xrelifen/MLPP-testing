// fuzz_gradient.cpp
#include "LinReg/LinReg.hpp"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
    // 固定訓練集
    std::vector<std::vector<double>> trainX = {{1,2},{2,3},{3,4},{4,5}};
    std::vector<double> trainY = {3,5,7,9};
    MLPP::LinReg model(trainX, trainY, "None", 0.0, 0.0);

    // 讀 20 bytes：8+4+4+1+1+2(pad)
    unsigned char buf[20];
    if (fread(buf,1,sizeof(buf),stdin) < 14) return 0;

    // parse 參數
    double lr; memcpy(&lr, buf, 8);
    int epochs = int(*(int*)(buf+8));
    int batch  = int(*(int*)(buf+12));
    bool ui    = buf[16] & 1;
    int choice = buf[17] & 3;  // 0–3

    // 避免負 epochs/batch
    if (epochs < 0) epochs = -epochs;
    if (batch  < 1) batch = 1;

    switch(choice) {
      case 0: model.gradientDescent(lr, epochs, ui); break;
      case 1: model.SGD(lr, epochs, ui);             break;
      case 2: model.MBGD(lr, epochs, batch, ui);     break;
      default: model.NewtonRaphson(lr, epochs, ui);  break;
    }
    return 0;
}