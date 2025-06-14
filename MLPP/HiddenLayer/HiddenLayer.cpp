//
//  HiddenLayer.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "HiddenLayer.hpp"
#include "Activation/Activation.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Utilities/Utilities.hpp"

#include <iostream>
#include <random>

namespace MLPP {
    HiddenLayer::HiddenLayer(int n_hidden, std::string activation, std::vector<std::vector<double>> input, std::string weightInit, std::string reg, double lambda, double alpha)
    : n_hidden(n_hidden), activation(activation), input(input), weightInit(weightInit), reg(reg), lambda(lambda), alpha(alpha)
    {
        weights = Utilities::weightInitialization(input[0].size(), n_hidden, weightInit);
        bias = Utilities::biasInitialization(n_hidden);

        activation_map["Linear"] = &Activation::linear;
        activationTest_map["Linear"] = &Activation::linear;

        activation_map["Sigmoid"] = &Activation::sigmoid;
        activationTest_map["Sigmoid"] = &Activation::sigmoid;

        activation_map["Swish"] = &Activation::swish;
        activationTest_map["Swish"] = &Activation::swish;

        activation_map["Mish"] = &Activation::mish;
        activationTest_map["Mish"] = &Activation::mish;

        activation_map["SinC"] = &Activation::sinc;
        activationTest_map["SinC"] = &Activation::sinc;

        activation_map["Softplus"] = &Activation::softplus;
        activationTest_map["Softplus"] = &Activation::softplus;

        activation_map["Softsign"] = &Activation::softsign;
        activationTest_map["Softsign"] = &Activation::softsign;

        activation_map["CLogLog"] = &Activation::cloglog;
        activationTest_map["CLogLog"] = &Activation::cloglog;

        activation_map["Logit"] = &Activation::logit;
        activationTest_map["Logit"] = &Activation::logit;

        activation_map["GaussianCDF"] = &Activation::gaussianCDF;
        activationTest_map["GaussianCDF"] = &Activation::gaussianCDF;

        activation_map["RELU"] = &Activation::RELU;
        activationTest_map["RELU"] = &Activation::RELU;

        activation_map["GELU"] = &Activation::GELU;
        activationTest_map["GELU"] = &Activation::GELU;

        activation_map["Sign"] = &Activation::sign;
        activationTest_map["Sign"] = &Activation::sign;

        activation_map["UnitStep"] = &Activation::unitStep;
        activationTest_map["UnitStep"] = &Activation::unitStep;

        activation_map["Sinh"] = &Activation::sinh;
        activationTest_map["Sinh"] = &Activation::sinh;

        activation_map["Cosh"] = &Activation::cosh;
        activationTest_map["Cosh"] = &Activation::cosh;

        activation_map["Tanh"] = &Activation::tanh;
        activationTest_map["Tanh"] = &Activation::tanh;

        activation_map["Csch"] = &Activation::csch;
        activationTest_map["Csch"] = &Activation::csch;   

        activation_map["Sech"] = &Activation::sech;
        activationTest_map["Sech"] = &Activation::sech;  

        activation_map["Coth"] = &Activation::coth;
        activationTest_map["Coth"] = &Activation::coth;  

        activation_map["Arsinh"] = &Activation::arsinh;
        activationTest_map["Arsinh"] = &Activation::arsinh;

        activation_map["Arcosh"] = &Activation::arcosh;
        activationTest_map["Arcosh"] = &Activation::arcosh;

        activation_map["Artanh"] = &Activation::artanh;
        activationTest_map["Artanh"] = &Activation::artanh;

        activation_map["Arcsch"] = &Activation::arcsch;
        activationTest_map["Arcsch"] = &Activation::arcsch;

        activation_map["Arsech"] = &Activation::arsech;
        activationTest_map["Arsech"] = &Activation::arsech;

        activation_map["Arcoth"] = &Activation::arcoth;
        activationTest_map["Arcoth"] = &Activation::arcoth;
    }

    void HiddenLayer::forwardPass(){
        LinAlg alg;
        Activation avn;
        z = alg.mat_vec_add(alg.matmult(input, weights), bias);
        a = (avn.*activation_map[activation])(z, 0);
    }

    void HiddenLayer::Test(std::vector<double> x){
        LinAlg alg;
        Activation avn;
        z_test = alg.addition(alg.mat_vec_mult(alg.transpose(weights), x), bias); 
        a_test = (avn.*activationTest_map[activation])(z_test, 0);
    }
}