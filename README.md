# ML++

Machine learning is a vast and exiciting discipline, garnering attention from specialists of many fields. Unfortunately, for C++ programmers and enthusiasts, there appears to be a lack of support in the field of machine learning. To fill that void and give C++ a true foothold in the ML sphere, this library was written. The intent with this library is for it to act as a crossroad between low-level developers and machine learning engineers.

<p align="center">
    <img src="https://user-images.githubusercontent.com/78002988/119920911-f3338d00-bf21-11eb-89b3-c84bf7c9f4ac.gif" 
    width = 600 height = 400>
</p>

## Make a Test
- We do AFL test and Google test for this project.
    - AFL test report
    - Google test report
        - LinReg Error:
            - The original two‑step formula that first solves for w and then computes b in the univariate case can easily produce NaNs or numerical errors; you can avoid this by solving the augmented system in one go.
        - LinAlg Error:
            - The LinearIndependenceNoCrash3x2 crash happens because the code uses the number of rows (A.size()) instead of the Gram matrix’s dimension when calling det, which goes out‑of‑bounds whenever rows > columns.
            - The PseudoinverseRectangular failure is due to det not having a special case for 1×1 matrices, so it returns zero, leading to a division by zero and NaNs.
        - Regularization Error:
            - The error was that calling the bare abs() invoked the integer overload (truncating your double weights to int before taking the absolute value), so fractional weights became zero and your Lasso/ElasticNet sums were wrong until you switched to std::abs(double) by including <cmath>.
        - Cost Error:
            - The MSE bug came from writing sum / 2 * size instead of sum / (2 * size) (so you were multiplying instead of dividing by the sample count)
            - The MAE derivative was wrong because it took the sign of y_hat instead of the sign of y_hat - y.
        - Utilities:
            - TP, FP, and TN were never initialized (only FN was), so they started with garbage values and made all of the counts—and thus your precision/recall/accuracy/F1—come out wrong.
        - UniLinReg:
            - Never handled the zero‐variance case in x, your slope formula ended up dividing by zero (for a single point or identical inputs), producing NaN until you added the check to set b₁=0 and b₀=mean(y).
## Citations
Various different materials helped me along the way of creating ML++ testing.
 - [MLPP](https://github.com/novak-99/MLPP)
 - [Google test](https://github.com/google/googletest)
 - [AFL](https://github.com/google/AFL)
