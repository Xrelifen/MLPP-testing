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
## Citations
Various different materials helped me along the way of creating ML++ testing.
 - (MLPP)[https://github.com/novak-99/MLPP]
 - (Google test)[https://github.com/google/googletest]
 - (AFL)[https://github.com/google/AFL]
