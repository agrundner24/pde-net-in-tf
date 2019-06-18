# PDE-Net in TensorFlow or: CNet
My implementation of PDE-Net (https://arxiv.org/pdf/1710.09668.pdf) in Tensorflow

To run it on the GPU: Need tensorflow 1.13.1 + cuDNN 7.5.0 + CUDA 10.0. Other versions might work as well, but we have used this combination.

To run it on the CPU: Maybe it suffices to delete the lines as explained on page 3 of module/GuideForAdjustingTheCode.pdf. Still need tensorflow 1.13.1.

- To test advection_diffusion:                      Run module/main.py.
- To test advection_diffusion with different data:  Change the fifth line in inferring_the_pde.py to: 'import alternative_advection_diffusion/generate_data as gD'
- To test diffusion with non-linear source:         Run module/non-linear_pde/main.py.
- To test burgers equation:                         Run module/burgers_eq/main.py
- To test robustness for burgers:                   Run module/results/testing_robustness/main.py
- To test it on user-defined PDE:                   Read module/GuideForAdjustingTheCode.pdf

Each of these scripts will do 100 tests and produce a results.txt-file with 100 entries of the following type:  
    *Program ran for 117 seconds*   
    *[-0.005465187, 1.9912698, 1.9911757, 0.5203466, -0.0024462892, 0.52137536, 0.0011048209, -0.001643187, 0.001238143, -0.0005990097]*   
    *MSE: 0.00010666*    
=> How long did it take to run this one test. What are the inferred coefficients (for the order see below). What is the mean squared error of the inferred coefficients.

The output stream contains the learned coefficients (the ones corresponding to the lowest order derivatives first) and moment-matrices.
In the linear case the order is as follows:

![equation](https://latex.codecogs.com/gif.latex?u_t%20%3D%20%5Cbegin%7Bpmatrix%7D%20u%20%26u_y%26u_x%26u_%7Byy%7D%26u_%7Bxy%7D%26u_%7Bxx%7D%26%20%5Ccdots%20%5Cend%7Bpmatrix%7D%20%5Ccdot%20%5Ctext%7Bcoef%7D)

Optionally you can change parameters in options-dictionary. Note: The mean square error calculation has to be adjusted when changing max_order.


Bear in mind that the FD-generation of data is not stable for most parameters.

