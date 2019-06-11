# pde-net-in-tf
My implementation of PDE-Net in Tensorflow

It suffices to run main.py and compare the results to the generated data

The output contains the learned coefficients (the ones corresponding to the lowest order derivatives first) and moment-matrices.
In the linear case, this is as follows:

![equation](https://latex.codecogs.com/gif.latex?u_t%20%3D%20%5Ctext%7Bcoef%7D%5Ccdot%20%5Cbegin%7Bpmatrix%7D%20u%20%5C%5Cu_y%5C%5Cu_x%5C%5Cu_%7Byy%7D%5C%5Cu_%7Bxy%7D%5C%5Cu_%7Bxx%7D%5C%5C%20%5Cvdots%20%5Cend%7Bpmatrix%7D)

As for examples, simply change the fifth line in inferring_the_pde.py to one of the following:
- import cfd_python.advection_diffusion.generate_data as gD
- import cfd_python.diffusion.generate_data as gD
- import cfd_python.linear_convection.generate_data as gD

Note that the FD-generation of data is not stable for most parameters.
