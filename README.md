# The Fortran-Keras Bridge (FKB)

This two-way bridge connects environments where deep learning resources are plentiful, with those where they are scarce. You can find the paper [here](https://arxiv.org/abs/2004.10652).

```
@article{ott2020fortran,
  title={A Fortran-Keras Deep Learning Bridge for Scientific Computing},
  author={Ott, Jordan and Pritchard, Mike and Best, Natalie and Linstead, Erik and Curcic, Milan and Baldi, Pierre},
  journal={arXiv preprint arXiv:2004.10652},
  year={2020}
}
```

![](https://github.com/scientific-computing/FKB/blob/master/Figures/logo.png?raw=true)

This library allows users to convert models built and trained in Keras to ones usable in Fortran. In order to make this possible FKB implements a neural network library in Fortran. The foundations of which are derived from [Milan Curcic's](https://github.com/modern-fortran/neural-fortran) original work.

## Additions
* An extendable layer type
  * The original library was only capable of a dense layer
    * Forward and backward operations occurred outside the layer (in the network module)
  * Ability to implement arbitrary layers
    * Simply extend the `layer_type` and specify these functions:
      * `forward`
      * `backward`
* Training
  * Backprop takes place inside the extended `layer_type`
  * Ability to training arbitrary cost functions
* Implemented layers
  * Dense
  * Dropout
  * Batch Normalization
* Ensembles
  * Read in a directory of network configs
  * Create a network for each config
  * Run in parallel using `$OMP PARALLEL` directives
  * Average results of all predictions in ensemble
* A two-way bridge between Keras and Fortran
  * Convert model trained in Keras (`h5` file) to Fortran
    * Any of the above layers are allowed
    * Sequential or Functional API
  * Convert Fortran models back to Keras
  * Check out [this](https://github.com/scientific-computing/FKB/tree/master/KerasWeightsProcessing#supported-models) for supported model types

---

## Getting started

Check out an example in the [getting started notebook](https://github.com/scientific-computing/FKB/blob/master/GettingStarted.ipynb)

Get the code:

```
git clone https://github.com/scientific-computing/FKB
```

Dependencies:

* Fortran 2018-compatible compiler
* OpenCoarrays (optional, for parallel execution, gfortran only)
* BLAS, MKL (optional)

### Build
* Tests and examples will be built in the `bin/` directory
* To use a different compiler modify `FC=mpif90 cmake .. -DSERIAL=1`

```
sh build_steps.sh
```

## Examples

### Loading a model trained in Keras

```
python convert_weights.py --weights_file path/to/keras_model.h5 --output_file path/to/model_config.txt
```

This would create the `model_config.txt` file with the following:
```
9                         --> How many total layers (includes input and activations)
input	5                 --> 5 inputs
dense	3                 --> Hidden layer 1 has 3 nodes
leakyrelu	0.3       --> Hidden layer 1 activation LeakyReLU with alpha = 0.3
dense	4                 --> Hidden layer 2 has 4 nodes
leakyrelu	0.3       --> Hidden layer 2 activation LeakyReLU with alpha = 0.3
dense	3                 --> Hidden layer 3 has 3 nodes
leakyrelu	0.3       --> Hidden layer 3 activation LeakyReLU with alpha = 0.3
dense	2                 --> 2 outputs in the last layer
linear	0                 --> Linear activation with no alpha
0.5                       --> Learning rate
<BIASES>
.
.
.
<DENSE LAYER WEIGHTS>
.
.
.
<BATCH NORMALIZATION PARAMETERS>
```

### Creating a network

Architecture descriptions are specified in a config text file:
```
9                         --> How many total layers (includes input and activations)
input	5                 --> 5 inputs
dense	3                 --> Hidden layer 1 has 3 nodes
leakyrelu	0.3       --> Hidden layer 1 activation LeakyReLU with alpha = 0.3
dense	4                 --> Hidden layer 2 has 4 nodes
leakyrelu	0.3       --> Hidden layer 2 activation LeakyReLU with alpha = 0.3
dense	3                 --> Hidden layer 3 has 3 nodes
leakyrelu	0.3       --> Hidden layer 3 activation LeakyReLU with alpha = 0.3
dense	2                 --> 2 outputs in the last layer
linear	0                 --> Linear activation with no alpha
0.5                       --> Learning rate
```

Then the network configuration can be loaded into FORTRAN:
```fortran
use mod_network, only: network_type
type(network_type) :: net

call net % load('model_config.txt')
```


### Ensembles
[mod_ensemble](https://github.com/scientific-computing/FKB/blob/master/src/lib/mod_ensemble.F90) allows ensembles of neural networks to be run in parallel. The `ensemble_type` will read all networks provided in the user specified directory. Calling `average` passes the input through all networks in the ensemble and averages their output. `noise_perturbation` is used to perturb the input to each model with Gaussian noise.

Put the names of the model files in `ensemble_members.txt`:
```
simple_model.txt
simple_model_with_weights.txt
```
Then to run an ensemble:
```
ensemble = ensemble_type('$HOME/Desktop/neural-fortran/ExampleModels/', noise_perturbation)

result1 = ensemble % average(input)
```

You can run the `test_ensembles.F90` file:
```
./test_ensembles $HOME/Desktop/neural-fortran/ExampleModels/
```

### Saving and loading from file

To save a network to a file, do:

```fortran
call net % save('model_config.txt')
```

Loading from file works the same way:

```fortran
call net % load('model_config.txt')
```
