# Convert Keras Weights to Text File


```
python convert_weights.py

--weights_file
  type: str
  help: path to keras weights file
  required: True

--output_file
  type: str
  help: path to output txt file, if not specified will use path of weights_file with txt extension
  required: False
```

## Format of the txt file
The `h5` is parsed into a `txt` file with the following configuration:
* `TOTAL_NUM_LAYERS`: number of layers (this includes input and activations)
* `SIZE_OF_INPUT`: length of the input vector to the neural network
* `LAYER_TYPE`: one of the following available layers
  * dense
  * dropout
  * batchnormalization
* `BIASES`: bias values of all dense layers
* `WEIGHTS`: weights of all dense layers
* `BATCHNORM_PARAMS`: params of BatchNorm layer in this order
  * beta
  * gamma
  * mean
  * variance

```
<TOTAL_NUM_LAYERS>
input <SIZE_OF_INPUT>
<LAYER_TYPE>  <LAYER_INFO>
.
.
.
<BIASES>
.
.
.
<WEIGHTS>
.
.
.
<BATCHNORM_PARAMS>
.
.
.
```

## Ensure Keras Output Matches Neural Fortran

`python examples/test_network.py`

This will:
1. Create a model in Keras with the following options
  * `--train`
    * [store true] train the model for a few epochs
  * `--batchnorm`
    * [store true] include batchnorm in network
  * `--dropout`
    * [float] how much dropout
  * `--num_dense_layers`
    * [int] number of dense layers in network
  * `--activation`
    * [str] activation type
  * `--model_type`
    * [str] sequential of functional model type

2. Get prediction of that model
3. Convert it to a Neural Fortran network
4. Run same input through Neural Fortran
5. Compare the Keras output to the Neural Fortran
