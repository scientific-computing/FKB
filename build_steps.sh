rm -rf build
mkdir build
cd build

FC=gfortran cmake .. -DSERIAL=1
# FC='mpif90 -qopenmp' cmake .. -DSERIAL=1

make

cd CMakeFiles/neural.dir/

mv mod_activation.mod.stamp mod_activation.o
mv mod_io.mod.stamp mod_io.o
mv mod_kinds.mod.stamp mod_kinds.o
mv mod_layer.mod.stamp mod_layer.o
mv mod_mnist.mod.stamp mod_mnist.o
mv mod_network.mod.stamp mod_network.o
mv mod_parallel.mod.stamp mod_parallel.o
mv mod_random.mod.stamp mod_random.o
mv mod_ensemble.mod.stamp mod_ensemble.o
mv mod_dense_layer.mod.stamp mod_dense_layer.o
mv mod_batchnorm_layer.mod.stamp mod_batchnorm_layer.o
mv mod_dropout_layer.mod.stamp mod_dropout_layer.o
