! test_keras.f90

! TO RUN
! ./test_keras $NF_PATH/ExampleModels/simple_model.txt

! this file is used in $NF_PATH/KerasWeightsProcessing/examples/test_network.py
! load a specified network from cmd line arg
! pass simple input through it
! print result

program test_keras
  use mod_kinds, only: ik, rk
  use mod_network, only: network_type

  implicit none

  type(network_type) :: net

  real(rk), allocatable :: result1(:), input(:)
  character(len=100), dimension(:), allocatable :: args

  allocate(args(1))
  call get_command_argument(1,args(1))

  ! load trained network from keras
  call net % load(args(1))

  input = [1, 2, 3, 4, 5]

  ! run test input through network
  result1 = net % output(input)
  print *, result1

end program test_keras
