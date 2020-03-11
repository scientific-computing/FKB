! test_training.f90

! TO RUN
! ./test_training $NF_PATH/ExampleModels/simple_model_with_weights.txt

! load the simple_model
! train on simple example
!   run for 10 epochs
!   print predictions, targets, and loss
!   loss should decrease to verify backprop works

program test_training
  use mod_kinds, only: ik, rk
  use mod_network, only: network_type

  implicit none

  type(network_type) :: net

  integer :: n
  real(rk), allocatable :: result1(:), input(:), label(:), loss, d_loss(:)
  character(len=100), dimension(:), allocatable :: args

  allocate(args(1))
  call get_command_argument(1,args(1))

  ! load trained network from keras
  call net % load(args(1))

  input = [10, 2, 3, 4, 5]
  label = [1, 2]

  do n = 1, 100
    ! run test input through network
    result1 = net % output(input)
    loss = net % loss(result1, label)

    print *, n, 'Prediction:', result1, 'Truth:', label, 'Loss:', loss
    d_loss = net % backprop(result1, label)
  end do

end program test_training
