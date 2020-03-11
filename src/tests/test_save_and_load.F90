! test_save_and_load.f90

! TO RUN
! ./test_save_and_load

! build a model using config file
! save the model to file
! load the model back
! assert that predictions match

program test_save_and_load
  use mod_kinds, only: ik, rk
  use mod_network, only: network_type

  implicit none

  type(network_type) :: net1, net2

  real(rk), allocatable :: result1(:), result2(:), input(:)


  ! load network from config file
  call net1 % load('../../ExampleModels/simple_model.txt')
  ! save network to config file
  call net1 % save('../../ExampleModels/simple_model_saved.txt')

  ! reload saved model
  call net2 % load('../../ExampleModels/simple_model_saved.txt')

  input = [1, 2, 3, 4, 5]

  ! run test input through network
  result1 = net1 % output(input)
  result2 = net2 % output(input)

  print *, 'Comparing results equal: ', all(result1 == result2)

end program test_save_and_load
