! test_ensembles.f90

! TO RUN
! ./test_ensembles $NF_PATH/ExampleModels/

! load ensemble members in directory
! get prediction
!   each model in the ensemble makes a prediction
!   results are averaged over all members

program test_ensembles
  use mod_kinds, only: ik, rk
  use mod_ensemble, only: ensemble_type

  implicit none

  type(ensemble_type) :: ensemble

  real(rk), allocatable :: result1(:), input(:)
  character(len=100), dimension(:), allocatable :: args

  allocate(args(1))
  call get_command_argument(1,args(1))

  ! build ensemble from members in specified directory
  ensemble = ensemble_type(args(1), 0.0)

  input = [1, 2, 3, 4, 5]

  ! run test input through network
  result1 = ensemble % average(input)
  print *, result1

end program test_ensembles
