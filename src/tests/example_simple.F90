program example_simple
  use mod_network, only: network_type
  implicit none
  type(network_type) :: net, net2
  real, allocatable :: input(:), output(:)
  integer :: i, n
  net = network_type([3, 5, 2])

  call net % save('my_simple_net.txt')
  call net2 % load('my_simple_net.txt')

  input = [0.2, 0.4, 0.6]
  output = [0.123456, 0.246802]

  do i = 1, 100
    call net2 % train(input, output, eta=1.0)
    print *, 'Iteration: ', i, 'Output:', net2 % output(input)
  end do
  call net2 % save('my_simple_net.txt')

end program example_simple
