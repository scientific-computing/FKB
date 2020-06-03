module mod_dense_layer

  use mod_layer
  use mod_activation
  use mod_kinds, only: ik, rk
  use mod_random, only: randn

  implicit none

  ! Dense layer - extends from base layer_type
  !   Implements matrix multiplication and an activation function
  type, extends(layer_type) :: Dense
    ! activation function parameter
    real(rk) :: alpha

    ! activation function and derivative of that function
    procedure(activation_function), pointer, nopass :: activation => null()
    procedure(activation_function), pointer, nopass :: activation_prime => null()

  contains

    procedure, public, pass(self) :: forward => dense_forward
    procedure, public, pass(self) :: backward => dense_backward

  end type Dense


  interface Dense
    module procedure :: constructor
  end interface Dense

contains

  type(Dense) function constructor(this_size, next_size, activation, alpha) result(layer)
    ! Layer class constructor
    !   this_size: number of neurons in the layer
    !   next_size: number of neurons in the next layer
    !   activation: type of activation function for the layer
    !   alpha: possible extra parameter for the activation function

    integer(ik), intent(in) :: this_size, next_size
    character(len=*), intent(in) :: activation
    real(rk), intent(in) :: alpha

    allocate(layer % b(next_size))
    allocate(layer % o(next_size))
    allocate(layer % z(next_size))

    layer % alpha = alpha
    layer % z = 0
    layer % w = randn(this_size, next_size) / this_size
    layer % b = 0 ! randn(this_size)

    ! not in training mode
    layer % training = .FALSE.

    ! FOR DEBUG PURPOSES
    ! print *, 'Creating dense layer', this_size, next_size, activation, alpha

    ! assign activation function
    select case(trim(activation))
      case('gaussian')
        layer % activation => gaussian
        layer % activation_prime => gaussian_prime
      case('relu')
        layer % activation => relu
        layer % activation_prime => relu_prime
      case('leakyrelu')
        layer % activation => leaky_relu
        layer % activation_prime => leaky_relu_prime
      case('sigmoid')
        layer % activation => sigmoid
        layer % activation_prime => sigmoid_prime
      case('step')
        layer % activation => step
        layer % activation_prime => step_prime
      case('tanh')
        layer % activation => tanhf
        layer % activation_prime => tanh_prime
      case('linear')
        layer % activation => linear
        layer % activation_prime => linear_prime
      case default
        layer % activation => sigmoid
        layer % activation_prime => sigmoid_prime
    end select

  end function constructor


  subroutine dense_forward(self, x)

    class(Dense), intent(in out) :: self
    real(rk), intent(in) :: x(:)

    self % i = x

    self % z = matmul(transpose(self % w), x) + self % b
    self % o = self % activation(self % z, self % alpha)

  end subroutine dense_forward


  subroutine dense_backward(self, g, lr)

    class(Dense), intent(in out) :: self
    real(rk), intent(in) :: g(:), lr
    real(rk), allocatable :: t(:), dw(:,:), db(:)

    db = self % activation_prime(self % z, self % alpha) * g

    dw = matmul(&
      reshape(self % i, (/size(self % i), 1/)), &
      reshape(db, (/1, size(db)/))&
    )

    self % gradient = matmul(self % w, db)

    ! weight updates
    self % w = self % w - lr * dw
    self % b = self % b - lr * db

  end subroutine dense_backward

end module mod_dense_layer
