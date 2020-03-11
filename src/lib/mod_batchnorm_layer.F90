module mod_batchnorm_layer

  use mod_layer
  use mod_kinds, only: ik, rk

  implicit none

  ! BatchNorm layer - extends from base layer_type
  !   Implements batch normalization
  type, extends(layer_type) :: BatchNorm
    ! epsilon parameter
    real(rk) :: epsilon

  contains

    procedure, public, pass(self) :: forward => batchnorm_forward
    procedure, public, pass(self) :: backward => batchnorm_backward

  end type BatchNorm

  interface BatchNorm
    module procedure :: constructor
  end interface BatchNorm

contains

  type(BatchNorm) function constructor(this_size) result(layer)
    ! BatchNorm class constructor
    !   this_size: size to allocate for current layer

    integer(ik), intent(in) :: this_size

    allocate(layer % o(this_size))
    allocate(layer % beta(this_size))
    allocate(layer % gama(this_size))
    allocate(layer % mean(this_size))
    allocate(layer % variance(this_size))

    ! not in training mode
    layer % training = .FALSE.

    ! epsilon default to 0.001
    layer % epsilon = 0.001

  end function constructor


  subroutine batchnorm_forward(self, x)

    class(BatchNorm), intent(in out) :: self
    real(rk), intent(in) :: x(:)

    if (self % training) then
      ! TODO:
      self % o = x
    else
      ! NOT TRAINING: standardize using learned values
      self % o = ((x - self % mean) / sqrt(self % variance + self % epsilon)) * self % gama + self % beta
    end if

  end subroutine batchnorm_forward


  subroutine batchnorm_backward(self, g, lr)

    class(BatchNorm), intent(in out) :: self
    real(rk), intent(in) :: g(:), lr

    ! TODO: implement backward pass
  end subroutine batchnorm_backward

end module mod_batchnorm_layer
