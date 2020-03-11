module mod_dropout_layer

  use mod_layer
  use mod_kinds, only: ik, rk

  implicit none

  ! Dropout layer - extends from base layer_type
  !   Implements the dropout algorithm
  type, extends(layer_type) :: Dropout
    ! probability of dropping a node
    real(rk) :: drop_prob

  contains

    procedure, public, pass(self) :: forward => dense_forward
    procedure, public, pass(self) :: backward => dense_backward

  end type Dropout

  interface Dropout
    module procedure :: constructor
  end interface Dropout

contains

  type(Dropout) function constructor(this_size, drop_prob) result(layer)
    ! Dropout class constructor
    !   this_size: size to allocate for current layer
    !   drop_prob: probability of dropping a node

    integer(ik), intent(in) :: this_size
    real(rk), intent(in) :: drop_prob
    allocate(layer % o(this_size))

    ! store layer drop probability
    layer % drop_prob = drop_prob
    ! not in training mode
    layer % training = .FALSE.

  end function constructor


  subroutine dense_forward(self, x)

    class(Dropout), intent(in out) :: self
    real(rk), intent(in) :: x(:)

    if (self % training) then
      ! TODO:
      self % o = x * self % drop_prob
    else
      ! NOT TRAINING: pass output forward
      self % o = x
    end if

  end subroutine dense_forward


  subroutine dense_backward(self, g, lr)

    class(Dropout), intent(in out) :: self
    real(rk), intent(in) :: g(:), lr

    ! TODO: implement backward pass
  end subroutine dense_backward

end module mod_dropout_layer
