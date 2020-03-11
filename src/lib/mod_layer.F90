module mod_layer

  use mod_kinds, only: ik, rk
  use mod_activation

  implicit none

  public :: array1d, array2d, db_init, db_co_sum, dw_init, dw_co_sum

  ! define layer type to be extended
  type :: layer_type

    logical :: training                    ! is the network in training mode
    real(rk), allocatable :: i(:)          ! store layer input
    real(rk), allocatable :: b(:)          ! biases
    real(rk), allocatable :: o(:)          ! output
    real(rk), allocatable :: gradient(:)   ! gradients
    real(rk), allocatable :: w(:,:)        ! weights
    real(rk), allocatable :: z(:)          ! arg. to activation function
    real(rk), allocatable :: beta(:)
    real(rk), allocatable :: gama(:)
    real(rk), allocatable :: mean(:)
    real(rk), allocatable :: variance(:)
  contains

    ! all layers must implement a forward and backward subroutine
    ! custom layers will follow this same structure
    procedure, public, pass(self) :: forward => layer_forward
    procedure, public, pass(self) :: backward => layer_backward

  end type layer_type

  type :: array1d
    real(rk), allocatable :: array(:)
  end type array1d

  type :: array2d
    real(rk), allocatable :: array(:,:)
  end type array2d

  interface array1d
    module procedure :: array1d_constructor
  end interface array1d

  interface array2d
    module procedure :: array2d_constructor
  end interface array2d

contains

  subroutine layer_forward(self, x)

    class(layer_type), intent(in out) :: self
    real(rk), intent(in) :: x(:)

    ! Do stuff for forward pass
  end subroutine layer_forward


  subroutine layer_backward(self, g, lr)

    class(layer_type), intent(in out) :: self
    real(rk), intent(in) :: g(:), lr

    ! Do stuff for backward pass
  end subroutine layer_backward


  pure type(array1d) function array1d_constructor(length) result(a)
    ! Overloads the default type constructor.
    integer, intent(in) :: length
    allocate(a % array(length))
    a % array = 0
  end function array1d_constructor


  pure type(array2d) function array2d_constructor(dims) result(a)
    ! Overloads the default type constructor.
    integer, intent(in) :: dims(2)
    allocate(a % array(dims(1), dims(2)))
    a % array = 0
  end function array2d_constructor


  pure subroutine db_init(db, dims)
    ! Initialises biases structure.
    type(array1d), allocatable, intent(in out) :: db(:)
    integer(ik), intent(in) :: dims(:)
    integer :: n, nm
    nm = size(dims)
    allocate(db(nm))
    do n = 1, nm - 1
      db(n) = array1d(dims(n))
    end do
    db(n) = array1d(dims(n))
  end subroutine db_init


  pure subroutine dw_init(dw, dims)
    ! Initialises weights structure.
    type(array2d), allocatable, intent(in out) :: dw(:)
    integer(ik), intent(in) :: dims(:)
    integer :: n, nm
    nm = size(dims)
    allocate(dw(nm))
    do n = 1, nm - 1
      dw(n) = array2d(dims(n:n+1))
    end do
    dw(n) = array2d([dims(n), 1])
  end subroutine dw_init


  subroutine db_co_sum(db)
    ! Performs a collective sum of bias tendencies.
    type(array1d), allocatable, intent(in out) :: db(:)
    integer(ik) :: n
    do n = 2, size(db)
#ifdef CAF
      call co_sum(db(n) % array)
#endif
    end do
  end subroutine db_co_sum


  subroutine dw_co_sum(dw)
    ! Performs a collective sum of weights tendencies.
    type(array2d), allocatable, intent(in out) :: dw(:)
    integer(ik) :: n
    do n = 1, size(dw) - 1
#ifdef CAF
      call co_sum(dw(n) % array)
#endif
    end do
  end subroutine dw_co_sum

end module mod_layer
