module mod_network

  use mod_kinds, only: ik, rk
  use mod_layer, only: array1d, array2d, db_init, dw_init,&
                       db_co_sum, dw_co_sum, layer_type
  use mod_dense_layer, only: Dense
  use mod_dropout_layer, only: Dropout
  use mod_batchnorm_layer, only: BatchNorm
  use mod_parallel, only: tile_indices

  implicit none

  private
  public :: network_type

  type layer_container
    class(layer_type), pointer :: p
  end type

  type :: network_type
    type(layer_container), allocatable :: layers(:)
    ! type(layer_type), allocatable :: layers(:)
    real(rk) :: lr
    integer(ik) :: num_dense_layers, input_size, output_size
    real(rk), allocatable :: layer_info(:)
    character(len=100), allocatable :: layer_names(:)

  contains

    procedure, public, pass(self) :: accuracy
    procedure, public, pass(self) :: backprop
    ! procedure, public, pass(self) :: fwdprop
    procedure, public, pass(self) :: init
    procedure, public, pass(self) :: load
    procedure, public, pass(self) :: loss
    procedure, public, pass(self) :: d_loss
    procedure, public, pass(self) :: output
    procedure, public, pass(self) :: save
    ! procedure, public, pass(self) :: set_activation
    procedure, public, pass(self) :: sync

  end type network_type

  interface network_type
    module procedure :: net_constructor
  endinterface network_type

contains

  type(network_type) function net_constructor(layer_names, layer_info) result(net)
    ! Network class constructor. Size of input array dims indicates the total
    ! number of layers (input + hidden + output), and the value of its elements
    ! corresponds the size of each layer.
    real(rk), intent(in) :: layer_info(:)
    character(len=100), intent(in) :: layer_names(:)

    call net % init(layer_names, layer_info)

    call net % sync(1)

  end function net_constructor

  real(rk) function accuracy(self, x, y)
    ! Given input x and output y, evaluates the position of the
    ! maximum value of the output and returns the number of matches
    ! relative to the size of the dataset.
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: x(:,:), y(:,:)
    integer(ik) :: i, good
    good = 0
    do i = 1, size(x, dim=2)
      if (all(maxloc(self % output(x(:,i))) == maxloc(y(:,i)))) then
        good = good + 1
      end if
    end do
    accuracy = real(good) / size(x, dim=2)
  end function accuracy

  subroutine init(self, layer_names, layer_info)
    ! Allocates and initializes the layers with given dimensions dims.
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: layer_info(:)
    integer(ik), allocatable :: dense_dims(:)
    integer(ik) :: n, i, dense_idx, unique_layers
    character(len=100), intent(in) :: layer_names(:)

    ! default learning rate
    self % lr = 0.001

    dense_idx = 0
    unique_layers = 0
    allocate(dense_dims(size(layer_info)))

    do n = 1, size(layer_names)
      ! count unique layers (i.e. not activations)
      select case(trim(layer_names(n)))
        case('input')
          dense_idx = dense_idx + 1
          ! input size
          self % input_size = layer_info(n)
          ! store the dimensions of the weights
          dense_dims(dense_idx) = layer_info(n)
        case('dense')
          dense_idx = dense_idx + 1
          unique_layers = unique_layers + 1
          ! store the dimensions of the weights
          dense_dims(dense_idx) = layer_info(n)
          ! output size
          self % output_size = layer_info(n)
        case('dropout')
          unique_layers = unique_layers + 1
        case('batchnormalization')
          unique_layers = unique_layers + 1
      end select
    end do

    ! allocate the number of unique layers (i.e. not activations)
    if (.not. allocated(self % layers)) allocate(self % layers(unique_layers))

    dense_idx = 0
    unique_layers = 0

    do n = 1, size(layer_names)
      select case(trim(layer_names(n)))
        case('input')
          ! dimension of input is first dimension of first dense layer
          dense_idx = dense_idx + 1
        case('dense')
          dense_idx = dense_idx + 1
          unique_layers = unique_layers + 1

          allocate(&
            self % layers(unique_layers) % p,&
            source=Dense(&
              dense_dims(dense_idx - 1), dense_dims(dense_idx),&                  ! shape of dense layer
              layer_names(n + 1), layer_info(n + 1))&                             ! activation function args
          )
        case('dropout')
          unique_layers = unique_layers + 1

          allocate(&
            self % layers(unique_layers) % p,&
            source=Dropout(dense_dims(dense_idx), layer_info(n))&                 ! layer dim & drop probability
          )
        case('batchnormalization')
          unique_layers = unique_layers + 1

          allocate(&
            self % layers(unique_layers) % p,&
            source=BatchNorm(dense_dims(dense_idx))&                              ! layer dim
          )
      end select

    end do

    ! store info as part of network
    self % layer_info = layer_info
    self % layer_names = layer_names
    self % num_dense_layers = dense_idx

  end subroutine init

  subroutine load(self, filename)
    ! Loads the network from file.
    integer(ik) :: fileunit, n, num_layers, end_of_file
    real(rk), allocatable :: layer_info(:)
    character(len=*), intent(in) :: filename
    class(network_type), intent(in out) :: self
    character(len=100), allocatable :: layer_names(:)

    open(newunit=fileunit, file=filename, status='old', action='read')

    ! number of layers in network; this includes input and activations
    read(fileunit, fmt=*, IOSTAT=end_of_file) num_layers

    ! allocate storage
    allocate(layer_names(num_layers))
    allocate(layer_info(num_layers))

    ! read through the network description
    do n = 1, num_layers
      read(fileunit, fmt=*, IOSTAT=end_of_file) layer_names(n), layer_info(n)
    end do

    ! initialize the network
    call self % init(layer_names, layer_info)

    ! reading learning rate
    read(fileunit, fmt=*, IOSTAT=end_of_file) self % lr

    ! read biases into dense layer
    do n = 1, size(self % layers)
      select type (layer => self % layers(n) % p)
        class is (Dense)
          read(fileunit, fmt=*, IOSTAT=end_of_file) self % layers(n) % p % b

          if (end_of_file < 0) then
            exit
          end if
      end select
    end do

    if (end_of_file > -1) then
      ! read weights into dense layer
      do n = 1, size(self % layers)
        select type (layer => self % layers(n) % p)
          class is (Dense)
            read(fileunit, fmt=*) self % layers(n) % p % w
        end select
      end do

      ! read batchnorm params into layer
      do n = 1, size(self % layers)
        select type (layer => self % layers(n) % p)
          class is (BatchNorm)
            read(fileunit, fmt=*) self % layers(n) % p % beta
            read(fileunit, fmt=*) self % layers(n) % p % gama
            read(fileunit, fmt=*) self % layers(n) % p % mean
            read(fileunit, fmt=*) self % layers(n) % p % variance
        end select
      end do
    end if

    close(fileunit)

  end subroutine load

  real(rk) function loss(self, y_true, y_pred)
    ! Given input x and expected output y, returns the loss of the network.
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: y_true(:), y_pred(:)
    loss = 0.5 * sum((y_true - y_pred)**2) / size(y_true)
  end function loss

  function d_loss(self, y_true, y_pred) result(loss)
    ! Given input x and expected output y, returns the loss of the network.
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: y_true(:), y_pred(:)
    real(rk), allocatable :: loss(:)

    loss = (y_true - y_pred) / size(y_true)
  end function d_loss

  function output(self, input) result(a)
    ! Use forward propagation to compute the output of the network.
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: input(:)
    real(rk), allocatable :: a(:)
    integer(ik) :: n

    associate(layers => self % layers)
      ! pass input to first layer
      call layers(1) % p % forward(input)

      ! iterate through layers passing activation forward
      do n = 2, size(layers)
        call layers(n) % p % forward(layers(n-1) % p % o)
      end do

      ! get activation from last layer
      a = layers(size(layers)) % p % o
    end associate

  end function output

  function backprop(self, y_true, y_pred) result(loss)
    ! Applies a backward propagation through the network
    ! and returns the weight and bias gradients.
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: y_true(:), y_pred(:)
    real(rk), allocatable :: loss(:)
    integer :: n

    ! calculate loss
    loss = self % d_loss(y_true, y_pred)

    n = size(self % layers)
    call self % layers(n) % p % backward(loss, self % lr)

    do n = size(self % layers) - 1, 1, -1
      call self % layers(n) % p % backward(&
        self % layers(n+1) % p % gradient,&
        self % lr)
    end do

  end function backprop

  subroutine save(self, filename)
    ! Saves the network to a file.
    class(network_type), intent(in out) :: self
    character(len=*), intent(in) :: filename
    integer(ik) :: fileunit, n

    open(newunit=fileunit, file=filename)
    ! total number of operations including activations
    write(fileunit, fmt=*) size(self % layer_info)

    do n = 1, size(self % layer_names)
      ! layer name \t info
      write(fileunit, fmt=*) self % layer_names(n), self % layer_info(n)
    end do

    ! writing learning rate
    write(fileunit, fmt=*) self % lr

    do n = 1, size(self % layers)
      select type (layer => self % layers(n) % p)
        class is (Dense)
          ! write biases of dense layer
          write(fileunit, fmt=*) self % layers(n) % p % b
      end select
    end do

    do n = 1, size(self % layers)
      select type (layer => self % layers(n) % p)
        class is (Dense)
          ! write weights of dense layer
          write(fileunit, fmt=*) self % layers(n) % p % w
      end select
    end do

    do n = 1, size(self % layers)
      select type (layer => self % layers(n) % p)
        class is (BatchNorm)
          ! write params of batchnorm layer
          write(fileunit, fmt=*) self % layers(n) % p % beta
          write(fileunit, fmt=*) self % layers(n) % p % gama
          write(fileunit, fmt=*) self % layers(n) % p % mean
          write(fileunit, fmt=*) self % layers(n) % p % variance
      end select
    end do

    close(fileunit)

  end subroutine save


  subroutine sync(self, image)
    ! Broadcasts network weights and biases from
    ! specified image to all others.
    class(network_type), intent(in out) :: self
    integer(ik), intent(in) :: image
    integer(ik) :: n
    if (num_images() == 1) return
    layers: do n = 1, size(self % layers) ! changed from dims
#ifdef CAF
      call co_broadcast(self % layers(n) % p % b, image)
      call co_broadcast(self % layers(n) % p % w, image)
#endif
    end do layers
  end subroutine sync

end module mod_network
