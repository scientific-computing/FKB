module mod_activation

  ! A collection of activation functions and their derivatives.

  use mod_kinds, only: ik, rk

  implicit none

  private

  public :: activation_function
  public :: gaussian, gaussian_prime
  public :: relu, relu_prime
  public :: sigmoid, sigmoid_prime
  public :: step, step_prime
  public :: tanhf, tanh_prime
  public :: linear, linear_prime
  public :: leaky_relu, leaky_relu_prime

  interface
    pure function activation_function(x, alpha)
      import :: rk
      real(rk), intent(in) :: x(:)
      real(rk), intent(in) :: alpha
      real(rk) :: activation_function(size(x))
    end function activation_function
  end interface

contains

  pure function gaussian(x, alpha) result(res)
    ! Gaussian activation function.
    real(rk), intent(in) :: x(:)
    real(rk), intent(in) :: alpha
    real(rk) :: res(size(x))
    res = exp(-x**2)
  end function gaussian

  pure function gaussian_prime(x, alpha) result(res)
    ! First derivative of the Gaussian activation function.
    real(rk), intent(in) :: x(:)
    real(rk), intent(in) :: alpha
    real(rk) :: tmp_alpha
    real(rk) :: res(size(x))

    tmp_alpha = 0.0
    res = -2 * x * gaussian(x, tmp_alpha)
  end function gaussian_prime

  pure function leaky_relu(x, alpha) result(res)
    !! Leaky REctified Linear Unit (RELU) activation function.
    real(rk), intent(in) :: alpha
    real(rk), intent(in) :: x(:)
    real(rk) :: res(size(x))
    res = max(alpha * x, x)
  end function leaky_relu

  pure function leaky_relu_prime(x, alpha) result(res)
    ! First derivative of the REctified Linear Unit (RELU) activation function.
    real(rk), intent(in) :: x(:)
    real(rk), intent(in) :: alpha
    real(rk) :: res(size(x))
    where (0.3 * x > 0)
      res = 1
    elsewhere
      res = 0
    end where
  end function leaky_relu_prime

  pure function relu(x, alpha) result(res)
    !! REctified Linear Unit (RELU) activation function.
    real(rk), intent(in) :: x(:)
    real(rk), intent(in) :: alpha
    real(rk) :: res(size(x))
    res = max(0., x)
  end function relu

  pure function relu_prime(x, alpha) result(res)
    ! First derivative of the REctified Linear Unit (RELU) activation function.
    real(rk), intent(in) :: x(:)
    real(rk), intent(in) :: alpha
    real(rk) :: res(size(x))
    where (x > 0)
      res = 1
    elsewhere
      res = 0
    end where
  end function relu_prime

  pure function linear(x, alpha) result(res)
    !! Linear activation function.
    real(rk), intent(in) :: x(:)
    real(rk), intent(in) :: alpha
    real(rk) :: res(size(x))
    res = x
  end function linear

  pure function linear_prime(x, alpha) result(res)
    !! Linear activation function.
    real(rk), intent(in) :: x(:)
    real(rk), intent(in) :: alpha
    real(rk) :: res(size(x))
    res = 1
  end function linear_prime

  pure function sigmoid(x, alpha) result(res)
    ! Sigmoid activation function.
    real(rk), intent(in) :: x(:)
    real(rk), intent(in) :: alpha
    real(rk) :: res(size(x))
    res = 1 / (1 + exp(-x))
  endfunction sigmoid

  pure function sigmoid_prime(x, alpha) result(res)
    ! First derivative of the sigmoid activation function.
    real(rk), intent(in) :: x(:)
    real(rk), intent(in) :: alpha
    real(rk) ::	tmp_alpha
    real(rk) :: res(size(x))

    tmp_alpha = 0.0
    res = sigmoid(x, tmp_alpha) * (1 - sigmoid(x, tmp_alpha))
  end function sigmoid_prime

  pure function step(x, alpha) result(res)
    ! Step activation function.
    real(rk), intent(in) :: x(:)
    real(rk), intent(in) :: alpha
    real(rk) :: res(size(x))
    where (x > 0)
      res = 1
    elsewhere
      res = 0
    end where
  end function step

  pure function step_prime(x, alpha) result(res)
    ! First derivative of the step activation function.
    real(rk), intent(in) :: x(:)
    real(rk), intent(in) :: alpha
    real(rk) :: res(size(x))
    res = 0
  end function step_prime

  pure function tanhf(x, alpha) result(res)
    ! Tangent hyperbolic activation function.
    ! Same as the intrinsic tanh, but must be
    ! defined here so that we can use procedure
    ! pointer with it.
    real(rk), intent(in) :: x(:)
    real(rk), intent(in) :: alpha
    real(rk) :: res(size(x))
    res = tanh(x)
  end function tanhf

  pure function tanh_prime(x, alpha) result(res)
    ! First derivative of the tanh activation function.
    real(rk), intent(in) :: x(:)
    real(rk), intent(in) :: alpha
    real(rk) :: res(size(x))
    res = 1 - tanh(x)**2
  end function tanh_prime

end module mod_activation
