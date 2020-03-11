module mod_kinds

  use iso_fortran_env, only: int32, int64, real32, real64, real128

  implicit none

  private
  public :: ik, rk, fk

#ifdef REAL64
  integer,parameter :: rk = real64
#elif REAL128
  integer,parameter :: rk = real128
#else
  integer,parameter :: rk = real32
#endif

integer,parameter :: fk = selected_real_kind(12)

#ifdef INT64
  integer, parameter :: ik = int64
#else
  integer, parameter :: ik = int32
#endif

end module mod_kinds
