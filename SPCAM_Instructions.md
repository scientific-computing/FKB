# SPCAM Instructions

### Modifications to SPCAM

1. Change `openmp` to `qopenmp`
  * `models/atm/cam/bld/Makefile.stampede`
  * `models/utils/esmf/build/linux_intel/base_variables`

2. Add to [Makefile.stampede](https://gitlab.com/mspritch/spcam3.0-neural-net/blob/nn_fbp_engy_ess/models/atm/cam/bld/Makefile.stampede)
```
NEURAL_MOD := -I$HOME/neural-fortran/build/include/
NEURAL_A := -L$HOME/neural-fortran/build/lib/
NEURAL_O := -L$HOME/neural-fortran/build/CMakeFiles/neural.dir/
```

3. Add `$(NEURAL_O) $(NEURAL_A) -lneural` to [line](https://gitlab.com/mspritch/spcam3.0-neural-net/blob/nn_fbp_engy_ess/models/atm/cam/bld/Makefile.stampede#L167)

4. Add `$(NEURAL_MOD)` to [line](https://gitlab.com/mspritch/spcam3.0-neural-net/blob/nn_fbp_engy_ess/models/atm/cam/bld/Makefile.stampede#L330)


### Neural-Fortran
```
cd $HOME
git clone https://github.com/jordanott/neural-fortran.git
cd neural-fortran
```

Set desired compiler in `build_steps.sh`
  * For example, using mpif90: `FC=mpif90 cmake .. -DSERIAL=1`

Then compile 
`sh build_steps.sh`


### Example Use Case

```
! example_nn.F90
! Outline of how to use neural fortran

module example_nn

! -------- NEURAL-FORTRAN --------
use mod_kinds, only: ik, rk
use mod_network , only: network_type
use mod_ensemble, only: ensemble_type
! --------------------------------

  implicit none
  save 

  private  

#ifdef ENSEMBLE
  type(ensemble_type) :: example_ensemble
#else
  type(network_type) :: example_nn
#endif


  public get_output, init_model
  contains

  subroutine get_output (inputs)
  ! allocate inputs and outputs 
  
#ifdef ENSEMBLE
    output = example_ensemble % average(input)
#else
    ! use neural fortran library
    output = example_nn % output(input)
#endif

  end subroutine get_output


  subroutine init_model()    

#ifdef ENSEMBLE
    ! loading all models in Models/
    example_ensemble = ensemble_type('./Models/')
#else
    ! Loading a single network from model.txt
    call example_nn % load('./keras_matrices/model.txt')
#endif

  end subroutine init_model

end module example_nn
```

### Build SPCAM

