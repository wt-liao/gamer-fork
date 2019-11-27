#include "CUFLU.h"

#if (  MODEL == HYDRO  &&  ( FLU_SCHEME == MHM || FLU_SCHEME == MHM_RP )  )



// external functions
#ifdef __CUDACC__

#include "CUFLU_Shared_FluUtility.cu"
#include "CUFLU_Shared_DataReconstruction.cu"
#include "CUFLU_Shared_ComputeFlux.cu"
#include "CUFLU_Shared_FullStepUpdate.cu"

#if   ( RSOLVER == EXACT )
# include "CUFLU_Shared_RiemannSolver_Exact.cu"
#elif ( RSOLVER == ROE )
# include "CUFLU_Shared_RiemannSolver_Roe.cu"
#elif ( RSOLVER == HLLE )
# include "CUFLU_Shared_RiemannSolver_HLLE.cu"
#elif ( RSOLVER == HLLC )
# include "CUFLU_Shared_RiemannSolver_HLLC.cu"
#endif

#include "CUFLU_SetConstMem_FluidSolver.cu"

#else // #ifdef __CUDACC__

void Hydro_DataReconstruction( const real g_ConVar   [][ CUBE(FLU_NXT) ],
                                     real g_PriVar   [][ CUBE(FLU_NXT) ],
                                     real g_FC_Var   [][NCOMP_TOTAL][ CUBE(N_FC_VAR) ],
                                     real g_Slope_PPM[][NCOMP_TOTAL][ CUBE(N_SLOPE_PPM) ],
                               const bool Con2Pri, const int NIn, const int NGhost, const real Gamma,
                               const LR_Limiter_t LR_Limiter, const real MinMod_Coeff,
                               const real dt, const real dh, const real MinDens, const real MinPres,
                               const bool NormPassive, const int NNorm, const int NormIdx[],
                               const bool JeansMinPres, const real JeansMinPres_Coeff );
void Hydro_ComputeFlux( const real g_FC_Var [][NCOMP_TOTAL][ CUBE(N_FC_VAR) ],
                              real g_FC_Flux[][NCOMP_TOTAL][ CUBE(N_FC_FLUX) ],
                        const int Gap, const real Gamma, const bool CorrHalfVel, const real g_Pot_USG[],
                        const double g_Corner[], const real dt, const real dh, const double Time,
                        const OptGravityType_t GravityType, const double ExtAcc_AuxArray[], const real MinPres,
                        const bool DumpIntFlux, real g_IntFlux[][NCOMP_TOTAL][ SQR(PS2) ] );
void Hydro_FullStepUpdate( const real g_Input[][ CUBE(FLU_NXT) ], real g_Output[][ CUBE(PS2) ], char g_DE_Status[],
                           const real g_Flux[][NCOMP_TOTAL][ CUBE(N_FC_FLUX) ], const real dt, const real dh,
                           const real Gamma, const real MinDens, const real MinPres, const real DualEnergySwitch,
                           const bool NormPassive, const int NNorm, const int NormIdx[] );
#if   ( RSOLVER == EXACT )
void Hydro_RiemannSolver_Exact( const int XYZ, real Flux_Out[], const real L_In[], const real R_In[], const real Gamma );
#elif ( RSOLVER == ROE )
void Hydro_RiemannSolver_Roe( const int XYZ, real Flux_Out[], const real L_In[], const real R_In[],
                              const real Gamma, const real MinPres );
#elif ( RSOLVER == HLLE )
void Hydro_RiemannSolver_HLLE( const int XYZ, real Flux_Out[], const real L_In[], const real R_In[],
                               const real Gamma, const real MinPres );
#elif ( RSOLVER == HLLC )
void Hydro_RiemannSolver_HLLC( const int XYZ, real Flux_Out[], const real L_In[], const real R_In[],
                               const real Gamma, const real MinPres );
#endif
#if ( FLU_SCHEME == MHM_RP )
void Hydro_Con2Pri( const real In[], real Out[], const real Gamma_m1, const real MinPres,
                    const bool NormPassive, const int NNorm, const int NormIdx[],
                    const bool JeansMinPres, const real JeansMinPres_Coeff );
real Hydro_CheckMinPresInEngy( const real Dens, const real MomX, const real MomY, const real MomZ, const real Engy,
                               const real Gamma_m1, const real _Gamma_m1, const real MinPres );
#endif

#endif // #ifdef __CUDACC__ ... else ...


// internal functions
#if ( FLU_SCHEME == MHM_RP )
GPU_DEVICE
static void Hydro_RiemannPredict_Flux( const real g_ConVar[][ CUBE(FLU_NXT) ],
                                             real g_Half_Flux[][NCOMP_TOTAL][ CUBE(N_FC_FLUX) ],
                                       const real Gamma, const real MinPres );
GPU_DEVICE
static void Hydro_RiemannPredict( const real g_ConVar_In[][ CUBE(FLU_NXT) ],
                                  const real g_Half_Flux[][NCOMP_TOTAL][ CUBE(N_FC_FLUX) ],
                                        real g_Half_Var [][ CUBE(FLU_NXT) ],
                                  const real dt, const real dh, const real Gamma, const real MinDens, const real MinPres,
                                  const bool NormPassive, const int NNorm, const int NormIdx[],
                                  const bool JeansMinPres, const real JeansMinPres_Coeff );
                                  
#if (defined SUPPORT_GRACKLE) && (defined GRACKLE_H2_SOBOLEV)
GPU_DEVICE
static void Hydro_H2_Opacity(const real g_Half_PriVar[][ CUBE(FLU_NXT) ], real g_Output[][ CUBE(PS2) ],
                             const double H2_Op_T_Table[], const double H2_Op_Alpha_Table[],
                             const int H2_Op_N_elem, const real dh, const real Gamma, const real Unit_Dens );
#endif
                             
#endif // MHM_RP




//-------------------------------------------------------------------------------------------------------
// Function    :  CPU/CUFLU_FluidSolver_MHM
// Description :  CPU/GPU fluid solver based on the MUSCL-Hancock scheme
//
// Note        :  1. The three-dimensional evolution is achieved by using the unsplit method
//                2. Two half-step prediction schemes are supported, including "MHM" and "MHM_RP"
//                   MHM    : use interpolated face-centered values to calculate the half-step fluxes
//                   MHM_RP : use Riemann solver to calculate the half-step fluxes
//                3. Ref :
//                   MHM    : "Riemann Solvers and Numerical Methods for Fluid Dynamics
//                             - A Practical Introduction ~ by Eleuterio F. Toro"
//                   MHM_RP : Stone & Gardiner, NewA, 14, 139 (2009)
//                4. See include/CUFLU.h for the values and description of different symbolic constants
//                   such as N_FC_VAR, N_FC_FLUX, N_SLOPE_PPM, N_FL_FLUX, N_HF_VAR
//                5. Arrays with a prefix "g_" are stored in the global memory of GPU
//
// Parameter   :  g_Flu_Array_In     : Array storing the input fluid variables
//                g_Flu_Array_Out    : Array to store the output fluid variables
//                g_DE_Array_Out     : Array to store the dual-energy status
//                g_Flux_Array       : Array to store the output fluxes
//                g_Corner_Array     : Array storing the physical corner coordinates of each patch group (for UNSPLIT_GRAVITY)
//                g_Pot_Array_USG    : Array storing the input potential for UNSPLIT_GRAVITY
//                g_PriVar           : Array to store the primitive variables
//                g_Slope_PPM        : Array to store the slope for the PPM reconstruction
//                g_FC_Var           : Array to store the half-step variables
//                g_FC_Flux          : Array to store the face-centered fluxes
//                NPatchGroup        : Number of patch groups to be evaluated
//                dt                 : Time interval to advance solution
//                dh                 : Cell size
//                Gamma              : Ratio of specific heats
//                StoreFlux          : true --> store the coarse-fine fluxes
//                LR_Limiter         : Slope limiter for the data reconstruction in the MHM/MHM_RP/CTU schemes
//                                     (0/1/2/3/4) = (vanLeer/generalized MinMod/vanAlbada/
//                                                    vanLeer + generalized MinMod/extrema-preserving) limiter
//                MinMod_Coeff       : Coefficient of the generalized MinMod limiter
//                Time               : Current physical time                                     (for UNSPLIT_GRAVITY only)
//                GravityType        : Types of gravity --> self-gravity, external gravity, both (for UNSPLIT_GRAVITY only)
//                c_ExtAcc_AuxArray  : Auxiliary array for adding external acceleration          (for UNSPLIT_GRAVITY only)
//                                     --> When using GPU, this array is stored in the constant memory and does
//                                         not need to be passed as a function argument
//                                         --> Declared in CUFLU_SetConstMem_FluidSolver.cu with the prefix "c_" to
//                                             highlight that this is a constant variable on GPU
//                MinDens/Pres       : Minimum allowed density and pressure
//                DualEnergySwitch   : Use the dual-energy formalism if E_int/E_kin < DualEnergySwitch
//                NormPassive        : true --> normalize passive scalars so that the sum of their mass density
//                                              is equal to the gas mass density
//                NNorm              : Number of passive scalars to be normalized
//                                     --> Should be set to the global variable "PassiveNorm_NVar"
//                c_NormIdx          : Target variable indices to be normalized
//                                     --> Should be set to the global variable "PassiveNorm_VarIdx"
//                                     --> When using GPU, this array is stored in the constant memory and does
//                                         not need to be passed as a function argument
//                                         --> Declared in CUFLU_SetConstMem_FluidSolver.cu with the prefix "c_" to
//                                             highlight that this is a constant variable on GPU
//                JeansMinPres       : Apply minimum pressure estimated from the Jeans length
//                JeansMinPres_Coeff : Coefficient used by JeansMinPres = G*(Jeans_NCell*Jeans_dh)^2/(Gamma*pi);
//-------------------------------------------------------------------------------------------------------
#ifdef __CUDACC__
__global__
void CUFLU_FluidSolver_MHM(
   const real   g_Flu_Array_In [][NCOMP_TOTAL][ CUBE(FLU_NXT) ],
         real   g_Flu_Array_Out[][NCOMP_TOTAL][ CUBE(PS2) ],
         char   g_DE_Array_Out [][ CUBE(PS2) ],
         real   g_Flux_Array   [][9][NCOMP_TOTAL][ SQR(PS2) ],
   const double g_Corner_Array [][3],
   const real   g_Pot_Array_USG[][ CUBE(USG_NXT_F) ],
         real   g_PriVar       [][NCOMP_TOTAL][ CUBE(FLU_NXT) ],
         real   g_Slope_PPM    [][3][NCOMP_TOTAL][ CUBE(N_SLOPE_PPM) ],
         real   g_FC_Var       [][6][NCOMP_TOTAL][ CUBE(N_FC_VAR) ],
         real   g_FC_Flux      [][3][NCOMP_TOTAL][ CUBE(N_FC_FLUX) ],
   const real dt, const real dh, const real Gamma, const bool StoreFlux,
   const LR_Limiter_t LR_Limiter, const real MinMod_Coeff,
   const double Time, const OptGravityType_t GravityType,
   const real MinDens, const real MinPres, const real DualEnergySwitch,
   const bool NormPassive, const int NNorm,
   const bool JeansMinPres, const real JeansMinPres_Coeff )
#else
void CPU_FluidSolver_MHM(
   const real   g_Flu_Array_In [][NCOMP_TOTAL][ CUBE(FLU_NXT) ],
         real   g_Flu_Array_Out[][NCOMP_TOTAL][ CUBE(PS2) ],
         char   g_DE_Array_Out [][ CUBE(PS2) ],
         real   g_Flux_Array   [][9][NCOMP_TOTAL][ SQR(PS2) ],
   const double g_Corner_Array [][3],
   const real   g_Pot_Array_USG[][ CUBE(USG_NXT_F) ],
         real   g_PriVar       [][NCOMP_TOTAL][ CUBE(FLU_NXT) ],
         real   g_Slope_PPM    [][3][NCOMP_TOTAL][ CUBE(N_SLOPE_PPM) ],
         real   g_FC_Var       [][6][NCOMP_TOTAL][ CUBE(N_FC_VAR) ],
         real   g_FC_Flux      [][3][NCOMP_TOTAL][ CUBE(N_FC_FLUX) ],
   const int NPatchGroup, const real dt, const real dh, const real Gamma,
   const bool StoreFlux, const LR_Limiter_t LR_Limiter, const real MinMod_Coeff,
   const double Time, const OptGravityType_t GravityType,
   const double c_ExtAcc_AuxArray[], const real MinDens, const real MinPres,
   const real DualEnergySwitch, const bool NormPassive, const int NNorm, const int c_NormIdx[],
   const bool JeansMinPres, const real JeansMinPres_Coeff, 
   const double H2_Op_T_Table[], const double H2_Op_Alpha_Table[], 
   const int H2_Op_N_elem, const real Unit_Dens )
#endif // #ifdef __CUDACC__ ... else ...
{

#  ifdef UNSPLIT_GRAVITY
   const bool CorrHalfVel_Yes = true;
#  else
   const bool CorrHalfVel_No  = false;
#  endif
#  if   ( FLU_SCHEME == MHM )
   const bool Con2Pri_Yes     = true;
#  elif ( FLU_SCHEME == MHM_RP )
   const bool Con2Pri_No      = false;
#  endif


// openmp pragma for the CPU solver
#  ifndef __CUDACC__
#  pragma omp parallel
#  endif
   {
//    point to the arrays associated with different OpenMP threads (for CPU) or CUDA thread blocks (for GPU)
#     ifdef __CUDACC__
      const int array_idx = blockIdx.x;
#     else
#     ifdef OPENMP
      const int array_idx = omp_get_thread_num();
#     else
      const int array_idx = 0;
#     endif
#     endif // #ifdef __CUDACC__ ... else ...

      real (*const g_FC_Var_1PG   )[NCOMP_TOTAL][ CUBE(N_FC_VAR)    ] = g_FC_Var   [array_idx];
      real (*const g_FC_Flux_1PG  )[NCOMP_TOTAL][ CUBE(N_FC_FLUX)   ] = g_FC_Flux  [array_idx];
      real (*const g_PriVar_1PG   )             [ CUBE(FLU_NXT)     ] = g_PriVar   [array_idx];
      real (*const g_Slope_PPM_1PG)[NCOMP_TOTAL][ CUBE(N_SLOPE_PPM) ] = g_Slope_PPM[array_idx];

#     if ( FLU_SCHEME == MHM_RP )
      real (*const g_Half_Flux_1PG)[NCOMP_TOTAL][ CUBE(N_FC_FLUX) ] = g_FC_Flux_1PG;
      real (*const g_Half_Var_1PG )             [ CUBE(FLU_NXT)   ] = g_PriVar_1PG;
#     endif


//    loop over all patch groups
//    --> CPU/GPU solver: use different (OpenMP threads) / (CUDA thread blocks)
//        to work on different patch groups
#     ifdef __CUDACC__
      const int P = blockIdx.x;
#     else
#     pragma omp for schedule( runtime )
      for (int P=0; P<NPatchGroup; P++)
#     endif
      {

//       1. half-step prediction
//       1-a. MHM_RP: use Riemann solver to calculate the half-step fluxes
#        if ( FLU_SCHEME == MHM_RP )

//       1-a-1. evaluate the half-step first-order fluxes by Riemann solver
         Hydro_RiemannPredict_Flux( g_Flu_Array_In[P], g_Half_Flux_1PG, Gamma, MinPres );


//       1-a-2. evaluate the half-step solutions
         Hydro_RiemannPredict( g_Flu_Array_In[P], g_Half_Flux_1PG, g_Half_Var_1PG, dt, dh, Gamma, MinDens, MinPres,
                               NormPassive, NNorm, c_NormIdx, JeansMinPres, JeansMinPres_Coeff );


//       1-a-3. evaluate the face-centered values by data reconstruction
//              --> note that g_Half_Var_1PG[] returned by Hydro_RiemannPredict() stores the primitive variables
         Hydro_DataReconstruction( NULL, g_Half_Var_1PG, g_FC_Var_1PG, g_Slope_PPM_1PG,
                                   Con2Pri_No, N_HF_VAR, FLU_GHOST_SIZE-2,
                                   Gamma, LR_Limiter, MinMod_Coeff, dt, dh, MinDens, MinPres,
                                   NormPassive, NNorm, c_NormIdx, JeansMinPres, JeansMinPres_Coeff );


//       1-b. MHM: use interpolated face-centered values to calculate the half-step fluxes
#        elif ( FLU_SCHEME == MHM )

//       evaluate the face-centered values by data reconstruction
         Hydro_DataReconstruction( g_Flu_Array_In[P], g_PriVar_1PG, g_FC_Var_1PG, g_Slope_PPM_1PG,
                                   Con2Pri_Yes, FLU_NXT, FLU_GHOST_SIZE-1,
                                   Gamma, LR_Limiter, MinMod_Coeff, dt, dh, MinDens, MinPres,
                                   NormPassive, NNorm, c_NormIdx, JeansMinPres, JeansMinPres_Coeff );
#        endif // #if ( FLU_SCHEME == MHM_RP ) ... else ...


//       2. evaluate the full-step fluxes
#        ifdef UNSPLIT_GRAVITY
         Hydro_ComputeFlux( g_FC_Var_1PG, g_FC_Flux_1PG, 1, Gamma, CorrHalfVel_Yes,
                            g_Pot_Array_USG[P], g_Corner_Array[P],
                            dt, dh, Time, GravityType, c_ExtAcc_AuxArray, MinPres,
                            StoreFlux, g_Flux_Array[P] );
#        else
         Hydro_ComputeFlux( g_FC_Var_1PG, g_FC_Flux_1PG, 1, Gamma, CorrHalfVel_No,
                            NULL, NULL,
                            NULL_REAL, NULL_REAL, NULL_REAL, GRAVITY_NONE, NULL, MinPres,
                            StoreFlux, g_Flux_Array[P] );
#        endif


//       3. full-step evolution
         Hydro_FullStepUpdate( g_Flu_Array_In[P], g_Flu_Array_Out[P], g_DE_Array_Out[P],
                               g_FC_Flux_1PG, dt, dh, Gamma, MinDens, MinPres, DualEnergySwitch,
                               NormPassive, NNorm, c_NormIdx );
                               
#        if ( FLU_SCHEME == MHM_RP ) && (defined SUPPORT_GRACKLE) && (defined GRACKLE_H2_SOBOLEV)
         Hydro_H2_Opacity(g_Half_Var_1PG, g_Flu_Array_Out[P], H2_Op_T_Table, H2_Op_Alpha_Table, 
                          H2_Op_N_elem, dh, Gamma, Unit_Dens);
#        endif 

      } // loop over all patch groups
   } // OpenMP parallel region

} // FUNCTION : CPU_FluidSolver_MHM



#if ( FLU_SCHEME == MHM_RP )
//-------------------------------------------------------------------------------------------------------
// Function    :  Hydro_RiemannPredict_Flux
// Description :  Evaluate the half-step face-centered fluxes by Riemann solver
//
// Note        :  1. Work for the MHM_RP scheme
//                2. Currently support the exact, Roe, HLLE, and HLLC solvers
//                3. g_Half_Flux[] is accessed with the stride N_FC_FLUX
//                   --> Fluxes on the **left** face of the (i+1,j+1,k+1) element in g_ConVar[] will
//                       be stored in the (i,j,k) element of g_Half_Flux[]
//
// Parameter   :  g_ConVar    : Array storing the input conserved variables
//                g_Half_Flux : Array to store the output face-centered fluxes
//                Gamma       : Ratio of specific heats
//                MinPres     : Minimum allowed pressure
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE
void Hydro_RiemannPredict_Flux( const real g_ConVar[][ CUBE(FLU_NXT) ],
                                      real g_Half_Flux[][NCOMP_TOTAL][ CUBE(N_FC_FLUX) ],
                                const real Gamma, const real MinPres )
{

   const int didx_cvar[3] = { 1, FLU_NXT, SQR(FLU_NXT) };
   real ConVar_L[NCOMP_TOTAL], ConVar_R[NCOMP_TOTAL], Flux_1Face[NCOMP_TOTAL];

#  if ( RSOLVER == EXACT )
   const real Gamma_m1 = Gamma - (real)1.0;
   real PriVar_L[NCOMP_TOTAL], PriVar_R[NCOMP_TOTAL];
#  endif


// loop over different spatial directions
   for (int d=0; d<3; d++)
   {
      int gap[3];

      switch ( d )
      {
         case 0 : gap[0] = 0;  gap[1] = 1;  gap[2] = 1;  break;
         case 1 : gap[0] = 1;  gap[1] = 0;  gap[2] = 1;  break;
         case 2 : gap[0] = 1;  gap[1] = 1;  gap[2] = 0;  break;
      }

      const int size_i  = ( N_FC_FLUX - gap[0] );
      const int size_ij = ( N_FC_FLUX - gap[1] )*size_i;

      CGPU_LOOP( idx, N_FC_FLUX*SQR(N_FC_FLUX-1) )
      {
         const int i_flux   = idx % size_i;
         const int j_flux   = idx % size_ij / size_i;
         const int k_flux   = idx / size_ij;
         const int idx_flux = IDX321( i_flux, j_flux, k_flux, N_FC_FLUX, N_FC_FLUX );

         const int i_cvar   = i_flux + gap[0];
         const int j_cvar   = j_flux + gap[1];
         const int k_cvar   = k_flux + gap[2];
         const int idx_cvar = IDX321( i_cvar, j_cvar, k_cvar, FLU_NXT, FLU_NXT );

//       get the left and right states
         for (int v=0; v<NCOMP_TOTAL; v++)
         {
            ConVar_L[v] = g_ConVar[v][ idx_cvar              ];
            ConVar_R[v] = g_ConVar[v][ idx_cvar+didx_cvar[d] ];
         }

//       invoke the Riemann solver
#        if   ( RSOLVER == EXACT )
         const bool NormPassive_No  = false;  // do NOT convert any passive variable to mass fraction for the Riemann solvers
         const bool JeansMinPres_No = false;

         Hydro_Con2Pri( ConVar_L, PriVar_L, Gamma_m1, MinPres, NormPassive_No, NULL_INT, NULL, JeansMinPres_No, NULL_REAL );
         Hydro_Con2Pri( ConVar_R, PriVar_R, Gamma_m1, MinPres, NormPassive_No, NULL_INT, NULL, JeansMinPres_No, NULL_REAL );

         Hydro_RiemannSolver_Exact( d, Flux_1Face, PriVar_L, PriVar_R, Gamma );
#        elif ( RSOLVER == ROE )
         Hydro_RiemannSolver_Roe  ( d, Flux_1Face, ConVar_L, ConVar_R, Gamma, MinPres );
#        elif ( RSOLVER == HLLE )
         Hydro_RiemannSolver_HLLE ( d, Flux_1Face, ConVar_L, ConVar_R, Gamma, MinPres );
#        elif ( RSOLVER == HLLC )
         Hydro_RiemannSolver_HLLC ( d, Flux_1Face, ConVar_L, ConVar_R, Gamma, MinPres );
#        else
#        error : ERROR : unsupported Riemann solver (EXACT/ROE) !!
#        endif

//       store the results in g_Half_Flux[]
         for (int v=0; v<NCOMP_TOTAL; v++)   g_Half_Flux[d][v][idx_flux] = Flux_1Face[v];
      } // CGPU_LOOP( idx, N_FC_FLUX*SQR(N_FC_FLUX-1) )
   } // for (int d=0; d<3; d++)


#  ifdef __CUDACC__
   __syncthreads();
#  endif

} // FUNCTION : Hydro_RiemannPredict_Flux



//-------------------------------------------------------------------------------------------------------
// Function    :  Hydro_RiemannPredict
// Description :  Evolve the cell-centered variables by half time-step using the fluxes returned
//                by Hydro_RiemannPredict_Flux()
//
// Note        :  1. Work for the MHM_RP scheme
//                2. For the performance consideration, the output data are converted to primitive variables
//                   --> Reducing the global memory access on GPU
//
// Parameter   :  g_ConVar_In        : Array storing the input conserved variables
//                g_Half_Flux        : Array storing the input face-centered fluxes
//                                     --> Accessed with the stride N_FC_FLUX
//                g_Half_Var         : Array to store the output primitive variables
//                                     --> Accessed with the stride N_HF_VAR
//                                     --> Although its actually allocated size is FLU_NXT^3 since it points to g_PriVar_1PG[]
//                dt                 : Time interval to advance solution
//                dh                 : Cell size
//                Gamma              : Ratio of specific heats
//                MinDens/Pres       : Minimum allowed density and pressure
//                NormPassive        : true --> convert passive scalars to mass fraction
//                NNorm              : Number of passive scalars for the option "NormPassive"
//                                     --> Should be set to the global variable "PassiveNorm_NVar"
//                NormIdx            : Target variable indices for the option "NormPassive"
//                                     --> Should be set to the global variable "PassiveNorm_VarIdx"
//                JeansMinPres       : Apply minimum pressure estimated from the Jeans length
//                JeansMinPres_Coeff : Coefficient used by JeansMinPres = G*(Jeans_NCell*Jeans_dh)^2/(Gamma*pi);
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE
void Hydro_RiemannPredict( const real g_ConVar_In[][ CUBE(FLU_NXT) ],
                           const real g_Half_Flux[][NCOMP_TOTAL][ CUBE(N_FC_FLUX) ],
                                 real g_Half_Var [][ CUBE(FLU_NXT) ],
                           const real dt, const real dh, const real Gamma, const real MinDens, const real MinPres,
                           const bool NormPassive, const int NNorm, const int NormIdx[],
                           const bool JeansMinPres, const real JeansMinPres_Coeff )
{

   const int  didx_flux[3] = { 1, N_FC_FLUX, SQR(N_FC_FLUX) };
   const real dt_dh2       = (real)0.5*dt/dh;
   const real  Gamma_m1    = Gamma - (real)1.0;
   const real _Gamma_m1    = (real)1.0 / Gamma_m1;

   const int N_HF_VAR2 = SQR(N_HF_VAR);
   CGPU_LOOP( idx_out, CUBE(N_HF_VAR) )
   {
      const int i_flux   = idx_out % N_HF_VAR;
      const int j_flux   = idx_out % N_HF_VAR2 / N_HF_VAR;
      const int k_flux   = idx_out / N_HF_VAR2;
      const int idx_flux = IDX321( i_flux, j_flux, k_flux, N_FC_FLUX, N_FC_FLUX );

      const int i_in     = i_flux + 1;
      const int j_in     = j_flux + 1;
      const int k_in     = k_flux + 1;
      const int idx_in   = IDX321( i_in, j_in, k_in, FLU_NXT, FLU_NXT );

      real out_con[NCOMP_TOTAL], out_pri[NCOMP_TOTAL], dflux[3][NCOMP_TOTAL];

//    calculate the flux differences
      for (int d=0; d<3; d++)
      for (int v=0; v<NCOMP_TOTAL; v++)    dflux[d][v] = g_Half_Flux[d][v][ idx_flux+didx_flux[d] ] - g_Half_Flux[d][v][idx_flux];

//    update the input cell-centered conserved variables with the flux differences
      for (int v=0; v<NCOMP_TOTAL; v++)
         out_con[v] = g_ConVar_In[v][idx_in] - dt_dh2*( dflux[0][v] + dflux[1][v] + dflux[2][v] );

//    ensure positive density and pressure
      out_con[0] = FMAX( out_con[0], MinDens );
      out_con[4] = Hydro_CheckMinPresInEngy( out_con[0], out_con[1], out_con[2], out_con[3], out_con[4], Gamma_m1, _Gamma_m1, MinPres );
#     if ( NCOMP_PASSIVE > 0 )
      for (int v=NCOMP_FLUID; v<NCOMP_TOTAL; v++)
      out_con[v] = FMAX( out_con[v], TINY_NUMBER );
#     endif

//    conserved --> primitive variables
      Hydro_Con2Pri( out_con, out_pri, Gamma_m1, MinPres, NormPassive, NNorm, NormIdx, JeansMinPres, JeansMinPres_Coeff );

//    store the results to g_Half_Var[]
      for (int v=0; v<NCOMP_TOTAL; v++)   g_Half_Var[v][idx_out] = out_pri[v];
   } // i,j,k


#  ifdef __CUDACC__
   __syncthreads();
#  endif

} // FUNCTION : Hydro_RiemannPredict
#endif // #if ( FLU_SCHEME == MHM_RP )


#if (FLU_SCHEME == MHM_RP) && (defined SUPPORT_GRACKLE) && (defined GRACKLE_H2_SOBOLEV)
//-------------------------------------------------------------------------------------------------------
// Function    :  Hydro_H2_Opacity
// Description :  ###Evolve the cell-centered variables by half time-step using the fluxes returned
//                by Hydro_RiemannPredict_Flux()
//
// Note        :  ##1. Work for the MHM_RP scheme
//                ##2. For the performance consideration, the output data are converted to primitive variables
//                   --> Reducing the global memory access on GPU
//
// Parameter   :  ##g_ConVar_In        : Array storing the input conserved variables
//                ##g_Half_Flux        : Array storing the input face-centered fluxes
//                                     --> Accessed with the stride N_FC_FLUX
//                ##g_Half_Var         : Array to store the output primitive variables

//-------------------------------------------------------------------------------------------------------
GPU_DEVICE
void Hydro_H2_Opacity(const real g_Half_PriVar[][ CUBE(FLU_NXT) ], real g_Output[][ CUBE(PS2) ],
                      const double H2_Op_T_Table[], const double H2_Op_Alpha_Table[],
                      const int H2_Op_N_elem, const real dh, const real Gamma, const real Unit_Dens ){
                         
   real dens, _dens, pres, kbT, lnkbT, cs;
   real rho_HI, rho_HII, rho_H2I, rho_HeI, rho_e;
   real n_HI, n_HII, n_H2I, n_HeI, n_e, n_tot, mu;
   real alpha, dvx_dx, dvy_dy, dvz_dz, tau_x, tau_y, tau_z; 
   int Idx, ID_iL, ID_iR, ID_jL, ID_jR, ID_kL, ID_kR ;
   
   const double m_H           = (real)1.672621898e-24;
   const double _m_H          = (real)1.0/m_H;
   const double _m_2H         = _m_H / (real)2.0;
   const double _m_4H         = _m_H / (real)4.0;

   const double _2dh          = (real)0.5/dh ;
   const double Gamma_m1      = Gamma - (real)1.0;
   const double _Gamma_m1     = (real)1.0 / Gamma_m1; 
   
   const double Grackle_lnT_Start = H2_Op_T_Table[0];
   const double Grackle_dlnT      = H2_Op_T_Table[1] - H2_Op_T_Table[0];
   const double  _Grackle_dlnT    = (real)1.0 / Grackle_dlnT ; 
   
   const double* Alpha_Table = H2_Op_Alpha_Table; 
   const double* T_Table     = H2_Op_T_Table ; 
   
#  ifdef DUAL_ENERGY
   const bool CheckMinPres_Yes = true; 
#  endif
   
   const int size_ij = SQR(PS2);
   CGPU_LOOP( idx_out, CUBE(PS2) )
   {
      const int i_out    = idx_out % PS2;
      const int j_out    = idx_out % size_ij / PS2;
      const int k_out    = idx_out / size_ij;
      const int idx_op   = IDX321( i_out, j_out, k_out, PS2, PS2 );

      const int i_in     = i_out + FLU_GHOST_SIZE;
      const int j_in     = j_out + FLU_GHOST_SIZE;
      const int k_in     = k_out + FLU_GHOST_SIZE;
      const int idx_in   = IDX321( i_in, j_in, k_in, FLU_NXT, FLU_NXT );
      
      // 1.0 get mean moleculat weight mu:
      //     mu    = Sum(rho_HI + rho_HII + rho_H2 + rho_HeI) / (m_H*n_tot)
      //     n_tot = Sum(n_HI + n_HII + n_H2 + n_HeI + n_e)
      rho_HI  = g_Half_PriVar[Idx_HI ][idx_in] * Unit_Dens;
      rho_HII = g_Half_PriVar[Idx_HII][idx_in] * Unit_Dens;
      rho_H2I = g_Half_PriVar[Idx_H2I][idx_in] * Unit_Dens;
      rho_HeI = g_Half_PriVar[Idx_HeI][idx_in] * Unit_Dens;
      rho_e   = g_Half_PriVar[Idx_e  ][idx_in] * Unit_Dens;
      
      n_HI    = rho_HI  * _m_H ;
      n_HII   = rho_HII * _m_H ;
      n_H2I   = rho_H2I * _m_2H;
      n_HeI   = rho_HeI * _m_4H;
      n_e     = rho_e   * _m_H ;
      n_tot   = n_HI+n_HII+n_H2I+n_HeI+n_e;
      
      mu      = (rho_HI+rho_HII+rho_H2I+rho_HeI)/(m_H*n_tot);
      
      // 2.0 get half step physical variables
#     ifndef DUAL_ENERGY
      pres  = g_Half_PriVar[ENGY][idx_in];
#     else
      pres  = Hydro_DensEntropy2Pres(g_Half_PriVar[DENS][idx_in], g_Half_PriVar[ENPY][idx_in], 
                                     Gamma_m1, CheckMinPres_Yes, MIN_PRES);
#     endif
                                     
      dens  = g_Half_PriVar[DENS][idx_in];
      _dens = (real)1.0 / dens;
      kbT   = pres * _dens * mu; // kbT -> dimensionless (kb*T)/(m_H)
      lnkbT = LOG(kbT); 
      cs    = SQRT( Gamma*pres*_dens) ;
      
      
      // 3.0 get alpha 
      // get T_idx corresponding to table & interpolate
      Idx = int( (lnkbT-Grackle_lnT_Start)*_Grackle_dlnT ) ; 
      
      if (Idx<0)                        { alpha = Alpha_Table[0]; }
      else if (Idx >= (H2_Op_N_elem-1)) { alpha = Alpha_Table[H2_Op_N_elem-1]; } 
      else {
         alpha  = Alpha_Table[Idx] + 
                  (Alpha_Table[Idx+1]-Alpha_Table[Idx]) * ( (lnkbT-T_Table[Idx]) * _Grackle_dlnT ); 
      }
      
      // 4.0 get velocity gradient
      ID_iL = IDX321( i_in-1, j_in,   k_in,   FLU_NXT, FLU_NXT );
      ID_iR = IDX321( i_in+1, j_in,   k_in,   FLU_NXT, FLU_NXT );
      ID_jL = IDX321( i_in,   j_in-1, k_in,   FLU_NXT, FLU_NXT );
      ID_jR = IDX321( i_in,   j_in+1, k_in,   FLU_NXT, FLU_NXT );
      ID_kL = IDX321( i_in,   j_in,   k_in-1, FLU_NXT, FLU_NXT );
      ID_kR = IDX321( i_in,   j_in,   k_in+1, FLU_NXT, FLU_NXT );
      
      // calculate velocity gradient
      dvx_dx = (g_Half_PriVar[MOMX][ID_iR] - g_Half_PriVar[MOMX][ID_iL]) * _2dh;
      dvy_dy = (g_Half_PriVar[MOMY][ID_jR] - g_Half_PriVar[MOMY][ID_jL]) * _2dh;
      dvz_dz = (g_Half_PriVar[MOMZ][ID_kR] - g_Half_PriVar[MOMZ][ID_kL]) * _2dh;
      
      // 5.0 calculate tau
      tau_x = alpha* FABS(cs/dvx_dx); 
      tau_y = alpha* FABS(cs/dvy_dy);
      tau_z = alpha* FABS(cs/dvz_dz);
      
      // 6.0 store variables: alpha and tau
      // note: real tau = (tau*length_unit) * n_H2 <- do this in grackle
      g_Output[Idx_alpha ][idx_out] = alpha; 
      g_Output[Idx_OpTauX][idx_out] = tau_x;
      g_Output[Idx_OpTauY][idx_out] = tau_y;
      g_Output[Idx_OpTauZ][idx_out] = tau_z;

      
      //### if tau is very small, it might become difficult to calculate beta
      //### check this in grackle
      
   } // CGPU_LOOP
   
#  ifdef __CUDACC__
   __syncthreads();
#  endif
   
} // FUNCTION: Hydro_H2_Opacity

#endif // (FLU_SCHEME == MHM_RP) && (defined SUPPORT_GRACKLE) && (defined GRACKLE_H2_SOBOLEV)

#endif // #if (  MODEL == HYDRO  &&  ( FLU_SCHEME == MHM || FLU_SCHEME == MHM_RP )  )
