#include "GAMER.h"

#ifdef SUPPORT_GRACKLE


// global variables for accessing h_Che_Array[]
// --> declared in Init_MemAllocate_Grackle.cpp
extern int Che_NField;
extern int CheIdx_Dens;
extern int CheIdx_sEint;
extern int CheIdx_Ek;
extern int CheIdx_e;
extern int CheIdx_HI;
extern int CheIdx_HII;
extern int CheIdx_HeI;
extern int CheIdx_HeII;
extern int CheIdx_HeIII;
extern int CheIdx_HM;
extern int CheIdx_H2I;
extern int CheIdx_H2II;
extern int CheIdx_DI;
extern int CheIdx_DII;
extern int CheIdx_HDI;
extern int CheIdx_Metal;

#ifdef GRACKLE_H2_SOBOLEV
extern int CheIdx_H2_TauX ;
extern int CheIdx_H2_TauY ;
extern int CheIdx_H2_TauZ ;
#endif 




//-------------------------------------------------------------------------------------------------------
// Function    :  Grackle_Prepare
// Description :  Fill up the input host array h_Che_Array[] for the Grackle solver
//
// Note        :  1. Prepare Che_NField variables
//                   --> Che_NField and the corresponding array indices in h_Che_Array[] (e.g., CheIdx_Dens)
//                       are declared and set by Init_MemAllocate_Grackle()
//                2. This function always prepares the latest FluSg data
//
// Parameter   :  lv          : Target refinement level
//                h_Che_Array : Host array to store the prepared data
//                NPG         : Number of patch groups prepared at a time
//                PID0_List   : List recording the patch indices with LocalID==0 to be udpated
//-------------------------------------------------------------------------------------------------------
void Grackle_Prepare( const int lv, real h_Che_Array[], const int NPG, const int *PID0_List )
{

// check
#  ifdef GAMER_DEBUG
   if ( CheIdx_Dens == Idx_Undefined )
      Aux_Error( ERROR_INFO, "CheIdx_Dens is undefined !!\n" );
   if ( CheIdx_sEint == Idx_Undefined )
      Aux_Error( ERROR_INFO, "CheIdx_sEint is undefined !!\n" );
   if ( CheIdx_Ek == Idx_Undefined )
      Aux_Error( ERROR_INFO, "CheIdx_Ek is undefined !!\n" );

   if ( GRACKLE_PRIMORDIAL >= GRACKLE_PRI_CHE_NSPE6 ) {
      if (  Idx_e == Idx_Undefined  ||  CheIdx_e == Idx_Undefined  )
         Aux_Error( ERROR_INFO, "[Che]Idx_e is undefined for \"GRACKLE_PRI_CHE_NSPE6\" !!\n" );
      if (  Idx_HI == Idx_Undefined  ||  CheIdx_HI == Idx_Undefined  )
         Aux_Error( ERROR_INFO, "[Che]Idx_HI is undefined for \"GRACKLE_PRI_CHE_NSPE6\" !!\n" );
      if (  Idx_HII == Idx_Undefined  ||  CheIdx_HII == Idx_Undefined  )
         Aux_Error( ERROR_INFO, "[Che]Idx_HII is undefined for \"GRACKLE_PRI_CHE_NSPE6\" !!\n" );
      if (  Idx_HeI == Idx_Undefined  ||  CheIdx_HeI == Idx_Undefined  )
         Aux_Error( ERROR_INFO, "[Che]Idx_HeI is undefined for \"GRACKLE_PRI_CHE_NSPE6\" !!\n" );
      if (  Idx_HeII == Idx_Undefined  ||  CheIdx_HeII == Idx_Undefined  )
         Aux_Error( ERROR_INFO, "[Che]Idx_HeII is undefined for \"GRACKLE_PRI_CHE_NSPE6\" !!\n" );
      if (  Idx_HeIII == Idx_Undefined  ||  CheIdx_HeIII == Idx_Undefined  )
         Aux_Error( ERROR_INFO, "[Che]Idx_HeIII is undefined for \"GRACKLE_PRI_CHE_NSPE6\" !!\n" );
   }

   if ( GRACKLE_PRIMORDIAL >= GRACKLE_PRI_CHE_NSPE9 ) {
      if (  Idx_HM == Idx_Undefined  ||  CheIdx_HM == Idx_Undefined  )
         Aux_Error( ERROR_INFO, "[Che]Idx_HM is undefined for \"GRACKLE_PRI_CHE_NSPE9\" !!\n" );
      if (  Idx_H2I == Idx_Undefined  ||  CheIdx_H2I == Idx_Undefined  )
         Aux_Error( ERROR_INFO, "[Che]Idx_H2I is undefined for \"GRACKLE_PRI_CHE_NSPE9\" !!\n" );
      if (  Idx_H2II == Idx_Undefined  ||  CheIdx_H2II == Idx_Undefined  )
         Aux_Error( ERROR_INFO, "[Che]Idx_H2II is undefined for \"GRACKLE_PRI_CHE_NSPE9\" !!\n" );
   }

   if ( GRACKLE_PRIMORDIAL >= GRACKLE_PRI_CHE_NSPE12 ) {
      if (  Idx_DI == Idx_Undefined  ||  CheIdx_DI == Idx_Undefined  )
         Aux_Error( ERROR_INFO, "[Che]Idx_DI is undefined for \"GRACKLE_PRI_CHE_NSPE12\" !!\n" );
      if (  Idx_DII == Idx_Undefined  ||  CheIdx_DII == Idx_Undefined  )
         Aux_Error( ERROR_INFO, "[Che]Idx_DII is undefined for \"GRACKLE_PRI_CHE_NSPE12\" !!\n" );
      if (  Idx_HDI == Idx_Undefined  ||  CheIdx_HDI == Idx_Undefined  )
         Aux_Error( ERROR_INFO, "[Che]Idx_HDI is undefined for \"GRACKLE_PRI_CHE_NSPE12\" !!\n" );
   }

   if ( GRACKLE_METAL ) {
      if (  Idx_Metal == Idx_Undefined  ||  CheIdx_Metal == Idx_Undefined  )
         Aux_Error( ERROR_INFO, "[Che]Idx_Metal is undefined for \"GRACKLE_METAL\" !!\n" );
   }
   
#  ifdef GRACKLE_H2_SOBOLEV
   if ( Idx_OpTauX == Idx_Undefined || CheIdx_H2_TauX == Idx_Undefined )
      Aux_Error( ERROR_INFO, "[Che]Idx_OpTauX is undefined for \"GRACKLE_H2_SOBOLEV\" !!\n" );
   if ( Idx_OpTauY == Idx_Undefined || CheIdx_H2_TauY == Idx_Undefined )
      Aux_Error( ERROR_INFO, "[Che]Idx_OpTauY is undefined for \"GRACKLE_H2_SOBOLEV\" !!\n" );
   if ( Idx_OpTauZ == Idx_Undefined || CheIdx_H2_TauZ == Idx_Undefined )
      Aux_Error( ERROR_INFO, "[Che]Idx_OpTauZ is undefined for \"GRACKLE_H2_SOBOLEV\" !!\n" );
#  endif
   
#  endif // #ifdef GAMER_DEBUG


   const int  Size1pg         = CUBE(PS2);
   const int  Size1v          = NPG*Size1pg;
   const real mass_ratio_pe   = Const_mp/Const_me;
#  ifdef DUAL_ENERGY
   const real  Gamma_m1       = GAMMA - (real)1.0;
   const real _Gamma_m1       = (real)1.0 / Gamma_m1;
   const bool CheckMinPres_No = false;
#  endif

   real *Ptr_Dens0  = h_Che_Array + CheIdx_Dens *Size1v;
   real *Ptr_sEint0 = h_Che_Array + CheIdx_sEint*Size1v;
   real *Ptr_Ek0    = h_Che_Array + CheIdx_Ek   *Size1v;
   real *Ptr_e0     = h_Che_Array + CheIdx_e    *Size1v;
   real *Ptr_HI0    = h_Che_Array + CheIdx_HI   *Size1v;
   real *Ptr_HII0   = h_Che_Array + CheIdx_HII  *Size1v;
   real *Ptr_HeI0   = h_Che_Array + CheIdx_HeI  *Size1v;
   real *Ptr_HeII0  = h_Che_Array + CheIdx_HeII *Size1v;
   real *Ptr_HeIII0 = h_Che_Array + CheIdx_HeIII*Size1v;
   real *Ptr_HM0    = h_Che_Array + CheIdx_HM   *Size1v;
   real *Ptr_H2I0   = h_Che_Array + CheIdx_H2I  *Size1v;
   real *Ptr_H2II0  = h_Che_Array + CheIdx_H2II *Size1v;
   real *Ptr_DI0    = h_Che_Array + CheIdx_DI   *Size1v;
   real *Ptr_DII0   = h_Che_Array + CheIdx_DII  *Size1v;
   real *Ptr_HDI0   = h_Che_Array + CheIdx_HDI  *Size1v;
   real *Ptr_Metal0 = h_Che_Array + CheIdx_Metal*Size1v;
   
#  ifdef GRACKLE_H2_SOBOLEV
   real *Ptr_H2_Tau_X0 = h_Che_Array + CheIdx_H2_TauX * Size1v;
   real *Ptr_H2_Tau_Y0 = h_Che_Array + CheIdx_H2_TauY * Size1v;
   real *Ptr_H2_Tau_Z0 = h_Che_Array + CheIdx_H2_TauZ * Size1v;
#  endif 


#  pragma omp parallel
   {

// thread-private variables
   int  idx_p, idx_pg, PID, PID0, offset;    // idx_p/idx_pg: array indices within a patch/patch group
   real Dens, Px, Py, Pz, Etot, _Dens, Ek, sEint;
   real (*fluid)[PS1][PS1][PS1]=NULL;

   real *Ptr_Dens=NULL, *Ptr_sEint=NULL, *Ptr_Ek=NULL, *Ptr_e=NULL, *Ptr_HI=NULL, *Ptr_HII=NULL;
   real *Ptr_HeI=NULL, *Ptr_HeII=NULL, *Ptr_HeIII=NULL, *Ptr_HM=NULL, *Ptr_H2I=NULL, *Ptr_H2II=NULL;
   real *Ptr_DI=NULL, *Ptr_DII=NULL, *Ptr_HDI=NULL, *Ptr_Metal=NULL;
#  ifdef GRACKLE_H2_SOBOLEV
   real *Ptr_H2_Tau_X=NULL, *Ptr_H2_Tau_Y=NULL, *Ptr_H2_Tau_Z=NULL;
#  endif

#  pragma omp for schedule( static )
   for (int TID=0; TID<NPG; TID++)
   {
      PID0      = PID0_List[TID];
      idx_pg    = 0;
      offset    = TID*Size1pg;

      Ptr_Dens  = Ptr_Dens0  + offset;
      Ptr_sEint = Ptr_sEint0 + offset;
      Ptr_Ek    = Ptr_Ek0    + offset;
      Ptr_e     = Ptr_e0     + offset;
      Ptr_HI    = Ptr_HI0    + offset;
      Ptr_HII   = Ptr_HII0   + offset;
      Ptr_HeI   = Ptr_HeI0   + offset;
      Ptr_HeII  = Ptr_HeII0  + offset;
      Ptr_HeIII = Ptr_HeIII0 + offset;
      Ptr_HM    = Ptr_HM0    + offset;
      Ptr_H2I   = Ptr_H2I0   + offset;
      Ptr_H2II  = Ptr_H2II0  + offset;
      Ptr_DI    = Ptr_DI0    + offset;
      Ptr_DII   = Ptr_DII0   + offset;
      Ptr_HDI   = Ptr_HDI0   + offset;
      Ptr_Metal = Ptr_Metal0 + offset;
#     ifdef GRACKLE_H2_SOBOLEV
      Ptr_H2_Tau_X = Ptr_H2_Tau_X0 + offset;
      Ptr_H2_Tau_Y = Ptr_H2_Tau_Y0 + offset;
      Ptr_H2_Tau_Z = Ptr_H2_Tau_Z0 + offset;
#     endif

      for (int LocalID=0; LocalID<8; LocalID++)
      {
         PID   = PID0 + LocalID;
         idx_p = 0;
         fluid = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid;

         for (int k=0; k<PS1; k++)
         for (int j=0; j<PS1; j++)
         for (int i=0; i<PS1; i++)
         {
            Dens  = *( fluid[DENS][0][0] + idx_p );
            Px    = *( fluid[MOMX][0][0] + idx_p );
            Py    = *( fluid[MOMY][0][0] + idx_p );
            Pz    = *( fluid[MOMZ][0][0] + idx_p );
            Etot  = *( fluid[ENGY][0][0] + idx_p );
            _Dens = (real)1.0 / Dens;
            Ek    = (real)0.5*( SQR(Px) + SQR(Py) + SQR(Pz) )*_Dens;

//          use the dual-energy variable to calculate the internal energy if applicable
#           ifdef DUAL_ENERGY

#           if   ( DUAL_ENERGY == DE_ENPY )
            sEint = Hydro_DensEntropy2Pres( Dens, *(fluid[ENPY][0][0]+idx_p), Gamma_m1, CheckMinPres_No, NULL_REAL )
                    *_Dens*_Gamma_m1;
#           elif ( DUAL_ENERGY == DE_EINT )
#           error : DE_EINT is NOT supported yet !!
#           endif

#           else // #ifdef DUAL_ENERGY

            sEint  = Etot - Ek;
#           ifdef MHD
            sEint -= MHD_GetCellCenteredBEnergyInPatch( lv, PID, i, j, k, amr->MagSg[lv] );
#           endif
            sEint *= _Dens;
#           endif // #ifdef DUAL_ENERGY ... else

//          mandatory fields
            Ptr_Dens [idx_pg] = Dens;
            Ptr_sEint[idx_pg] = sEint;
            Ptr_Ek   [idx_pg] = Ek;

//          6-species network
            if ( GRACKLE_PRIMORDIAL >= GRACKLE_PRI_CHE_NSPE6 ) {
            Ptr_e    [idx_pg] = *( fluid[Idx_e    ][0][0] + idx_p );// * mass_ratio_pe;
            Ptr_HI   [idx_pg] = *( fluid[Idx_HI   ][0][0] + idx_p );
            Ptr_HII  [idx_pg] = *( fluid[Idx_HII  ][0][0] + idx_p );
            Ptr_HeI  [idx_pg] = *( fluid[Idx_HeI  ][0][0] + idx_p );
            Ptr_HeII [idx_pg] = *( fluid[Idx_HeII ][0][0] + idx_p );
            Ptr_HeIII[idx_pg] = *( fluid[Idx_HeIII][0][0] + idx_p );
            }

//          9-species network
            if ( GRACKLE_PRIMORDIAL >= GRACKLE_PRI_CHE_NSPE9 ) {
            Ptr_HM   [idx_pg] = *( fluid[Idx_HM   ][0][0] + idx_p );
            Ptr_H2I  [idx_pg] = *( fluid[Idx_H2I  ][0][0] + idx_p );
            Ptr_H2II [idx_pg] = *( fluid[Idx_H2II ][0][0] + idx_p );
            }

//          12-species network
            if ( GRACKLE_PRIMORDIAL >= GRACKLE_PRI_CHE_NSPE12 ) {
            Ptr_DI   [idx_pg] = *( fluid[Idx_DI   ][0][0] + idx_p );
            Ptr_DII  [idx_pg] = *( fluid[Idx_DII  ][0][0] + idx_p );
            Ptr_HDI  [idx_pg] = *( fluid[Idx_HDI  ][0][0] + idx_p );
            }

//          metallicity for metal cooling
            if ( GRACKLE_METAL )
            Ptr_Metal[idx_pg] = *( fluid[Idx_Metal][0][0] + idx_p );
            
#           ifdef GRACKLE_H2_SOBOLEV
            Ptr_H2_Tau_X[idx_pg] = *( fluid[Idx_OpTauX][0][0] + idx_p );
            Ptr_H2_Tau_Y[idx_pg] = *( fluid[Idx_OpTauY][0][0] + idx_p );
            Ptr_H2_Tau_Z[idx_pg] = *( fluid[Idx_OpTauZ][0][0] + idx_p );
#           endif 

            idx_p  ++;
            idx_pg ++;
         } // i,j,k

      } // for (int LocalID=0; LocalID<8; LocalID++)
   } // for (int TID=0; TID<NPG; TID++)

   } // end of OpenMP parallel region


// set cell size and link pointers for different fields
   Che_FieldData->grid_dx         = amr->dh[lv];

   Che_FieldData->density         = Ptr_Dens0;
   Che_FieldData->internal_energy = Ptr_sEint0;

   if ( GRACKLE_PRIMORDIAL >= GRACKLE_PRI_CHE_NSPE6 ) {
   Che_FieldData->e_density       = Ptr_e0;
   Che_FieldData->HI_density      = Ptr_HI0;
   Che_FieldData->HII_density     = Ptr_HII0;
   Che_FieldData->HeI_density     = Ptr_HeI0;
   Che_FieldData->HeII_density    = Ptr_HeII0;
   Che_FieldData->HeIII_density   = Ptr_HeIII0;
   }

   if ( GRACKLE_PRIMORDIAL >= GRACKLE_PRI_CHE_NSPE9 ) {
   Che_FieldData->HM_density      = Ptr_HM0;
   Che_FieldData->H2I_density     = Ptr_H2I0;
   Che_FieldData->H2II_density    = Ptr_H2II0;
   }

   if ( GRACKLE_PRIMORDIAL >= GRACKLE_PRI_CHE_NSPE12 ) {
   Che_FieldData->DI_density      = Ptr_DI0;
   Che_FieldData->DII_density     = Ptr_DII0;
   Che_FieldData->HDI_density     = Ptr_HDI0;
   }

   if ( GRACKLE_METAL )
   Che_FieldData->metal_density   = Ptr_Metal0;
   
#  ifdef GRACKLE_H2_SOBOLEV
   if ( GRACKLE_H2_OPA_APPROX == 2 ) {
   Che_FieldData->H2_Sobolev_tau_x = Ptr_H2_Tau_X0;
   Che_FieldData->H2_Sobolev_tau_y = Ptr_H2_Tau_Y0;
   Che_FieldData->H2_Sobolev_tau_z = Ptr_H2_Tau_Z0;
   }
#  endif

} // FUNCTION : Grackle_Prepare



#endif // #ifdef SUPPORT_GRACKLE
