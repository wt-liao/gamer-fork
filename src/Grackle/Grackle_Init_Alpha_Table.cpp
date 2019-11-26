#include "GAMER.h"


//### decalre Grackle_Load_Alpha_Table

#if (defined SUPPORT_GRACKLE) && (defined GRACKLE_H2_SOBOLEV) 
//-------------------------------------------------------------------------------------------------------
// Function    :  Grackle_Load_Alpha_Table
// Description :  Load table of H2 abosorption coefficient
//-------------------------------------------------------------------------------------------------------
void Grackle_Init_Alpha_Table(){
   
   // check compile
#  if (FLU_SCHEME != MHM_RP)
   Aux_Error( ERROR_INFO, "GRACKLE_H2_SOBOLEV only work with MHM_RP solver !!\n" );
#  endif
   
#  ifndef DUAL_ENERGY
   Aux_Error( ERROR_INFO, "GRACKLE_H2_SOBOLEV only work with DUAL_ENERGY scheme !!\n" );
#  endif
   
   // read in 
   FILE * T_Table;
   FILE * Alpha_Table; 
   
   // allocate memory
   H2_Op_T_Table     = new double[H2_Op_N_elem]; 
   H2_Op_Alpha_Table = new double[H2_Op_N_elem];
   
   
   // 1.0 open the files: T_Table should be in Ln(T) space
   T_Table     = fopen( "H2_Op_T" ,    "rb" );
   Alpha_Table = fopen( "H2_Op_Alpha", "rb" );
   
   if (T_Table == NULL)
      Aux_error(ERROR_INFO, "ERROR: cannot find <H2_Op_T>: H2 Opacity T-table . \n");
   if (Alpha_Table == NULL)
      Aux_error(ERROR_INFO, "ERROR: cannot find <H2_Op_Alpha>: H2 Opacity Alpha-table. \n");
   
   
   // 2.0 find the size of the array -> N of eletments 
   fseek (T_Table ,     0 , SEEK_END);
   fseek (Alpha_Table , 0 , SEEK_END);
   
   long Table_T_Size     = ftell(T_Table)    ;
   long Table_Alpha_Size = ftell(Alpha_Table);
   
   const int N_elem = int( Table_T_Size/sizeof(double) ); 
   
   if (H2_Op_N_elem != N_elem)
      Aux_error(ERROR_INFO, "Element in H2 Table should be %d, but is %d. \n", H2_Op_N_elem, N_elem);
   
   if (Table_T_Size != Table_Alpha_Size)
      Aux_error(ERROR_INFO, "Error in H2 Opacity: size of T-table=%d and Alpha-table=%d. \n ", 
                Table_T_Size, Table_Alpha_Size) ;

   if (MPI_Rank == 0)
      Aux_Message(stdout, "H2 Opacity NOTE: %d elements in H2 Opacity Table. \n", N_elem); 
   
   
   // 3.0 store T-table and Alpha-tables  
   rewind(T_Table);
   rewind(Alpha_Table);
   
   fread(H2_Op_T_Table,     sizeof(double), N_elem, T_Table    ); 
   fread(H2_Op_Alpha_Table, sizeof(double), N_elem, Alpha_Table); 
   
   
   // 4.0 rescale T: svae ln(RT) instead of ln(T)
   const double length_unit = Che_Units.length_units;
   const double time_unit   = Che_Units.time_units;
   const double const_R     = (Const_kB/Const_mH) * SQR(time_unit/length_unit) ;
   const double lnR         = LOG(const_R);
   
   for (int n=0; n<N_elem; n++)  H2_Op_T_Table[n] += lnR;
   
   
   // 5.0 close file
   fclose(T_Table); 
   fclose(Alpha_Table);
   

   
   //### free H2_Op_T_Table, H2_Op_Alpha_Table in End_MemFree_Grackle(). Done!
   
}

#endif