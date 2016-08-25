#! /bin/bash
set -o nounset

main() {

   buildCR_to_LxFunction
   buildCoolingFunction

   return
}

# ----------------------------------------------------------------------------------- #
# functions
# ----------------------------------------------------------------------------------- #

function buildCR_to_LxFunction
{

   Z_ARRAY=( 0.00 0.15 0.25 0.40 )
   OUTFILE=../data/CRtoLx.ascii

   echo "# ZGas Tx  z  log10M500c flux_0.5_2.0  CR_pn  CR_MOS  Lambda  Lx_bolo  Lx_0.1_2.4  Lx_0.5_2.0" > $OUTFILE
   for Z in ${Z_ARRAY[@]}; do
      awk '!(/^#/) {print '$Z', $0}' $HOME/projects/Stacked_X_ray/info/CRtoLx_Z${Z}.ascii >> $OUTFILE
   done


   return
}

function buildCoolingFunction
{

   Z_ARRAY=( 0.00 0.15 0.25 0.40 )
   NT=$( awk '!(/^#/) && $2==0.01 {n+=1} END {print n}' info/CRtoLx_Z${Z_ARRAY[0]}.ascii )
   NZ=${#Z_ARRAY[@]}
   NTNZ=$( python -c "print $NT*$NZ")

   OUTFILE=../data/Lambda.ascii
   echo "# ZGas Tx Lambda" > $OUTFILE
   for Z in ${Z_ARRAY[@]}; do
      awk '!(/^#/) && $2==0.01 {print '$Z', $1, $7}' $HOME/projects/Stacked_X_ray/info/CRtoLx_Z${Z}.ascii >> $OUTFILE
   done

   # c format
   # logLambda_t_ij = logLambda_t[j*xsize + i]
   # i = log(Tx)
   # j = Zgas
   awk '!(/^#/) {ni+=1;
                  if(ni=='$NT'*'$NZ'){
                     printf("%.8f", log($3))
                  }else{
                     printf("%.8f,", log($3))
                  }
               }
               BEGIN {printf("double logLambda_t['$NTNZ'] = {")}
               END {printf( "};\n")} '  $OUTFILE


   awk '!(/^#/) && $2==0.01 {ni+=1;
                  if(ni=='$NT'){
                     printf("%.8f", log($1))
                  }else{
                     printf("%.8f,", log($1))
                  }
               }
               BEGIN {printf("double logTx_t['$NT'] = {")}
               END {printf( "};\n")} ' $HOME/projects/Stacked_X_ray/info/CRtoLx_Z${Z_ARRAY[0]}.ascii

   echo "double ZGas_t[$NZ] = {" $( echo  ${Z_ARRAY[@]} | tr " " , ) "};"


   return

   # for c-format Lambda[i][j] (not used)
   awk '!(/^#/) {nj+=1;
               if(nj==1){
                  ni+=1;
                  printf("{%g,", $3)
               }else if(nj=='$NT' && ni=='$NZ'){
                  printf("%g}", $3)
               }else if(nj=='$NT'){
                  printf("%g,},\n", $3)
                  nj=0;
               }else{
                  printf("%g,", $3)
               }
               }

               BEGIN {printf("double Lambda_t['$NZ']['$NT'] = {")}
               END {printf( "};\n")} ' $OUTFILE

}


# ----------------------------------------------------------------------------------- #
# main
# ----------------------------------------------------------------------------------- #

main $@
