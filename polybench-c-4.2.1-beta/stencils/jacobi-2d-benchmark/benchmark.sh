model='XeonGold_6150'
reserve=36
mem='1GB'

N=8192
tsteps=1000
runs=25

module load new intel/2018.1
mkdir bin
make

mkdir timings
cd timings

mkdir mpi
mkdir hybrid
mkdir omp

for ranks in $(seq 1 $reserve)
do
  if [ "$1" = "mpi" ] || [ "$1" = "all" ]
  then
      cd mpi
      export OMP_NUM_THREADS=1
      bsub -We 01:00 -n $ranks -J "jacobi_2d_mpi_strong[$ranks]%36" -R "span[ptile=1]" -R "rusage[mem=$mem]" -R "select[model=$model]" -oo "output_$ranks.txt" mpirun -n $ranks ../../bin/jacobi-2d-mpi $runs $N $tsteps
      cd ..
  fi

  if [ "$1" = "hybrid" ]  || [ "$1" = "all" ]
  then
    cd hybrid

    if [ "$ranks" -eq "1" ]
    then
      mpi_ranks=1
      omp_ranks=1
    elif [ "$ranks" -eq "2" ]
    then
      mpi_ranks=2
      omp_ranks=1
    elif [ "$ranks" -eq "3" ]
    then
      mpi_ranks=3
      omp_ranks=1
    elif [ "$ranks" -eq "4" ]
    then
      mpi_ranks=2
      omp_ranks=2
    elif [ "$ranks" -eq "5" ]
    then
      mpi_ranks=5
      omp_ranks=1
    elif [ "$ranks" -eq "6" ]
    then
      mpi_ranks=3
      omp_ranks=2
    elif [ "$ranks" -eq "7" ]
    then
      mpi_ranks=7
      omp_ranks=1
    elif [ "$ranks" -eq "8" ]
    then
      mpi_ranks=4
      omp_ranks=2
    elif [ "$ranks" -eq "9" ]
    then
      mpi_ranks=3
      omp_ranks=3
    elif [ "$ranks" -eq "10" ]
    then
      mpi_ranks=5
      omp_ranks=2
    elif [ "$ranks" -eq "11" ]
    then
      mpi_ranks=11
      omp_ranks=1
    elif [ "$ranks" -eq "12" ]
    then
      mpi_ranks=4
      omp_ranks=3
    elif [ "$ranks" -eq "13" ]
    then
      mpi_ranks=13
      omp_ranks=1
    elif [ "$ranks" -eq "14" ]
    then
      mpi_ranks=7
      omp_ranks=2
    elif [ "$ranks" -eq "15" ]
    then
      mpi_ranks=5
      omp_ranks=3
    elif [ "$ranks" -eq "16" ]
    then
      mpi_ranks=4
      omp_ranks=4
    elif [ "$ranks" -eq "17" ]
    then
      mpi_ranks=17
      omp_ranks=1
    elif [ "$ranks" -eq "18" ]
    then
      mpi_ranks=6
      omp_ranks=3
    elif [ "$ranks" -eq "19" ]
    then
      mpi_ranks=19
      omp_ranks=1
    elif [ "$ranks" -eq "20" ]
    then
      mpi_ranks=5
      omp_ranks=4
    elif [ "$ranks" -eq "21" ]
    then
      mpi_ranks=7
      omp_ranks=3
    elif [ "$ranks" -eq "22" ]
    then
      mpi_ranks=11
      omp_ranks=2
    elif [ "$ranks" -eq "23" ]
    then
      mpi_ranks=23
      omp_ranks=1
    elif [ "$ranks" -eq "24" ]
    then
      mpi_ranks=6
      omp_ranks=4
    elif [ "$ranks" -eq "25" ]
    then
      mpi_ranks=5
      omp_ranks=5
    elif [ "$ranks" -eq "26" ]
    then
      mpi_ranks=13
      omp_ranks=2
    elif [ "$ranks" -eq "27" ]
    then
      mpi_ranks=9
      omp_ranks=3
    elif [ "$ranks" -eq "28" ]
    then
      mpi_ranks=7
      omp_ranks=4
    elif [ "$ranks" -eq "29" ]
    then
      mpi_ranks=29
      omp_ranks=1
    elif [ "$ranks" -eq "30" ]
    then
      mpi_ranks=6
      omp_ranks=5
    elif [ "$ranks" -eq "31" ]
    then
      mpi_ranks=31
      omp_ranks=1
    elif [ "$ranks" -eq "32" ]
    then
      mpi_ranks=8
      omp_ranks=4
    elif [ "$ranks" -eq "33" ]
    then
      mpi_ranks=11
      omp_ranks=3
    elif [ "$ranks" -eq "34" ]
    then
      mpi_ranks=17
      omp_ranks=2
    elif [ "$ranks" -eq "35" ]
    then
      mpi_ranks=7
      omp_ranks=5
    elif [ "$ranks" -eq "36" ]
    then
      mpi_ranks=9
      omp_ranks=4
    fi

    export OMP_NUM_THREADS=$omp_ranks
    bsub -We 04:00 -n $ranks -J "lu_hybrid_strong[$ranks]%36" -R "span[ptile=$omp_ranks]" -R "rusage[mem=$mem]" -R "select[model=$model]" -oo "output_$ranks.txt" "unset LSB_AFFINITY_HOSTFILE ; mpirun -n $mpi_ranks -ppn $omp_ranks ../../bin/jacobi-2d-hybrid $runs $N $tsteps"
  
    cd ..
  fi

  if [ "$1" = "openmp" ] || [ "$1" = "all" ]
  then
      
      cd omp
      export OMP_NUM_THREADS=$ranks

      if [ "$ranks" -le 4 ]
      then
          bsub -W 16:00 -n $ranks -J "jacobi_2d_lu_omp_strong[$ranks]%36" -R "span[ptile=$ranks]" -R "rusage[mem=$mem]" -R "select[model=$model]" -oo "output_$ranks.txt" ../../bin/jacobi-2d-omp $runs $N $tsteps
      else
          bsub -W 04:00 -n $ranks -J "jacobi_2d_lu_omp_strong[$ranks]%36" -R "span[ptile=$ranks]" -R "rusage[mem=$mem]" -R "select[model=$model]" -oo "output_$ranks.txt" ../../bin/jacobi-2d-omp $runs $N $tsteps
      fi

      cd ..
  fi
done
cd ..
