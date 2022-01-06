model='XeonGold_6150'
N=8192
runs=25

gcd() (
    if (( $1 % $2 == 0)); then
        echo $2
     else
        gcd $2 $(( $1 % $2 ))
    fi
)

module load new intel/2018.1
mkdir bin
make

mkdir timings
cd timings

mkdir mpi
mkdir hybrid
mkdir scalapack
mkdir omp

mem='1GB'

let reserve=36
for ranks in $(seq 1 $reserve)
do
  if [ "$1" = "mpi" ] || [ "$1" = "all" ]
  then
      cd mpi
      bsub -We 01:00 -n $reserve -J "lu_mpi_strong[$ranks]%36" -R "span[ptile=36]" -R "rusage[mem=$mem]" -R "select[model=$model]" -oo "output_$ranks.txt" mpirun -n $ranks ../../bin/lu-mpi $runs $N
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
    bsub -We 01:00 -n $ranks -J "lu_hybrid_strong[$ranks]%36" -R "span[ptile=$omp_ranks]" -R "rusage[mem=$mem]" -R "select[model==$model]" -oo "output_$ranks.txt" mpirun -n $mpi_ranks ../../bin/lu-hybrid $runs $N
  
    cd ..
  fi

  if [ "$1" = "scalapack" ] || [ "$1" = "all" ]
  then
      cd scalapack

      bsub -W 01:00 -n $reserve -J "lu_scalapack_strong[$ranks]%36" -R "span[ptile=36]" -R "rusage[mem=$mem]" -R "select[model=$model]" -oo "output_$ranks.txt" mpirun -n $ranks ../../bin/lu-scalapack $runs $N

      cd ..
  fi

  if [ "$1" = "openmp" ] || [ "$1" = "all" ]
  then
      
      cd omp
      export OMP_NUM_THREADS=$ranks

      if [ "$ranks" -le 4 ]
      then
          bsub -W 16:00 -n $reserve -J "lu_omp_strong[$ranks]%36" -R "span[ptile=36]" -R "rusage[mem=$mem]" -R "select[model=$model]" -oo "output_$ranks.txt" ../../bin/lu-omp $runs $N
      else
          bsub -W 04:00 -n $reserve -J "lu_omp_strong[$ranks]%36" -R "span[ptile=36]" -R "rusage[mem=$mem]" -R "select[model=$model]" -oo "output_$ranks.txt" ../../bin/lu-omp $runs $N
      fi

      cd ..
  fi
done
cd ..
