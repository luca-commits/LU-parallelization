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
      bsub -We 01:00 -n $reserve -R "span[ptile=36]" -R "rusage[mem=$mem]" -R "select[model==$model]" -oo "output_$ranks.txt" mpirun -n $ranks ../../bin/lu-mpi $runs $N
      cd ..
  fi

  # if [ "$1" = "hybrid"]
  # then
  #   cd hybrid

  #   if [ "gcd $ranks 12" -neq "0"]
  #   then

  #   elif [ "gcd $ranks 8" -neq "0" ]
  #   then
      
  #   fi

  #   bsub -We 01:00 -n $reserve -R "span[ptile=24]" -R "rusage[mem=$mem]" -R "select[model==$model]" -oo "output_$ranks.txt" mpirun -n $ranks ../../bin/lu-mpi $runs $N
  # fi

  if [ "$1" = "scalapack" ] || [ "$1" = "all" ]
  then
      cd scalapack

      if [ "$ranks" -le 4 ]
      then
          bsub -W 16:00 -n $reserve -R "span[ptile=36]" -R "rusage[mem=$mem]" -R "select[model==$model]" -oo "output_$ranks.txt" mpirun -n $ranks ../../bin/lu-scalapack $runs $N
      else
          bsub -W 04:00 -n $reserve -R "span[ptile=36]" -R "rusage[mem=$mem]" -R "select[model==$model]" -oo "output_$ranks.txt" mpirun -n $ranks ../../bin/lu-scalapack $runs $N
      fi

      cd ..
  fi

  if [ "$1" = "openmp" ] || [ "$1" = "all" ]
  then
      
      cd omp
      export OMP_NUM_THREADS=$ranks

      if [ "$ranks" -le 4 ]
      then
          bsub -W 16:00 -n $reserve -R "span[ptile=36]" -R "rusage[mem=$mem]" -R "select[model==$model]" -oo "output_$ranks.txt" ../../bin/lu-omp $runs $N
      else
          bsub -W 04:00 -n $reserve -R "span[ptile=36]" -R "rusage[mem=$mem]" -R "select[model==$model]" -oo "output_$ranks.txt" ../../bin/lu-omp $runs $N
      fi

      cd ..
  fi
done
cd ..
