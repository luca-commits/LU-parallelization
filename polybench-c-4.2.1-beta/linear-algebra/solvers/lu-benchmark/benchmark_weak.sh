model='XeonGold_5118'
N_base=1024
runs=25

module load new intel/2018.1
mkdir bin
make

mkdir timings_weak
cd timings_weak

mkdir mpi
mkdir scalapack
mkdir omp

for nodes in $(seq 1 2)
do
  if [ $nodes -eq 1 ]
  then 
    mem='1GB'
  else
    mem='0.5GB'
  fi

  let reserve=nodes*24
  let reserve_low=reserve-23
  for ranks in $(seq $reserve_low $reserve)
  do
    scaling=$(bc <<< "scale=2; sqrt($ranks)")
    N=$(bc <<< "($N_base * $scaling) / 1")

    echo $N

    if [ "$1" = "mpi" ]
    then
        cd mpi
        bsub -We 01:00 -n $reserve -R "span[ptile=24]" -R "rusage[mem=$mem]" -R "select[model==$model]" -o "output_$ranks.txt" mpirun -n $ranks ../../bin/lu-mpi $runs $N
    fi

    if [ "$1" = "scalapack" ]
    then
        cd scalapack

        if [ "$ranks" -le 4 ]
        then
            bsub -W 16:00 -n $reserve -R "span[ptile=24]" -R "rusage[mem=$mem]" -R "select[model==$model]" -o "output_$ranks.txt" mpirun -n $ranks ../../bin/lu-scalapack $runs $N
        else
            bsub -W 04:00 -n $reserve -R "span[ptile=24]" -R "rusage[mem=$mem]" -R "select[model==$model]" -o "output_$ranks.txt" mpirun -n $ranks ../../bin/lu-scalapack $runs $N
        fi
    fi

    if [ "$1" = "openmp" ]
    then
        cd omp
        export OMP_NUM_THREADS=$ranks

        if [ "$ranks" -le 4 ]
        then
            bsub -W 16:00 -n $reserve -R "span[ptile=24]" -R "rusage[mem=$mem]" -R "select[model==$model]" -o "output_$ranks.txt" ../../bin/lu-omp $runs $N
        else
            bsub -W 04:00 -n $reserve -R "span[ptile=24]" -R "rusage[mem=$mem]" -R "select[model==$model]" -o "output_$ranks.txt" ../../bin/lu-omp $runs $N
        fi
    fi
    cd ..
  done
done
cd ..
