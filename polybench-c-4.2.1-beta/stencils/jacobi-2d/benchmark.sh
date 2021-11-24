model='XeonGold_5118'

export I_MPI_STATS=4,ipm

mkdir data_5118
cd data_5118
for nodes in $(seq 1 2)
do
  if [ $nodes -eq 1 ]
  then 
    mem='2GB'
  else
    mem='1GB'
  fi
  let reserve=nodes*24
  let reserve_low=reserve-23
  for ranks in $(seq $reserve_low $reserve)
  do
    mkdir $ranks
    cd $ranks
    bsub -We 01:00 -n $reserve -R "span[ptile=24]" -R fullnode -R "rusage[mem=$mem]" -R "select[model==$model]" -o "output.txt" mpirun -n $ranks ../../../pagerank_parallel $graph
    cd ..
  done
done
cd ..