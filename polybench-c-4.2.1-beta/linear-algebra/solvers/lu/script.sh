for N in {1..25}
do
  export OMP_NUM_THREADS=2
  bsub -n 2 -W 04:00 -R "span[ptile=2]" -R "select[model==XeonGold_6150]" ./lu
done

