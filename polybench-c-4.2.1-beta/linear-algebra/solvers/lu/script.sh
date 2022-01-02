for N in {1..25}
do
  export OMP_NUM_THREADS=32
  bsub -n 32 -W 10:00 -R "span[ptile=32]" -R "select[model==XeonGold_6150]" ./lu
done

