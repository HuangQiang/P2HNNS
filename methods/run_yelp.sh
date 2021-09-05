#!/bin/bash
make clean
make -j

# ------------------------------------------------------------------------------
#  Parameters for Point-to-Hyperplane Nearest Neighbor Search
# ------------------------------------------------------------------------------
dname=Yelp
n=77079
d=51
qn=100
b=0.9
cf=config
dtype=float32

I_list=(0 1)
oFolder_list=(results results_normalized)
dFolder_list=(bin bin_normalized)
length=`expr ${#dFolder_list[*]} - 1`

for j in $(seq 0 ${length})
do
  I=${I_list[j]}
  dPath=../data/${dFolder_list[j]}/${dname}/${dname}
  oPath=../${oFolder_list[j]}/${dname}

  # ----------------------------------------------------------------------------
  # Ground Truth
  ./p2h -alg 0 -n ${n} -qn ${qn} -d ${d} -dt ${dtype} -dn ${dname} \
    -ds ${dPath}.ds -qs ${dPath}.q -ts ${dPath}.gt -op ${oPath}

  # Linear-Scan
  ./p2h -alg 1 -n ${n} -qn ${qn} -d ${d} -dt ${dtype} -dn ${dname} \
    -ds ${dPath}.ds -qs ${dPath}.q -ts ${dPath}.gt -op ${oPath}

  # # Random-Scan
  # ./p2h -alg 2 -n ${n} -qn ${qn} -d ${d} -cf ${cf} -dt ${dtype} -dn ${dname} \
  #   -ds ${dPath}.ds -qs ${dPath}.q -ts ${dPath}.gt -op ${oPath}

  # # Sorted-Scan
  # ./p2h -alg 3 -n ${n} -qn ${qn} -d ${d} -cf ${cf} -dt ${dtype} -dn ${dname} \
  #   -ds ${dPath}.ds -qs ${dPath}.q -ts ${dPath}.gt -op ${oPath}

  # # ----------------------------------------------------------------------------
  # # # EH (Embedding Hyperplane Hash)
  # # for l in 8 16 32 64 128 256
  # # do
  # #   for m in 2 4 6 8 10
  # #   do
  # #     ./p2h -alg 4 -n ${n} -qn ${qn} -d ${d} -I ${I} -m ${m} -l ${l} -b ${b} \
  # #       -cf ${cf} -dt ${dtype} -dn ${dname} -ds ${dPath}.ds -qs ${dPath}.q \
  # #       -ts ${dPath}.gt -op ${oPath}
  # #   done
  # # done

  # # BH (Bilinear Hyperplane Hash)
  # for l in 8 16 32 64 128 256
  # do
  #   for m in 2 4 6 8 10
  #   do
  #     ./p2h -alg 5 -n ${n} -qn ${qn} -d ${d} -I ${I} -m ${m} -l ${l} -b ${b} \
  #       -cf ${cf} -dt ${dtype} -dn ${dname} -ds ${dPath}.ds -qs ${dPath}.q \
  #       -ts ${dPath}.gt -op ${oPath}
  #   done
  # done

  # # MH (Multilinear Hyperplane Hash)
  # for l in 8 16 32 64 128 256
  # do
  #   for m in 2 4 6 8 10
  #   do
  #     for M in 4 8 16
  #     do
  #       ./p2h -alg 6 -n ${n} -qn ${qn} -d ${d} -I ${I} -m ${m} -l ${l} -M ${M} \
  #         -b ${b} -cf ${cf} -dt ${dtype} -dn ${dname} -ds ${dPath}.ds \
  #         -qs ${dPath}.q -ts ${dPath}.gt -op ${oPath}
  #     done
  #   done
  # done

  # # ----------------------------------------------------------------------------
  # FH (Furthest Hyperpalne Hash)
  # for m in 8 16 32 64 128 256
  # do
  #   for s in 1 2 4 8
  #   do
  #     ./p2h -alg 7 -n ${n} -qn ${qn} -d ${d} -m ${m} -s ${s} -b ${b} -cf ${cf} \
  #       -dt ${dtype} -dn ${dname} -ds ${dPath}.ds -qs ${dPath}.q \
  #       -ts ${dPath}.gt -op ${oPath}
  #   done
  # done

  # # FH^- (Furthest Hyperpalne Hash without Data-Dependent Multi-Partitioning)
  # for m in 8 16 32 64 128 256
  # do
  #   for s in 1 2 4 8
  #   do
  #     ./p2h -alg 8 -n ${n} -qn ${qn} -d ${d} -m ${m} -s ${s} -cf ${cf} \
  #       -dt ${dtype} -dn ${dname} -ds ${dPath}.ds -qs ${dPath}.q \
  #       -ts ${dPath}.gt -op ${oPath}
  #   done
  # done

  # # NH (Nearest Hyperpalne Hash with LCCS-LSH)
  # w=0.1
  # for m in 8 16 32 64 128 256
  # do
  #   for s in 1 2 4 8
  #   do
  #     ./p2h -alg 9 -n ${n} -qn ${qn} -d ${d} -m ${m} -s ${s} -w ${w} -cf ${cf} \
  #       -dt ${dtype} -dn ${dname} -ds ${dPath}.ds -qs ${dPath}.q \
  #       -ts ${dPath}.gt -op ${oPath}
  #   done
  # done

  # # ----------------------------------------------------------------------------
  # # FH_wo_S (FH without Sampling)
  # for m in 8 16 32 64 128 256
  # do
  #   ./p2h -alg 10 -n ${n} -qn ${qn} -d ${d} -m ${m} -b ${b} -cf ${cf} \
  #     -dt ${dtype} -dn ${dname} -ds ${dPath}.ds -qs ${dPath}.q -ts ${dPath}.gt \
  #     -op ${oPath}
  # done

  # # FH^-_wo_S (FH^- without Sampling)
  # for m in 8 16 32 64 128 256
  # do
  #   ./p2h -alg 11 -n ${n} -qn ${qn} -d ${d} -m ${m} -cf ${cf} \
  #     -dt ${dtype} -dn ${dname} -ds ${dPath}.ds -qs ${dPath}.q -ts ${dPath}.gt \
  #     -op ${oPath}
  # done

  # # NH_wo_S (NH without Sampling)
  # w=1.0
  # for m in 8 16 32 64 128 256
  # do
  #   ./p2h -alg 12 -n ${n} -qn ${qn} -d ${d} -m ${m} -w ${w} -cf ${cf} \
  #     -dt ${dtype} -dn ${dname} -ds ${dPath}.ds -qs ${dPath}.q -ts ${dPath}.gt \
  #     -op ${oPath}
  # done
done
