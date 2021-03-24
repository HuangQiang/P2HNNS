#!/bin/bash
g++ -std=c++11 -w -O3 -o generate generate.cc 
qn=100

# ------------------------------------------------------------------------------
#  Yelp
# ------------------------------------------------------------------------------
n=77079
d=50
dname=Yelp
in=${dname}.bin
bin=../bin/${dname}/${dname}
bin_normalized=../bin_normalized/${dname}/${dname}

./generate ${n} ${d} ${qn} 0 ${in} ${bin}
./generate ${n} ${d} ${qn} 1 ${in} ${bin_normalized}

# ------------------------------------------------------------------------------
#  Music
# ------------------------------------------------------------------------------
n=1000000
d=100
dname=Music
in=${dname}.bin
bin=../bin/${dname}/${dname}
bin_normalized=../bin_normalized/${dname}/${dname}

./generate ${n} ${d} ${qn} 0 ${in} ${bin}
./generate ${n} ${d} ${qn} 1 ${in} ${bin_normalized}

# ------------------------------------------------------------------------------
#  GloVe100
# ------------------------------------------------------------------------------
n=1183514
d=100
dname=GloVe100
in=${dname}.bin
bin=../bin/${dname}/${dname}
bin_normalized=../bin_normalized/${dname}/${dname}

./generate ${n} ${d} ${qn} 0 ${in} ${bin}
./generate ${n} ${d} ${qn} 1 ${in} ${bin_normalized}

# ------------------------------------------------------------------------------
#  Tiny1M
# ------------------------------------------------------------------------------
n=1000000
d=384
dname=Tiny1M
in=${dname}.bin
bin=../bin/${dname}/${dname}
bin_normalized=../bin_normalized/${dname}/${dname}

./generate ${n} ${d} ${qn} 0 ${in} ${bin}
./generate ${n} ${d} ${qn} 1 ${in} ${bin_normalized}

# ------------------------------------------------------------------------------
#  Msong
# ------------------------------------------------------------------------------
n=992272
d=420
dname=Msong
in=${dname}.bin
bin=../bin/${dname}/${dname}
bin_normalized=../bin_normalized/${dname}/${dname}

./generate ${n} ${d} ${qn} 0 ${in} ${bin}
./generate ${n} ${d} ${qn} 1 ${in} ${bin_normalized}
