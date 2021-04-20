#!/usr/bin/sh
cd build
make -j4

if [ "$1" != "" ]; then
    perf record -F 97 --call-graph dwarf "$@"
else
    perf record -F 97 --call-graph dwarf ./robustness -i ../instances/watts_strogatz_1000_7_0.3.nkb -j 1 -a6 -eps2 0.9 -eps 0.99 -k 30 > foo.txt

fi

cd ../FlameGraph
cp ../build/perf.data .
perf script | ./stackcollapse-perf.pl |./flamegraph.pl > perf.svg
firefox perf.svg
