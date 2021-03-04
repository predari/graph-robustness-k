#!/usr/bin/sh
cd build
make -j4

if [ "$1" != "" ]; then
    perf record -F 97 --call-graph dwarf "$@"
else
    perf record -F 97 --call-graph dwarf ./robustness -i ../instances/WattsStrogatz_100_5_0.5.gml -k 6 -a6 -tr -eps 0.2

fi

cd ../FlameGraph
cp ../build/perf.data .
perf script | ./stackcollapse-perf.pl |./flamegraph.pl > perf.svg
firefox perf.svg
