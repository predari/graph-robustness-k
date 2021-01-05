#!/usr/bin/sh
cd build
make -j4

if [ "$1" != "" ]; then
    perf record -F 97 --call-graph dwarf "$@"
else
    perf record -F 97 --call-graph dwarf ./robustness -k 4.0 -km linear -i ../instances/WattsStrogatz_1000_7_0.3.gml -a4 -h2
fi

cd ../FlameGraph
cp ../build/perf.data .
perf script | ./stackcollapse-perf.pl |./flamegraph.pl > perf.svg
firefox perf.svg