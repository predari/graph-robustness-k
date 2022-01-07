# Graph Robustness

## Installation and Building


    git clone git@gitlab.informatik.hu-berlin.de:macsy/code-student/goerg-graph-robustness.git

    cd goerg-graph-robustness

Initialize submodules. Access to the separate NetworKit fork is at https://gitlab.informatik.hu-berlin.de/goergmat/networkit-robustness is required.

    git submodule update --init --recursive

Build

    mkdir build
    cd build
    cmake .. -DNETWORKIT_WITH_SANITIZERS=address
    make

Download Instances

    cd ..
    python3 load_instances.py


## Usage

Example. Run main algorithm, k=20, ε=0.9, ε_UST = 10, using LAMG, 6 threads. 

    cd build
    ./robustness -a6 -i ../instances/barabasi_albert_2_100_2.nkb -k 20 -eps 0.9 -eps2 10 -lamg -j 6


For more details see help string

    ./robustness --help
