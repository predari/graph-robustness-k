#!/usr/bin/python




import networkit as nk

seed = 1
def addER(n, p):
    nk.setSeed(seed, True)
    g = nk.generators.ErdosRenyiGenerator(n, p).generate()
    writer = nk.graphio.GMLGraphWriter()
    writer.write(g, "/dev/stdout")

def addBA(k, nMax, n0):
    nk.setSeed(seed, True)
    g = nk.generators.BarabasiAlbertGenerator(k, nMax, n0).generate()
    writer = nk.graphio.GMLGraphWriter()
    writer.write(g, "/dev/stdout")

def addWS(nNodes, nNeighbors, p):
    nk.setSeed(seed, True)
    g = nk.generators.WattsStrogatzGenerator(nNodes, nNeighbors, p).generate()
    writer = nk.graphio.GMLGraphWriter()
    writer.write(g, "/dev/stdout")


#def writeER(n, p):
#    print("  - generator:\n      args: ['./generator.py', '-er', '{0}', '{1}']\n    items: ErdosRenyi_{0}_{1}.gml".format(n, p))
#def writeWS(a, b, p):
#    print("  - generator:\n      args: ['./generator.py', '-ws', '{0}', '{1}', '{2}']\n    items: WattsStrogatz_{0}_{1}_{2}.gml".format(a, b, p))
#def writeBA(a, b, p):
#    print("  - generator:\n      args: ['./generator.py', '-ba', '{0}', '{1}']\n    items: BarabasiAlbert_{0}_{1}_{2}.gml".format(n, p))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Error: Graph Generator needs more arguments.")
    if (sys.argv[1] == "-er"):
        n = int(sys.argv[2])
        p = float(sys.argv[3])
        addER(n, p)
    elif (sys.argv[1] == "-ba"):
        k = int(sys.argv[2])
        nMax = int(sys.argv[3])
        n0 = int(sys.argv[4])
        addBA(k, nMax, n0)
    elif (sys.argv[1] == "-ws"):
        nNodes = int(sys.argv[2])
        nNeighbors = int(sys.argv[3])
        p = float(sys.argv[4])
        addWS(nNodes, nNeighbors, p)


    if (sys.argv[1] == "-we"):
        def writeER(n, p):
            print("  - generator:\n      args: ['./generator.py', '-er', '{0}', '{1}']\n    items:  \n      - ErdosRenyi_{0}_{1}.gml".format(n, p))
        def writeWS(a, b, p):
            print("  - generator:\n      args: ['./generator.py', '-ws', '{0}', '{1}', '{2}']\n    items:  \n      - WattsStrogatz_{0}_{1}_{2}.gml".format(a, b, p))
        def writeBA(a, b, p):
            print("  - generator:\n      args: ['./generator.py', '-ba', '{0}', '{1}', '{2}']\n    items: \n      - BarabasiAlbert_{0}_{1}_{2}.gml".format(a, b, p))

    #writeER(10, 0.4)
    #writeER(30, 0.3)
    #writeER(128, 0.1)
    #writeER(300, 0.05)
    #writeER(600, 0.05)
    #writeER(1000, 0.02)
    #writeER(3000, 0.01)

    #writeWS(10, 3, 0.4)
    #writeWS(30, 5, 0.4)
    #writeWS(100, 5, 0.5)
    #writeWS(300, 7, 0.5)
    #writeWS(1000, 7, 0.3)

    #writeBA(2, 128, 2)
    #writeBA(2, 1000, 2)
