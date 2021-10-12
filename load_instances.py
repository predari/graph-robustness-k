
import networkit as nk  
import os
import os.path


instance_str = """\
instdir: "instances"
instances:
  - repo: local
    items: """


def dl(url, name):
    return os.system("""\
if [ -f "/tmp/{1}" ]; then
    echo "{1} already downloaded."
    false
else
    wget {0} -O /tmp/{1}
fi""".format(url, name))

def graph_inst(g, name):
    global instance_str
    instance_str += "\n      - " + name + ".nkb"
    _g = nk.components.ConnectedComponents.extractLargestConnectedComponent(g, True)
    _g.removeMultiEdges()
    _g.sortEdges()
    nk.graphio.NetworkitBinaryWriter().write(_g, "instances/"+name+".nkb")


def txt_to_inst(name, sep=" ", first_node=0, comment_prefix="#", continuous=True):
    g = nk.graphio.EdgeListReader(sep, first_node, comment_prefix, continuous).read("instances/"+name+".txt")
    graph_inst(g, name)


def dl_txt_gz(url, name):
    if os.path.isfile("instances/{0}.nkb".format(name)):
        return False
    dl(url, name)
    error = os.system("""\
gunzip /tmp/{1} -c > instances/{1}.txt""".format(url, name))

def csv_to_inst(in_path, name):
    os.system("tail -n +2 {0} | sed -e 's/,/ /g' > {1}".format(in_path, "instances/" + name + ".txt"))
    txt_to_inst(name)

def dl_tar_bz2(url, archive_path_to_instance, name):
    if os.path.isfile("instances/{0}.nkb".format(name)):
        return False
    dl(url, name+".tar.bz2")
    os.system('tar -xf /tmp/{0}.tar.bz2 -C /tmp/'.format(name))
    os.system('mv /tmp/{0} instances/{1}.txt'.format(archive_path_to_instance, name))

def load_deezer_europe_instance():
    if os.path.isfile("instances/deezer_europe.nkb"):
        return False
    dl("https://snap.stanford.edu/data/deezer_europe.zip", "deezer_europe.zip")
    os.system("unzip /tmp/deezer_europe.zip -d /tmp/")
    csv_to_inst("/tmp/deezer_europe/deezer_europe_edges.csv", "deezer_europe")





def gen_er_inst(n, p):
    nk.setSeed(1, True)
    g = nk.generators.ErdosRenyiGenerator(n, p).generate()
    name = "erdos_renyi_{0}_{1}".format(n, p)
    graph_inst(g, name)

def gen_ba_inst(k, nMax, n0):
    nk.setSeed(1, True)
    g = nk.generators.BarabasiAlbertGenerator(k, nMax, n0).generate()
    name = "barabasi_albert_{0}_{1}_{2}".format(k, nMax, n0)
    graph_inst(g, name)

def gen_ws_inst(nNodes, nNeighbors, p):
    nk.setSeed(1, True)
    g = nk.generators.WattsStrogatzGenerator(nNodes, nNeighbors, p).generate()
    name = "watts_strogatz_{0}_{1}_{2}".format(nNodes, nNeighbors, p)
    graph_inst(g, name)

    

if __name__ == "__main__":
    if not os.path.isdir("instances"):
        os.system("mkdir instances")

    dl_txt_gz("https://snap.stanford.edu/data/facebook_combined.txt.gz", "facebook_ego_combined")
    txt_to_inst("facebook_ego_combined")
    #dl_txt_gz("https://snap.stanford.edu/data/ca-AstroPh.txt.gz", "arxiv-astro-ph")
    #txt_to_inst("arxiv-astro-ph", "\t", 0, "#", False)
    #dl_txt_gz("https://snap.stanford.edu/data/ca-CondMat.txt.gz", "arxiv-condmat")
    #txt_to_inst("arxiv-condmat", "\t", 0, "#", False)
    dl_txt_gz("https://snap.stanford.edu/data/ca-GrQc.txt.gz", "arxiv-grqc")
    txt_to_inst("arxiv-grqc", "\t", 0, "#", False)
    dl_txt_gz("https://snap.stanford.edu/data/ca-HepPh.txt.gz", "arxiv-heph")
    txt_to_inst("arxiv-heph", "\t", 0, "#", False)
    dl_txt_gz("https://snap.stanford.edu/data/ca-HepTh.txt.gz", "arxiv-hephth")
    txt_to_inst("arxiv-hephth", "\t", 0, "#", False)

    


    if not all(os.path.isfile(p) for p in ["instances/twitch_de.nkb", "instances/twitch_engb.nkb"]):
        dl("https://snap.stanford.edu/data/twitch.zip", "twitch.zip")
        os.system("unzip -u /tmp/twitch.zip -d /tmp/")
        csv_to_inst("/tmp/twitch/DE/musae_DE_edges.csv", "twitch_de")
        csv_to_inst("/tmp/twitch/ENGB/musae_ENGB_edges.csv", "twitch_engb")
    
    dl_tar_bz2("https://konect.cc/files/download.tsv.opsahl-powergrid.tar.bz2", "opsahl-powergrid/out.opsahl-powergrid", "opsahl-powergrid")
    txt_to_inst("opsahl-powergrid", " ", 1, "%")
    
    dl_tar_bz2("http://konect.cc/files/download.tsv.flickrEdges.tar.bz2", "flickrEdges/out.flickrEdges", "flickr")
    txt_to_inst("flickr", " ", 1, "%")

#    dl_tar_bz2("http://konect.cc/files/download.tsv.facebook-wosn-links.tar.bz2", "facebook-wosn-links/out.facebook-wosn-links", "facebook-wosn-links")
#    txt_to_inst("facebook-wosn-links", " ", 1, "%")

    dl_tar_bz2("http://konect.cc/files/download.tsv.loc-brightkite_edges.tar.bz2", "loc-brightkite_edges/out.loc-brightkite_edges", "loc-brightkite_edges")
    txt_to_inst("loc-brightkite_edges", " ", 1, "%")


    #dl_tar_bz2("http://konect.cc/files/download.tsv.marvel.tar.bz2", "marvel/out.marvel", "marvel")
    #txt_to_inst("marvel", "\t", 1, "%")
    #dl_tar_bz2("http://konect.cc/files/download.tsv.dimacs10-as-22july06.tar.bz2", "dimacs10-as-22july06/out.dimacs10-as-22july06", "dimacs-net")
    #txt_to_inst("dimacs-net", "\t", 1, "%")
    #dl_tar_bz2("http://konect.cc/files/download.tsv.topology.tar.bz2", "topology/out.topology", "topology")
    #txt_to_inst("topology", "\t", 1, "%")

    load_deezer_europe_instance()

    #gen_er_inst(10, 0.4)
    #gen_er_inst(30, 0.3)
    gen_er_inst(128, 0.1)
    gen_er_inst(300, 0.05)
    gen_er_inst(600, 0.05)
    gen_er_inst(1000, 0.02)
    gen_er_inst(3000, 0.01)
    gen_er_inst(10000, 0.001)

    #gen_ws_inst(10, 3, 0.4)
    #gen_ws_inst(30, 5, 0.4)
    gen_ws_inst(100, 5, 0.5)
    gen_ws_inst(300, 7, 0.5)
    gen_ws_inst(1000, 7, 0.3)
    gen_ws_inst(3000, 7, 0.3)
    gen_ws_inst(10000, 7, 0.2)

    gen_ba_inst(2, 100, 2)
    gen_ba_inst(2, 300, 2)
    gen_ba_inst(2, 1000, 2)
    gen_ba_inst(2, 3000, 2)

    print(instance_str)