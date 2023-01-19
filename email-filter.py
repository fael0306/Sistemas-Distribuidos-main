import sys, os, time, math
from pyDF import*
sys.path.append(os.environ["PYDFHOME"])

def readFile(args):
    filename = args[0]
    f = open(filename, "r")
    vetor = []
    for linha in f:
        vetor.append(linha)
    f.close()
    return nprocs, vetor

def filterMails(args):
    sp1 = args
    sp = sp1[0].split("@")
    if len(sp) == 2:
        aux = sp[1][:-1]
        if (aux == "outlook.com"):
            ret = sp1[0][:-1]
        else:
            ret = ""
    else:
        ret = ""
    return ret

def printMails(args):
    if args[0] != "":
        print args[0]

nprocs = int(sys.argv[1])
filename = sys.argv[2]
graph = DFGraph()
sched = Scheduler(graph, nprocs, mpi_enabled = False)
fp = open(filename, "r")
src = Source(fp)
graph.add(src)
nd = FilterTagged(filterMails, 1)
graph.add(nd)
ser = Serializer(printMails, 1)
graph.add(ser)
src.add_edge(nd, 0)
nd.add_edge(ser, 0)
t0 = time.time()
sched.start()
t1 = time.time()

print "Execution time %.3f" %(t1-t0)
