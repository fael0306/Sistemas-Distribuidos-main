from multiprocessing import Process, Queue, Pipe
from threading import Thread
from mpi4py import MPI

class Worker(Process):
    def __init__(self, graph, operand_queue, conn, worker_id):
        Process.__init__(self)
        self.oper_queue = operand_queue
        self.graph = graph
        self.worker_id = worker_id
        self.conn = conn

    def run(self):
        self.oper_queue.put(Oper(self.worker_id, None, None, None))  # Request a task to start

        while True:
            task = self.conn.recv()
            node = self.graph.nodes[task.node_id]
            node.run(task.args, self.worker_id, self.oper_queue)

class Task:
    def __init__(self, f, node_id, args=None):
        self.node_id = node_id
        self.args = args

class DFGraph:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)
        node.id = len(self.nodes) - 1

class Node:
    def __init__(self, f, input_count):
        self.f = f
        self.in_ports = [[] for _ in range(input_count)]
        self.dsts = []
        self.affinity = None

    def add_edge(self, dst, dst_port):
        self.dsts.append((dst.id, dst_port))

    def pin(self, worker_id):
        self.affinity = worker_id

    def run(self, args, worker_id, oper_queue):
        if len(self.in_ports) == 0:
            opers = self.create_oper(self.f(), worker_id)
        else:
            opers = self.create_oper(self.f([a.val for a in args]), worker_id)
        self.send_ops(opers, oper_queue)

    def send_ops(self, opers, oper_queue):
        oper_queue.put(opers)

    def create_oper(self, value, worker_id):
        opers = []
        if not self.dsts:
            opers.append(Oper(worker_id, None, None, None))
        else:
            for (dst_id, dst_port) in self.dsts:
                oper = Oper(worker_id, dst_id, dst_port, value)
                opers.append(oper)
        return opers

    def match(self):
        args = []
        for port in self.in_ports:
            if port:
                arg = port[0]
                args.append(arg)
                port.remove(arg)
        if len(args) == len(self.in_ports):
            return args
        else:
            return None

class Oper:
    def __init__(self, prod_id, dst_id, dst_port, val):
        self.worker_id = prod_id
        self.dst_id = dst_id
        self.dst_port = dst_port
        self.val = val
        self.request_task = True

class Scheduler:
    TASK_TAG = 0
    TERMINATE_TAG = 1

    def __init__(self, graph, n_workers=1, mpi_enabled=True):
        self.oper_queue = Queue()
        self.graph = graph
        self.tasks = []
        self.worker_conns = []
        self.conn = []
        self.waiting = []
        self.n_workers = n_workers

        if mpi_enabled:
            self.mpi_rank = MPI.COMM_WORLD.Get_rank()
            self.mpi_size = MPI.COMM_WORLD.Get_size()
            self.n_slaves = self.mpi_size - 1
        else:
            self.mpi_rank = None

    def mpi_handle(self):
        if self.mpi_rank == 0:
            self.mpi_master()
        else:
            self.mpi_slave()

    def mpi_master(self):
        slaves = []
        for i in range(1, self.mpi_size):
            slaves.append(i)

        while self.tasks or self.waiting:
            status = MPI.Status()
            MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            slave = status.Get_source()

            if len(self.tasks) > 0:
                task = self.tasks.pop(0)
                self.conn[slave].send(task)
            else:
                self.waiting.append(slave)

        for slave in slaves:
            self.conn[slave].send(None)

    def mpi_slave(self):
        worker_id = self.mpi_rank - 1
        worker = Worker(self.graph, self.oper_queue, self.conn[worker_id], worker_id)
        worker.start()

        while True:
            oper = self.oper_queue.get()
            if oper.request_task:
                if len(self.waiting) > 0:
                    self.conn[worker_id].send(self.waiting.pop(0))
                else:
                    self.tasks.append(Task(None, None))
            elif oper.dst_id is None:
                break
            else:
                dst_conn = self.conn[oper.dst_id - 1]
                dst_conn.send(oper)

    def run(self):
        if self.n_workers > 1:
            self.run_multiprocessing()
        else:
            self.run_single_thread()

    def run_multiprocessing(self):
        for _ in range(self.n_workers):
            parent_conn, child_conn = Pipe()
            self.worker_conns.append(child_conn)
            self.conn.append(parent_conn)

        for i, conn in enumerate(self.worker_conns):
            worker = Worker(self.graph, self.oper_queue, conn, i)
            worker.start()

        for task in self.tasks:
            self.worker_conns[task.node_id].send(task)

        for _ in range(self.n_workers):
            self.oper_queue.get()

        for conn in self.worker_conns:
            conn.send(Oper(None, None, None, None))

        for worker in self.worker_conns:
            worker.join()

    def run_single_thread(self):
        worker = Worker(self.graph, self.oper_queue, None, 0)
        worker.run()

if __name__ == '__main__':
    graph = DFGraph()

    # Create nodes and add them to the graph
    node1 = Node(lambda: 1, 0)
    node2 = Node(lambda: 2, 0)
    node3 = Node(lambda values: sum(values), 2)

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    # Add edges between nodes
    node1.add_edge(node3, 0)
    node2.add_edge(node3, 1)

    # Set node affinities (optional)
    node1.pin(0)
    node2.pin(1)
    node3.pin(0)

    scheduler = Scheduler(graph, n_workers=2)
    scheduler.run()
