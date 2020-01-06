import tensorflow as tf
import math
import sys
import os
from timeit import default_timer as timer
from functools import reduce

class Param:
    def __init__(self, beta = 6.0, lat = [64, 64], tau = 2.0, nstep = 50, ntraj = 256, nrun = 4, nprint = 256, seed = 11*13, randinit = False, nth = int(os.environ.get('OMP_NUM_THREADS', '2')), nth_interop = 2):
        self.beta = beta
        self.lat = lat
        self.nd = len(lat)
        self.volume = reduce(lambda x,y:x*y, lat)
        self.tau = tau
        self.nstep = nstep
        self.dt = self.tau / self.nstep
        self.ntraj = ntraj
        self.nrun = nrun
        self.nprint = nprint
        self.seed = seed
        self.randinit = randinit
        self.nth = nth
        self.nth_interop = nth_interop
    def initializer(self):
        if self.randinit:
            return tf.random.uniform([param.nd] + param.lat, minval=-math.pi, maxval=math.pi, dtype=tf.float64)
        else:
            return tf.zeros([param.nd] + param.lat, dtype=tf.float64)
    def summary(self):
        return f"""latsize = {self.lat}
volume = {self.volume}
beta = {self.beta}
trajs = {self.ntraj}
tau = {self.tau}
steps = {self.nstep}
seed = {self.seed}
nth = {self.nth}
nth_interop = {self.nth_interop}
"""
    def uniquestr(self):
        lat = ".".join(str(x) for x in self.lat)
        return f"out_l{lat}_b{param.beta}_n{param.ntraj}_t{param.tau}_s{param.nstep}.out"

def action(param, f):
    return (-param.beta)*tf.reduce_sum(tf.cos(plaqphase(f)))
def force(param, f):
    with tf.GradientTape() as g:
        g.watch(f)
        s = action(param, f)
    return g.gradient(s, f)

plaqphase = lambda f: f[0,:] - f[1,:] - tf.roll(f[0,:], shift=-1, axis=1) + tf.roll(f[1,:], shift=-1, axis=0)
topocharge = lambda f: tf.floor(0.1 + tf.reduce_sum(regularize(plaqphase(f))) / (2*math.pi))
def regularize(f):
    p2 = 2*math.pi
    f_ = f - math.pi
    return p2*(f_/p2 - tf.math.floordiv(f_, p2) - 0.5)  # TODO: use math.floormod

def leapfrog(param, x, p):
    dt = param.dt
    x_ = x + 0.5*dt*p
    p_ = p + (-dt)*force(param, x_)
    for i in range(param.nstep-1):
        x_ = x_ + dt*p_
        p_ = p_ + (-dt)*force(param, x_)
    x_ = x_ + 0.5*dt*p_
    return (x_, p_)
@tf.function
def hmc(param, x):
    p = tf.random.normal(tf.shape(x), dtype=tf.float64)
    act0 = action(param, x) + 0.5*tf.reduce_sum(p*p)
    x_, p_ = leapfrog(param, x, p)
    xr = regularize(x_)
    act = action(param, xr) + 0.5*tf.reduce_sum(p_*p_)
    prob = tf.random.uniform([], dtype=tf.float64)
    dH = act-act0
    exp_mdH = tf.exp(-dH)
    acc = tf.less(prob, exp_mdH)
    newx = tf.cond(acc, lambda: xr, lambda: x)
    return (dH, exp_mdH, acc, newx)

put = lambda s: sys.stdout.write(s)

if __name__ == '__main__':
    param = Param(
        beta = 6.0, # 4.0
        lat = [64, 64], # [8, 8]
        tau = 2, # 0.3
        nstep = 50, # 3
        ntraj = 256, # 2**16 # 2**10 # 2**15
        nprint = 256,
        seed = 1331)

    tf.random.set_seed(param.seed)

    tf.config.set_soft_device_placement(True)
    tf.config.optimizer.set_jit(True)
    tf.config.threading.set_inter_op_parallelism_threads(param.nth_interop)  # ALCF suggests number of socket
    tf.config.threading.set_intra_op_parallelism_threads(param.nth)  # ALCF suggests number of physical cores
    os.environ["OMP_NUM_THREADS"] = str(param.nth)
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

    def run(param):
        with open(param.uniquestr(), "w") as O:
            params = param.summary()
            O.write(params)
            put(params)
            field = param.initializer() # mu, x, t
            plaq, topo = (action(param, field) / (-param.beta*param.volume), topocharge(field))
            status = f"Initial configuration:  plaq: {plaq}  topo: {topo}\n"
            O.write(status)
            put(status)
            ts = []
            for n in range(param.nrun):
                t = -timer()
                for i in range(param.ntraj):
                    dH, exp_mdH, acc, field = hmc(param, field)
                    plaq = action(param, field) / (-param.beta*param.volume)
                    topo = topocharge(field)
                    ifacc = "ACCEPT" if acc else "REJECT"
                    status = f"Traj: {n*param.ntraj+i+1:4}  {ifacc}:  dH: {dH:< 12.8}  exp(-dH): {exp_mdH:< 12.8}  plaq: {plaq:< 12.8}  topo: {topo:< 3.3}\n"
                    O.write(status)
                    if (i+1) % (param.ntraj//param.nprint) == 0:
                        put(status)
                t += timer()
                ts.append(t)
            print("Run times: ", ts)
            print("Per trajectory: ", [t/param.ntraj for t in ts])
    run(param)
