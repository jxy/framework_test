import tensorflow as tf
import math
import sys
import os
from timeit import default_timer as timer
from functools import reduce
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

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
            return tf.compat.v1.random_uniform_initializer(-math.pi, math.pi)
        else:
            return tf.compat.v1.zeros_initializer
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
    return (-param.beta)*tf.reduce_sum(input_tensor=tf.cos(plaqphase(f)))
def force(param, f):
    s = action(param, f)
    return tf.gradients(ys=s, xs=f)[0]

plaqphase = lambda f: f[0,:] - f[1,:] - tf.roll(f[0,:], shift=-1, axis=1) + tf.roll(f[1,:], shift=-1, axis=0)
topocharge = lambda f: tf.floor(0.1 + tf.reduce_sum(input_tensor=regularize(plaqphase(f))) / (2*math.pi))
def regularize(f):
    p2 = 2*math.pi
    f_ = f - math.pi
    return p2*(f_/p2 - tf.math.floordiv(f_, p2) - 0.5)

def leapfrog(param, x, p):
    dt = param.dt
    x_ = x + 0.5*dt*p
    p_ = p + (-dt)*force(param, x_)
    def body(i, xx, pp):
        xx_ = xx + dt*pp
        pp_ = pp + (-dt)*force(param, xx_)
        return (i+1, xx_, pp_)
    _, x__, pp = tf.while_loop(
        cond=lambda i, xx, pp: i < param.nstep-1,
        body=body,
        loop_vars=(0, x_, p_))
    xx = x__ + 0.5*dt*pp
    return (xx, pp)
def hmc(param, x):
    p = tf.random.normal(tf.shape(input=x), dtype=tf.float64)
    act0 = action(param, x) + 0.5*tf.reduce_sum(input_tensor=p*p)
    x_, p_ = leapfrog(param, x, p)
    xr = regularize(x_)
    act = action(param, xr) + 0.5*tf.reduce_sum(input_tensor=p_*p_)
    prob = tf.random.uniform([], dtype=tf.float64)
    dH = act-act0
    exp_mdH = tf.exp(-dH)
    acc = tf.less(prob, exp_mdH)
    newx = tf.cond(pred=acc, true_fn=lambda: xr, false_fn=lambda: x)
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

    tf.compat.v1.set_random_seed(param.seed)

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=param.nth,  # ALCF suggests number of physical cores
        inter_op_parallelism_threads=param.nth_interop,  # ALCF suggests number of socket
        allow_soft_placement = True)
    os.environ["OMP_NUM_THREADS"] = str(param.nth)
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

    field = tf.compat.v1.get_variable("field", [param.nd] + param.lat, dtype=tf.float64, initializer=param.initializer()) # mu, x, t
    dH, exp_mdH, acc, field_ = hmc(param, field)
    plaq = action(param, field_) / (-param.beta*param.volume)
    topo = topocharge(field_)
    update = field.assign(field_)

    def run(param):
        with open(param.uniquestr(), "w") as O:
            params = param.summary()
            O.write(params)
            put(params)
            with tf.compat.v1.Session(config=config) as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                plaq_, topo_ = sess.run((action(param, field) / (-param.beta*param.volume), topocharge(field)))
                status = f"Initial configuration:  plaq: {plaq_}  topo: {topo_}\n"
                O.write(status)
                put(status)
                ts = []
                for n in range(param.nrun):
                    t = -timer()
                    for i in range(param.ntraj):
                        _, dH_, exp_mdH_, acc_, plaq_, topo_ = sess.run((update, dH, exp_mdH, acc, plaq, topo))
                        ifacc = "ACCEPT" if acc_ else "REJECT"
                        status = f"Traj: {n*param.ntraj+i+1:4}  {ifacc}:  dH: {dH_:< 12.8}  exp(-dH): {exp_mdH_:< 12.8}  plaq: {plaq_:< 12.8}  topo: {topo_:< 3.3}\n"
                        O.write(status)
                        if (i+1) % (param.ntraj//param.nprint) == 0:
                            put(status)
                    t += timer()
                    ts.append(t)
                print("Run times: ", ts)
                print("Per trajectory: ", [t/param.ntraj for t in ts])
    run(param)
