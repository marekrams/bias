import csv
import os
from pathlib import Path
import time
from tqdm import tqdm  # progressbar
import yastn
import yastn.tn.fpeps as peps
from routines import NSpin12
import numpy as np
import scipy
import ray
import glob
from itertools import product


def mean(x):
    return sum(x) / len(x)

def load_drive(E=1.0, smooth=100):
    if E == -1:
        fJ = lambda x: x * (1- np.exp(-smooth * x * x))
        fg = lambda x: 3.04438 * (1 - fJ(x))
    elif E == -2:
        fJ = lambda x: x * (1- np.exp(-32 * x))
        fg = lambda x: 3.04438 * (1 - fJ(x))
    elif E == -3:
        fJ = lambda x: 0.5 + 1.5 * (x - 0.5) - 2 * (x - 0.5) ** 3
        fg = lambda x: 3.04438 * (1 - fJ(x))
    elif E == -4:
        fJ = lambda x: x ** 2
        fg = lambda x: (1 - x) ** 2
    elif E == -5:
        fJ = lambda x: 1 - np.exp(-smooth * x * x)
        fg = lambda x: 3.04438 * 2 * (1 - x * fJ(x))
    else:
        schedule = np.loadtxt('09-1265A-E_Advantage_system5_4_annealing_schedule.csv', delimiter=",")
        fg = scipy.interpolate.UnivariateSpline(schedule[:, 0], schedule[:, 1] * np.pi * (1 - np.exp(-smooth * schedule[:, 0]**2)) + np.exp(-smooth * schedule[:, 0]**2) * 30, s=0, k=4)
        fJ = scipy.interpolate.UnivariateSpline(schedule[:, 0], schedule[:, 2] * E * np.pi * (1 - np.exp(-smooth * schedule[:, 0]**2)), s=0, k=4)
    return fg, fJ


def get_sc(E):
    gc = 3.04438
    fg, fJ = load_drive(E)
    sc = scipy.optimize.fsolve(lambda s: fg(s) - gc * fJ(s), 0.5)
    return sc[0]




def ZX(ops):
    if ops.config.sym.SYM_ID == 'Z2':
        Z = ops.x()
        X = ops.z()
        vec_x1 = ops.vec_z(val=1)
        return Z, X, vec_x1, 'XX'
    else:
        Z = ops.z()
        X = ops.x()
        vec_x1 = ops.vec_x(val=1)
        return Z, X, vec_x1, 'ZZ'



def gate_Ising_cluster(Jzz, hz, hx, step, ops, net):
    Z, X, vec_x1, base = ZX(ops)

    if hz == 0:
        Hl = -hx * X
    else:
        Hl = -hx * X - hz * Z
    Hnn = -Jzz * peps.gates.fkron(Z, Z)
    Gnn = peps.gates.gate_nn_exp(step, ops.I(), Hnn)
    Gl = peps.gates.gate_local_exp(step, ops.I(), Hl)
    return peps.gates.distribute(net, gates_local=Gl, gates_nn=Gnn)


@ray.remote(num_cpus=1)
def run_quench(hz, ta, E, D, chi, which='NN+BP', dt=0.01, fs=61, sym='dense'):
    #
    geometry = peps.CheckerboardLattice()
    #
    # Define quench protocol
    #
    fg, fJ = load_drive(E=E)
    #
    # Load operators.
    ops = yastn.operators.Spin12(sym=sym)
    Z, X, vec_x1, base = ZX(ops)
    #
    path = Path(f"./bias_dwave/{hz=:0.4f}/{ta=:0.4f}/{E=:0.4f}/{base}/")
    path.mkdir(parents=True, exist_ok=True)
    print(path)
    #
    # see if there is a state to restart from
    #
    tmp = sorted(glob.glob(str(path / f"env_{D=}_{chi=}_{which=}_*.npy")))
    if tmp:
        tmp = tmp[-1]
        try:
            d = np.load(tmp, allow_pickle=True).item()
        except:
            print(tmp)
            d = np.load(tmp, allow_pickle=True).item()
        env_ctm = peps.load_from_dict(Z.config, d)
        psi = env_ctm.psi.ket
        si = float(tmp.split("_")[-1][3:9])
    else:
        # Initialize system in the product ground state at s=0.
        si = 0
        psi = peps.product_peps(geometry=geometry, vectors=vec_x1)
    #
    # simulation parameters
    opts_svd_ntu = {"D_total": D, "D_block": D // 2} if sym == 'Z2' else {"D_total": D}
    #
    if 'BP' in which:
        env = peps.EnvBP(psi, which=which)
        env.iterate_(max_sweeps=100, diff_tol=1e-8)
    else:
        env = peps.EnvNTU(psi, which=which)
    #
    t = ta * si
    #
    sc = get_sc(E)
    ss = list(np.arange(1, np.ceil(sc * 100)) / 100) + [sc] + list(np.arange(np.ceil(sc * 100 + 0.1), fs) / 100)
    ss = [s for s in ss if s > si + 0.0001]

    infoss = []
    dtold = dt
    for sf in ss:
        print(sf)
        ds = sf - si
        si = sf
        steps = round(np.ceil(ta * ds / dtold))
        dt = ta * ds / steps
        dt2 = 1j * dt / 2

        keep_time = time.time()
        for _ in tqdm(range(steps), disable=True):
            t += dt / 2
            gates = gate_Ising_cluster(fJ(t / ta), hz * fJ(t / ta), fg(t / ta), dt2, ops, geometry)
            infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_ntu)
            if 'BP' in which:
                env.iterate_(max_sweeps=100, diff_tol=1e-8)
            infoss.append(infos)
            t += dt / 2

        Delta = peps.accumulated_truncation_error(infoss, statistics='mean')
        print(f"Accumulated mean truncation error {Delta:0.5f}")
        #
        opts_svd_env = {'D_total': chi}
        env_ctm = peps.EnvCTM(psi, init='eye')
        info = env_ctm.ctmrg_(opts_svd=opts_svd_env, max_sweeps=100, corner_tol=1e-5)
        if info.converged is False:
            env_ctm = peps.EnvCTM(psi, init='rand')
            info = env_ctm.ctmrg_(opts_svd=opts_svd_env, max_sweeps=100, corner_tol=1e-5)
        print(info)
        #
        # Calculating 1-site <X_i> for all sites
        #
        Ex = mean(env_ctm.measure_1site(X).values()).real
        print("Ex", Ex)
        #
        Ez = mean(env_ctm.measure_1site(Z).values()).real
        print("Ez", Ez)
        #
        Ezz = mean(env_ctm.measure_nn(Z, Z).values()).real
        print("Ezz", Ezz)
        Ezzs = []
        for nx in range(1, 21):
            Ezzs.append(env_ctm.measure_line(Z, Z, sites=[(0, 0), (nx, 0)]).real)
        #
        #
        fieldnames = ["D", "which", "chi", "dt", "Delta", "time", "Ezz", "Ez", "Ex"]
        out = {"D" : D, "which" : which, "chi": chi, "dt": dt, "Delta": Delta,
                "time": time.time() - keep_time, "Ezz": Ezz, "Ez": Ez, "Ex": Ex}
        #
        fname = path / f"data_{sf=:0.4f}.csv"
        file_exists = os.path.isfile(fname)
        #
        with open(fname, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
            if not file_exists:
                writer.writeheader()
            writer.writerow(out)

        fieldnames2 = ["D", "which", "chi", "dt"] + [f"Ezz_{nx}" for nx in range(1, 21)]

        out2 = {"D" : D, "which" : which, "chi": chi, "dt": dt}
        for nx, v in enumerate(Ezzs, start=1):
            out2[f"Ezz_{nx}"] = v

        fname = path / f"Ezzs_{sf=:0.4f}.csv"
        file_exists = os.path.isfile(fname)

        with open(fname, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames2, delimiter=";")
            if not file_exists:
                writer.writeheader()
            writer.writerow(out2)

        fname = path / f"env_{D=}_{chi=}_{which=}_{sf=:0.4f}.npy"
        d = env_ctm.save_to_dict()
        d['info'] = info
        I = ops.I()
        four = {}
        err = 0
        for n00, n01, n10, n11 in product((1, -1), (1, -1), (1, -1), (1, -1)):
            O00 = (I + n00 * Z) / 2
            O01 = (I + n01 * Z) / 2
            O10 = (I + n10 * Z) / 2
            O11 = (I + n11 * Z) / 2
            v0 = env_ctm.measure_2x2(O00, O01, O10, O11, sites=((0, 0), (0, 1), (1, 0), (1, 1))).real
            v1 = env_ctm.measure_2x2(O00, O01, O10, O11, sites=((0, 1), (0, 2), (1, 1), (1, 2))).real
            four[((n00, n01), (n10, n11))] = (v0 + v1) / 2
            err = max(err, abs(v0 - v1))

        err = max(err, np.std([four[((-1, 1), (1, 1))], four[((1, -1), (1, 1))], four[((1, 1), (-1, 1))], four[((1, 1), (1, -1))]]))
        err = max(err, np.std([four[((-1, -1), (-1, 1))], four[((-1, -1), (1, -1))], four[((-1, 1), (-1, -1))], four[((1, -1), (-1, -1))]]))
        err = max(err, np.std([four[((-1, -1), (1, 1))], four[((1, 1), (-1, -1))], four[((-1, 1), (-1, 1))], four[((1, -1), (1, -1))]]))
        err = max(err, np.std([four[((1, -1), (-1, 1))], four[((-1, 1), (1, -1))]]))

        four['error'] = err

        Op = (I + Z) / 2
        Om = (I - Z) / 2
        four['single_u'] = env_ctm.measure_nsite(Om, Om, Om, Om, Op, sites=((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))).real
        four['single_d'] = env_ctm.measure_nsite(Op, Op, Op, Op, Om, sites=((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))).real
        d['four'] = four
        np.save(fname, d, allow_pickle=True)




@ray.remote(num_cpus=1)
def run_4point(fname):
    #
    print(fname)
    if 'ZZ' in fname:
        sym = 'dense'
    else:
        sym = 'Z2'
    ops = yastn.operators.Spin12(sym=sym)
    Z, X, vec_x1, base = ZX(ops)
    I = ops.I()
    #
    try:
        d = np.load(fname, allow_pickle=True).item()
    except:
        return
    if 'four' not in d:
        env = peps.load_from_dict(ops.config, d)
        four = {}
        err = 0
        for n00, n01, n10, n11 in product((1, -1), (1, -1), (1, -1), (1, -1)):
            O00 = (I + n00 * Z) / 2
            O01 = (I + n01 * Z) / 2
            O10 = (I + n10 * Z) / 2
            O11 = (I + n11 * Z) / 2
            v0 = env_ctm.measure_2x2(O00, O01, O10, O11, sites=((0, 0), (0, 1), (1, 0), (1, 1))).real
            v1 = env_ctm.measure_2x2(O00, O01, O10, O11, sites=((0, 1), (0, 2), (1, 1), (1, 2))).real
            four[((n00, n01), (n10, n11))] = (v0 + v1) / 2
            err = max(err, abs(v0 - v1))

        err = max(err, np.std([four[((-1, 1), (1, 1))], four[((1, -1), (1, 1))], four[((1, 1), (-1, 1))], four[((1, 1), (1, -1))]]))
        err = max(err, np.std([four[((-1, -1), (-1, 1))], four[((-1, -1), (1, -1))], four[((-1, 1), (-1, -1))], four[((1, -1), (-1, -1))]]))
        err = max(err, np.std([four[((-1, -1), (1, 1))], four[((1, 1), (-1, -1))], four[((-1, 1), (-1, 1))], four[((1, -1), (1, -1))]]))
        err = max(err, np.std([four[((1, -1), (-1, 1))], four[((-1, 1), (1, -1))]]))

        four['error'] = err
        d['four'] = four
        np.save(fname, d, allow_pickle=True)



@ray.remote(num_cpus=1)
def run_5point(fname):
    #
    print(fname)
    if 'ZZ' in fname:
        sym = 'dense'
    else:
        sym = 'Z2'
    ops = yastn.operators.Spin12(sym=sym)
    Z, X, vec_x1, base = ZX(ops)
    I = ops.I()
    #
    try:
        d = np.load(fname, allow_pickle=True).item()
    except:
        return
    if 'four' in d and 'single_u' not in d:
        env = peps.load_from_dict(ops.config, d)
        Op = (I + Z) / 2
        Om = (I - Z) / 2
        d['four']['single_u'] = env.measure_nsite(Om, Om, Om, Om, Op, sites=((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))).real
        d['four']['single_d'] = env.measure_nsite(Op, Op, Op, Op, Om, sites=((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))).real
        np.save(fname, d, allow_pickle=True)



@ray.remote(num_cpus=4)
def run_sample(hz, ta, E, D, chi, which='NN+BP', sf=0.01):
    #
    # Load operators.
    #
    ops = yastn.operators.Spin12(sym='dense')
    Z, X, vec_x1, base = ZX(ops)
    #
    # Initialize system in the product ground state at s=0.
    #
    path = Path(f"./bias_dwave/{hz=:0.4f}/{ta=:0.4f}/{E=:0.4f}/{base}/")
    fname = path / f"env_{D=}_{chi=}_{which=}_{sf=:0.4f}.npy"
    try:
        d = np.load(fname, allow_pickle=True).item()
    except:
        return
    print(d.keys())
    env_ctm = peps.load_from_dict(Z.config, d)
    print(env_ctm)
    #
    number = 1
    xmax, ymax = 128, 128
    opss = {1: ops.vec_z(val=+1), -1: ops.vec_z(val=-1)}

    samples = np.zeros((number, xmax, ymax), dtype=np.int16)
    fname = path / f"sample_{D=}_{chi=}_{which=}_{sf=:0.4f}_{xmax}x{ymax}.npy"

    t1 = time.time()
    for iii in range(number):
        out = env_ctm.sample(projectors=opss, xrange=(0, xmax), yrange=(0, ymax), return_probabilities=False, number=1)
        print(f"{iii} samples in time = {time.time() - t1}")
        for ix in range(xmax):
            for iy in range(ymax):
                samples[iii, ix, iy] = np.array(out[ix, iy], dtype=np.int16)
        np.save(fname, samples)


# if __name__ == '__main__':
#     refs = []
#     for fname in glob.glob("./bias_dwave/**/env_*.npy", recursive=True):
#         #if 'E=0.5000/ZZ/env_D=16_' in fname:
#         refs.append(run_5point.remote(fname))
#     ray.get(refs)


# if __name__ == '__main__':
#     #
#     # run_sample(0.0, 1.0, 1.0, 12, 16, which='NN+BP', sf=0.3)
#     #
#     scs = {0.5: 0.3445, 1.0: 0.3112}
#     refs = []
#     dt = 0.005
#     tas = [2.0, 4.0, 5.4] #[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 5.4, 6.3, 7.2, 8.3, 9.6]
#     hzs = [0, 0.125, 0.25, 0.5, 1.0]  #[0.01] #, 0.125]  # [0] + [2.0 ** (-x) for x in [8, 6, 4, 2, 0, -2]]
#     for EE in [0.5]:
#         for ta in tas:
#             for hz in hzs:
#                 for D in [16]:  # 4, 8, 16, 32
#                     for chi in [16,]:
#                         # for sf in [0.3112, 0.32,  0.33,  0.34,  0.35,  0.36,  0.37,  0.38,  0.39,  0.40,  0.41,  0.42,  0.43,  0.44,  0.45]:
#                         for sf in [0.3445, 0.35,  0.36,  0.37,  0.38,  0.39,  0.40, 0.41,  0.42,  0.43,  0.44,  0.45]:
#                             for which in ["NN+BP"]:
#                                 job = run_sample.remote(hz, ta, EE, D, chi, which, sf)
#                                 refs.append(job)
#     ray.get(refs)


# if __name__ == '__main__':
#     #
#     # run_sample(0.0, 1.0, 1.0, 12, 16, which='NN+BP', sf=0.3)
#     #
#     scs = {0.5: 0.3445, 1.0: 0.3112}
#     refs = []
#     dt = 0.005
#     tas = [2.0, 4.0, 5.4] #[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 5.4, 6.3, 7.2, 8.3, 9.6]
#     hzs = [0, 0.125, 0.25, 0.5, 1.0]  #[0.01] #, 0.125]  # [0] + [2.0 ** (-x) for x in [8, 6, 4, 2, 0, -2]]
#     for EE in [0.5]:
#         for ta in tas:
#             for hz in hzs:
#                 for D in [12]:  # 4, 8, 16, 32
#                     for chi in [16,]:
#                         # for sf in [0.3112, 0.32,  0.33,  0.34,  0.35,  0.36,  0.37,  0.38,  0.39,  0.40]:
#                         for sf in [0.3445, 0.35,  0.36,  0.37,  0.38,  0.39,  0.40, 0.41, 0.42, 0.43, 0.44, 0.45]:
#                             for which in ["NN+BP"]:
#                                 job = run_sample.remote(hz, ta, EE, D, chi, which, sf)
#                                 refs.append(job)
#     ray.get(refs)


if __name__ == '__main__':
    ray.init()
    refs = []
    dt = 0.005
    tas = 2 ** np.linspace(0, 4, 33)
    # tas = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16,] #  [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16, 32] # [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 5.4, 6.3, 7.2, 8.3, 9.6] #
    hzs = [0, 0.125, 0.25, 0.5, 1.0]   # [0.01] #, 0.125]
    for EE in [-2]:   #2.0]:
        for ta in tas:
            for hz in hzs:
                for D in [12, ]:  # 12, 16 # 4, 8, 16, 32
                    for chi in [16,]:
                        for which in ["NN+BP"]:
                            job = run_quench.remote(hz, ta, EE, D, chi, which, dt, 101)
                            refs.append(job)

    ray.get(refs)


# if __name__ == '__main__':
#     ray.init()
#     refs = []
#     dt = 0.01
#     tas = [4.0, 5.4] #  0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0 [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16, 32] # [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 5.4, 6.3, 7.2, 8.3, 9.6] #
#     hzs = [0.25, 0.5, 1.0]   # [0.01] #, 0.125]
#     for EE in [0.5, 1.0]:   #2.0]:op
#         for ta in tas:
#             for hz in hzs:
#                 for D in [16]:  # 4, 8, 16, 32
#                     for chi in [16,]:
#                         for which in ["NN+BP"]:
#                             job = run_quench.remote(hz, ta, EE, D, chi, which, dt, 61)  # 'Z2'
#                             refs.append(job)
#     ray.get(refs)
