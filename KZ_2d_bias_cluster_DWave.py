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



def mean(x):
    return sum(x) / len(x)


def load_drive():
    schedule = np.loadtxt('09-1265A-E_Advantage_system5_4_annealing_schedule.csv', delimiter=",")
    fg = scipy.interpolate.UnivariateSpline(schedule[:, 0], schedule[:, 1] * np.pi, s=0, k=4)
    fJ = scipy.interpolate.UnivariateSpline(schedule[:, 0], schedule[:, 2] * np.pi * (1 - np.exp(-40 * schedule[:, 0])), s=0, k=4)
    return fg, fJ


def get_sc(E):
    gc = 3.04438
    fg, fJ = load_drive()
    sc = scipy.optimize.fsolve(lambda s: fg(s) - gc * E * fJ(s), 0.5)
    return sc[0]


def exp_hamiltonian_nn(H, step):
    H = H.fuse_legs(axes = ((0, 1), (2, 3)))
    D, S = yastn.eigh(H, axes = (0, 1))
    D = yastn.exp(D, step=-step)
    G = yastn.ncon((S, D, S.conj()), ([-1, 1], [1, 2], [-3, 2]))
    G = G.unfuse_legs(axes=(0, 1))
    return peps.gates.decompose_nn_gate(G)


def exp_hamiltonian_local(Hl, step):
    """ exp(-step * Hl). """
    D, S = yastn.eigh(Hl, axes = (0, 1))
    D = yastn.exp(D, step=-step)
    Gl = yastn.ncon((S, D, S.conj()), ([-1, 1], [1, 2], [-3, 2]))
    return Gl


def gate_Ising_cluster(Jzz, hz, hx, step, ops, net):
    Z0, Z1, Z2, Z3 = ops.z(0, 4), ops.z(1, 4), ops.z(2, 4), ops.z(3, 4)
    X0, X1, X2, X3 = ops.x(0, 4), ops.x(1, 4), ops.x(2, 4), ops.x(3, 4)

    Hl = -hx * (X0 + X1 + X2 + X3) - hz * (Z0 + Z1 + Z2 + Z3)
    Hl = Hl - Jzz * (Z0 @ Z1 + Z2 @ Z3 + Z0 @ Z2 + Z1 @ Z3)

    Hh10 = -Jzz * peps.gates.fkron(Z1, Z0) 
    Hh32 = -Jzz * peps.gates.fkron(Z3, Z2)
    Hv20 = -Jzz * peps.gates.fkron(Z2, Z0) 
    Hv31 = -Jzz * peps.gates.fkron(Z3, Z1)

    Gh10 = exp_hamiltonian_nn(Hh10, step)
    Gh32 = exp_hamiltonian_nn(Hh32, step)
    Gv20 = exp_hamiltonian_nn(Hv20, step)    
    Gv31 = exp_hamiltonian_nn(Hv31, step)
    Gl = exp_hamiltonian_local(Hl, step)

    nn = []
    for bond in net.bonds(dirn='v'):
        nn.append(Gv20._replace(bond=bond))
        nn.append(Gv31._replace(bond=bond))
    for bond in net.bonds(dirn='h'):
        nn.append(Gh10._replace(bond=bond))
        nn.append(Gh32._replace(bond=bond))

    local = [peps.gates.Gate_local(Gl, site) for site in net.sites()]

    return peps.gates.Gates(nn=nn, local=local)


@ray.remote(num_cpus=4)
def run_quench(hz, ta, E, D, chi, which='NN+', dt=0.01):
    #
    geometry = peps.CheckerboardLattice()
    #
    # Define quench protocol
    #
    fg, fJ = load_drive()
    #
    # Load operators.
    ops = NSpin12(sym='dense')
    oZ = [ops.z(0, 4), ops.z(1, 4), ops.z(2, 4), ops.z(3, 4)]
    oX = [ops.x(0, 4), ops.x(1, 4), ops.x(2, 4), ops.x(3, 4)]
    #
    # Initialize system in the product ground state at s=0.
    psi = peps.product_peps(geometry=geometry, vectors=ops.vec_x(val=(1, 1, 1, 1)))
    #
    # simulation parameters
    opts_svd_ntu = {"D_total": D}
    env = peps.EnvBP(psi, which=which)
    env.iterate_(max_sweeps=50, diff_tol=1e-8)
    
    si, t = 0, 0
    ss = [get_sc(E), 0.4, 0.5, 0.6]
    infoss = []
    dtold = dt
    for sf in ss:
        ds = sf - si
        si = sf
        steps = round(ta * ds / dtold)
        dt = ta * ds / steps
        dt2 = 1j * dt / 2

        keep_time = time.time()
        for _ in tqdm(range(steps), disable=True):
            t += dt / 2
            gates = gate_Ising_cluster(fJ(t / ta), hz * fJ(t / ta), fg(t / ta), dt2, ops, geometry)
            infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_ntu)
            env.iterate_(max_sweeps=50, diff_tol=1e-8)
            infoss.append(infos)
            t += dt / 2

        Delta = peps.accumulated_truncation_error(infoss, statistics='mean')
        print(f"Accumulated mean truncation error {Delta:0.5f}")
        #
        opts_svd_env = {'D_total': chi}
        env_ctm = peps.EnvCTM(psi, init='eye')
        info = env_ctm.ctmrg_(opts_svd=opts_svd_env, max_sweeps=100, corner_tol=1e-5)
        print(info)
        #
        # Calculating 1-site <X_i> for all sites
        Ex = [mean(env_ctm.measure_1site(oX[n]).values()).real for n in range(4)]
        print("Ex", Ex)
        Ex = mean(Ex)
        #
        Ez = [mean(env_ctm.measure_1site(oZ[n]).values()).real for n in range(4)]
        print("Ez", Ez)
        Ez = mean(Ez)
        #
        Ezz = []
        Ezz.append(mean(env_ctm.measure_1site(oZ[0] @ oZ[1]).values()).real)
        Ezz.append(mean(env_ctm.measure_1site(oZ[0] @ oZ[2]).values()).real)
        Ezz.append(mean(env_ctm.measure_1site(oZ[1] @ oZ[3]).values()).real)
        Ezz.append(mean(env_ctm.measure_1site(oZ[2] @ oZ[3]).values()).real)
        print("Ezz", Ezz)
        Ezz = mean(Ezz)
        #
        path = Path(f"./bias_cluster_dwave/{hz=:0.4f}/{ta=:0.4f}/{E=:0.4f}/")
        path.mkdir(parents=True, exist_ok=True)
        #
        fieldnames = ["D", "which", "chi", "dt", "Delta", "time", "Ezz", "Ez", "Ex"]
        out = {"D" : D, "which" : which, "chi": chi, "dt": dt, "Delta": Delta,
                "time": time.time() - keep_time, "Ezz": Ezz, "Ez": Ez, "Ex": Ex}
        #
        fname = path / f"data_new_{sf=:0.4f}.csv"
        file_exists = os.path.isfile(fname)
        with open(fname, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
            if not file_exists:
                writer.writeheader()
            writer.writerow(out)
        
        # fname_ctm = path / f"ctm_{sf=:0.4f}_{D=}_{which=}_dt={dtold:0.2f}.npy"
        # data = {}
        # data["env"] = env_ctm.save_to_dict()
        # data["info"] = infoss
        # np.save(fname_ctm, data, allow_pickle=True)




@ray.remote(num_cpus=4)
def run_quench_old(hz, ta, E, D, which='NN+', dt=0.01):
    #
    geometry = peps.CheckerboardLattice()
    #
    # Define quench protocol
    #
    fg, fJ = load_drive()
    #
    # Load operators.
    ops = NSpin12(sym='dense')
    oZ = [ops.z(0, 4), ops.z(1, 4), ops.z(2, 4), ops.z(3, 4)]
    oX = [ops.x(0, 4), ops.x(1, 4), ops.x(2, 4), ops.x(3, 4)]
    #
    # Initialize system in the product ground state at s=0.
    psi = peps.product_peps(geometry=geometry, vectors=ops.vec_x(val=(1, 1, 1, 1)))
    #
    # simulation parameters
    opts_svd_ntu = {"D_total": D}
    env = peps.EnvBP(psi, which=which)
    env.iterate_(max_sweeps=50, diff_tol=1e-8)
    
    si, t = 0, 0
    ss = [get_sc(E), 0.6]
    infoss = []
    dtold = dt
    for sf in ss:
        ds = sf - si
        si = sf
        steps = round(ta * ds / dtold)
        dt = ta * ds / steps
        dt2 = 1j * dt / 2

        keep_time = time.time()
        for _ in tqdm(range(steps), disable=True):
            t += dt / 2
            gates = gate_Ising_cluster(fJ(t / ta), hz * fJ(t / ta), fg(t / ta), dt2, ops, geometry)
            infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_ntu)
            env.iterate_(max_sweeps=50, diff_tol=1e-8)
            infoss.append(infos)
            t += dt / 2

        Delta = peps.accumulated_truncation_error(infoss, statistics='mean')
        print(f"Accumulated mean truncation error {Delta:0.5f}")
        #
        opts_svd_env = {'D_total': 4 * D}
        env_ctm = peps.EnvCTM(psi, init='eye')
        info = env_ctm.ctmrg_(opts_svd=opts_svd_env, max_sweeps=100, corner_tol=1e-5)
        print(info)
        #
        # Calculating 1-site <X_i> for all sites
        Ex = [mean(env_ctm.measure_1site(oX[n]).values()).real for n in range(4)]
        print("Ex", Ex)
        Ex = mean(Ex)
        #
        Ez = [mean(env_ctm.measure_1site(oZ[n]).values()).real for n in range(4)]
        print("Ez", Ez)
        Ez = mean(Ez)
        #
        Ezz = []
        Ezz.append(mean(env_ctm.measure_1site(oZ[0] @ oZ[1]).values()).real)
        Ezz.append(mean(env_ctm.measure_1site(oZ[0] @ oZ[2]).values()).real)
        Ezz.append(mean(env_ctm.measure_1site(oZ[1] @ oZ[3]).values()).real)
        Ezz.append(mean(env_ctm.measure_1site(oZ[2] @ oZ[3]).values()).real)
        print("Ezz", Ezz)
        Ezz = mean(Ezz)
        #
        path = Path(f"./bias_cluster_dwave/{hz=:0.4f}/{ta=:0.4f}/{E=:0.4f}/")
        path.mkdir(parents=True, exist_ok=True)
        #
        fieldnames = ["D", "which", "dt", "Delta", "time", "Ezz", "Ez", "Ex"]
        out = {"D" : D, "which" : which, "dt": dt, "Delta": Delta,
                "time": time.time() - keep_time, "Ezz": Ezz, "Ez": Ez, "Ex": Ex}
        #
        fname = path / f"data_{sf=:0.4f}.csv"
        file_exists = os.path.isfile(fname)
        with open(fname, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
            if not file_exists:
                writer.writeheader()
            writer.writerow(out)
        
        fname_ctm = path / f"ctm_{sf=:0.4f}_{D=}_{which=}_dt={dtold:0.2f}.npy"
        data = {}
        data["env"] = env_ctm.save_to_dict()
        data["info"] = infoss
        np.save(fname_ctm, data, allow_pickle=True)



if __name__ == '__main__':
    ray.init()
    refs = []
    dt = 0.01
    tas = [1, 2, 3, 4, 5.378, 6.215, 7.182, 8.299, 9.59] #* (2.0 ** (ii/2)) for ii in range(13)]
    hzs = [0.01, 0.123285] # [0] + [2.0 ** (-x) for x in [8, 6, 4, 2, 0, -2]]
    for EE in [0.5, 1]:
        for ta in tas:
            for hz in hzs:
                for D in [20, 24]:
                    for chi in [1, 2, 4, 8]:
                        for which in ["BP"]:  # ,
                            # print(hz, ta, D, which, sym, dt)
                            # run_quench(hz, ta, D, which, dt)
                            # run_quench(hz, ta, E, D, which, dt)
                            job = run_quench.remote(hz, ta, EE, D, chi, which, dt)
                            refs.append(job)
    ray.get(refs)