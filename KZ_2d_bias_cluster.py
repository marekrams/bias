import csv
import os
from pathlib import Path
import ray
import time
from tqdm import tqdm  # progressbar
import yastn
import yastn.tn.fpeps as peps
from routines import NSpin12


def mean(x):
    return sum(x) / len(x)


def exp_hamiltonian_nn(H, step):
    H = H.fuse_legs(axes = ((0, 1), (2, 3)))
    D, S = yastn.eigh(H, axes = (0, 1))
    D = yastn.exp(D, step=-step)
    G = yastn.ncon((S, D, S.conj()), ([-1, 1], [1, 2], [-3, 2]))
    G = G.unfuse_legs(axes=(0, 1))
    return peps.gates.decompose_nn_gate(G)


def gate_Ising_cluster(Jzz, hz, hx, step, ops, net):
    Z0, Z1, Z2, Z3 = ops.z(0, 4), ops.z(1, 4), ops.z(2, 4), ops.z(3, 4)
    X0, X1, X2, X3 = ops.x(0, 4), ops.x(1, 4), ops.x(2, 4), ops.x(3, 4)

    Hl = -hx * (X0 + X1 + X2 + X3) -hz * (Z0 + Z1 + Z2 + Z3)
    Hl = Hl - Jzz * (Z0 @ Z1 + Z2 @ Z3 + Z0 @ Z2 + Z1 @ Z3)

    Hh = -Jzz * (peps.gates.fkron(Z1, Z0) + peps.gates.fkron(Z3, Z2))
    Hv = -Jzz * (peps.gates.fkron(Z2, Z0) + peps.gates.fkron(Z3, Z1))

    Gh = exp_hamiltonian_nn(Hh, step)
    Gv = exp_hamiltonian_nn(Hv, step)

    D, S = yastn.eigh(Hl, axes = (0, 1))
    D = yastn.exp(D, step=-step)
    Gl = yastn.ncon((S, D, S.conj()), ([-1, 1], [1, 2], [-3, 2]))

    nn = []
    for bond in net.bonds(dirn='v'):
        nn.append(Gv._replace(bond=bond))
    for bond in net.bonds(dirn='h'):
        nn.append(Gh._replace(bond=bond))

    local = [peps.gates.Gate_local(Gl, site) for site in net.sites()]

    return peps.gates.Gates(nn=nn, local=local)


# @ray.remote(num_cpus=18)
def run_quench(hz, tq, D, which='NN+', dt=0.01):
    #
    print(f"Starting {hz=} {tq=} {D=} {which=} {dt=}")
    #
    geometry = peps.CheckerboardLattice()
    #
    # Define quench protocol
    fZZ = lambda s : (1 + s - (4 / 27) * (s ** 3))
    fX = lambda s : 3.04438 * (1 - s + (4 / 27) * (s ** 3))
    #
    steps = round((3 * tq / 2) / dt)
    dt = (3 * tq / 2) / steps
    path = Path(f"./bias_cluster/{hz=:0.4f}/{tq=:0.4f}/")
    path.mkdir(parents=True, exist_ok=True)
    #
    # Load operators. Problem has Z2 symmetry, which we impose.
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
    #
    # execute time evolution
    infoss = []
    t = -3 * tq / 2
    dt2 = 1j * dt / 2
    keep_time = time.time()
    for sss in [0., 1.]:
        for step in tqdm(range(steps), disable=True):
            t += dt / 2
            gates = gate_Ising_cluster(fZZ(t / tq), hz * fZZ(t / tq), fX(t / tq), dt2, ops, geometry)
            infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_ntu)
            infoss.append(infos)
            t += dt / 2
            iii = env.iterate_(max_sweeps=20, diff_tol=1e-8)
            Delta = peps.accumulated_truncation_error(infoss, statistics='mean')
            print(f"Step {step}/{steps} Delta={Delta:0.5f}", iii)

        print("-----------------------")
        print(" Expectation values s=", sss)
        #
        phi = psi.copy()
        if D > 14:
            env_BP = peps.EnvBP(phi)
            env_BP.BP_(max_sweeps=50, diff_tol=1e-8)
            for DD in range(D-2, 10, -2):
                info = peps.truncate_(env_BP, opts_svd={"D_total": 12})
                env_BP.BP_(max_sweeps=50, diff_tol=1e-8)
            opts_svd_env = {'D_total': 48}
            print(info)
        else:
            opts_svd_env = {'D_total': 4 * D}

        env_ctm = peps.EnvCTM(phi, init='eye')
        info = env_ctm.ctmrg_(opts_svd=opts_svd_env, max_sweeps=100, corner_tol=1e-5)
        print(info)
        #
        # Calculating 1-site <X_i> for all sites
        Ex = [mean(env_ctm.measure_1site(oX[n]).values()).real for n in range(4)]
        print("Ex", Ex)
        Ex = mean(Ex)

        Ez = [mean(env_ctm.measure_1site(oZ[n]).values()).real for n in range(4)]
        print("Ez", Ez)
        Ez = mean(Ez)

        Ezz = []
        Ezz.append(mean(env_ctm.measure_1site(oZ[0] @ oZ[1]).values()).real)
        Ezz.append(mean(env_ctm.measure_1site(oZ[0] @ oZ[2]).values()).real)
        Ezz.append(mean(env_ctm.measure_1site(oZ[1] @ oZ[3]).values()).real)
        Ezz.append(mean(env_ctm.measure_1site(oZ[2] @ oZ[3]).values()).real)
        print("Ezz", Ezz)
        Ezz = mean(Ezz)

        fieldnames = ["D", "which", "sym", "dt", "Delta", "time", "Ezz", "Ez", "Ex"]
        out = {"D" : D, "which" : which, "sym": 'dense', "dt": dt, "Delta": Delta,
               "time": time.time() - keep_time, "Ezz": Ezz, "Ez": Ez, "Ex": Ex}

        fname = path / f"data_{sss=:0.2f}.csv"
        file_exists = os.path.isfile(fname)
        with open(fname, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
            if not file_exists:
                writer.writeheader()
            writer.writerow(out)


if __name__ == '__main__':
    # ray.init()
    refs = []
    dt = 0.01
    tqs = [0.1, ]  # [0.1 * (2.0 ** (ii/2)) for ii in range(11)]  # 1.6, 3.2
    hzs = [0.5, ]  # [0] + [2.0 ** (-x) for x in [8, 6, 4, 2, 0, -2]]
    for tq in tqs:
        for hz in hzs:
            for D in [12, ]:
                for which in ["NN+BP", "NN+"]:  # ,
                    # print(hz, tq, D, which, sym, dt)
                    # run_quench(hz, tq, D, which, dt)
                    run_quench(hz, tq, D, which, dt)
    #                 job = run_quench.remote(hz, tq, D, which, dt)
    #                 refs.append(job)
    # ray.get(refs)
