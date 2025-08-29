import csv
import os
from pathlib import Path
import ray
import time
from tqdm import tqdm  # progressbar
import yastn
import yastn.tn.fpeps as peps
from yastn.tn.fpeps.gates import gate_nn_Ising, gate_local_field


def mean(x):
    return sum(x) / len(x)


@ray.remote(num_cpus=2)
def run_quench(hz, tq, D, which='NN+', sym='dense', dt=0.01):
    #
    geometry = peps.CheckerboardLattice()
    #
    # Define quench protocol
    fXX = lambda s : (1 + s - (4 / 27) * (s ** 3))
    fZ = lambda s : 3.04438 * (1 - s + (4 / 27) * (s ** 3))
    #
    steps = round((3 * tq / 2) / dt)
    dt = (3 * tq / 2) / steps
    path = Path(f"./bias/{hz=:0.4f}/{tq=:0.4f}/")
    path.mkdir(parents=True, exist_ok=True)
    #
    # Load operators. Problem has Z2 symmetry, which we impose.
    ops = yastn.operators.Spin12(sym=sym)
    #
    # Initialize system in the product ground state at s=0.
    psi = peps.product_peps(geometry=geometry, vectors=ops.vec_z(val=1))
    #
    # simulation parameters
    opts_svd_ntu = {"D_total": D}
    env = peps.EnvNTU(psi, which=which)
    #
    # execute time evolution
    infoss = []
    t = -3 * tq / 2
    dt2 = 1j * dt / 2
    keep_time = time.time()
    for sss in [0., 1.]:
        for _ in tqdm(range(steps), disable=True):
            t += dt / 2
            gnn = gate_nn_Ising(-1 * fXX(t / tq), dt2, ops.I(), ops.x())
            glz = gate_local_field(fZ(t / tq), dt2, ops.I(), ops.z())
            if hz != 0:
                glx = gate_local_field(hz * fXX(t / tq), dt2, ops.I(), ops.x())
                gates = peps.gates.distribute(geometry, gates_nn=gnn, gates_local=[glz, glx])
            else:
                gates = peps.gates.distribute(geometry, gates_nn=gnn, gates_local=[glz])
            infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_ntu)
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
        Ex = mean(env_ctm.measure_1site(ops.x()).values()).real
        Ez = mean(env_ctm.measure_1site(ops.z()).values()).real
        Exx = mean(env_ctm.measure_nn(ops.x(), ops.x()).values()).real

        fieldnames = ["D", "which", "sym", "dt", "Delta", "time", "Exx", "Ex", "Ez"]
        out = {"D" : D, "which" : which, "sym": sym, "dt": dt, "Delta": Delta,
               "time": time.time() - keep_time, "Exx": Exx, "Ex": Ex, "Ez": Ez}

        fname = path / f"data_{sss=:0.2f}.csv"
        file_exists = os.path.isfile(fname)
        with open(fname, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
            if not file_exists:
                writer.writeheader()
            writer.writerow(out)

    # if sym == 'dense':
    #     pp = (ops.I() + ops.x()) / 2
    #     pm = (ops.I() - ops.x()) / 2
    #     smp = env_ctm.sample(xrange=(0, 10), yrange=(0, 10), projectors=[pp, pm], number=1000)



if __name__ == '__main__':
    ray.init()
    refs = []
    dt = 0.01
    tqs = [0.3, 0.6, 1.2, 2.4]
    hzs = [0, 0.125, 0.5]
    sym = 'dense'
    for tq in tqs:
        for hz in hzs:  # [(0, "Z2"), (0, "dense")]
            for D in [4, 6, 8, 10]:
                for which in ["NN+"]:  # ,
                    # print(hz, tq, D, which, sym, dt)
                    # run_quench(hz, tq, D, which, sym, dt)
                    job = run_quench.remote(hz, tq, D, which, sym, dt)
                    refs.append(job)
    ray.get(refs)