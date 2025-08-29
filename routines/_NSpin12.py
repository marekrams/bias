""" Generator of basic local spin-1/2 operators. """
from __future__ import annotations
import numpy as np
from yastn import YastnError, Tensor, Leg, ncon, eye
from yastn.operators import meta_operators, Spin12
from itertools import product

class NSpin12(meta_operators):
    # Predefined set of Pauli operators and spin-1/2 operators.
    def __init__(self, sym='dense', **kwargs):
        r"""
        A set of standard operators for 2^N-dimensional Hilbert space.
        Defines identity, Pauli operators and their eigenvectors.

        Parameters
        ----------
        sym : str
            Explicit symmetry to used. Allowed options are :code:`'dense'` or ``'Z2'``.

        kwargs
            Other YASTN configuration parameters can be provided, see :meth:`yastn.make_config`.

        Notes
        -----
        Basis order is fixed such that applying to_dense()
        for 'Z2' operators reduce them to the corresponding 'dense' operators.
        For 'Z2', the order is fixed following the Kronecker product of 1-site operators.

        Default configuration sets :code:`fermionic` to :code:`False`.
        """
        if sym not in ('dense', 'Z2'):
            raise YastnError("For NSpin12 sym should be in 'dense' or 'Z2'.")
        kwargs['fermionic'] = False
        kwargs['sym'] = sym
        super().__init__(**kwargs)
        self._sym = sym
        self.operators = ('I', 'x','z')
        kwargs['sym'] = 'Z2'
        self.ops12z2 = Spin12(**kwargs)


    def space(self, N) -> yastn.Leg:
        r""" :class:`yastn.Leg` describing local Hilbert space. """
        if self._sym == 'dense':
            leg = Leg(self.config, s=1, D=(2 ** N,))
        if self._sym == 'Z2':
            leg = Leg(self.config, s=1, t=(0, 1), D=(2 ** (N-1), 2 ** (N-1)))
        return leg


    def I(self, N) -> yastn.Tensor:
        r""" Identity operator. """
        return eye(self.config, legs=self.space(N), isdiag=False)


    def x(self, n, N) -> yastn.Tensor:
        r""" Pauli :math:`\sigma_n^x` operator for n in range(N). """
        X1, I1 = self.ops12z2.x(), self.ops12z2.I()
        x = X1 if n == 0 else I1
        for i in range(1, N):
            tmp = X1 if n == i else I1
            x = ncon([x, tmp], [(-0, -2), (-1, -3)]).fuse_legs(axes=[(0, 1), (2, 3)], mode='hard').drop_leg_history()
        if self._sym == 'Z2':
            return x
        return x.to_nonsymmetric()  # else: 'dense'

    def y(self, n, N) -> yastn.Tensor:
        r""" Pauli :math:`\sigma_n^x` operator for n in range(N). """
        Y1, I1 = self.ops12z2.y(), self.ops12z2.I()
        x = Y1 if n == 0 else I1
        for i in range(1, N):
            tmp = Y1 if n == i else I1
            x = ncon([x, tmp], [(-0, -2), (-1, -3)]).fuse_legs(axes=[(0, 1), (2, 3)], mode='hard').drop_leg_history()
        if self._sym == 'Z2':
            return x
        return x.to_nonsymmetric()  # else: 'dense'


    def z(self, n, N) -> yastn.Tensor:
        r""" Pauli :math:`\sigma_n^z` operator for n in range(N). """
        Z1, I1 = self.ops12z2.z(), self.ops12z2.I()
        z = Z1 if n == 0 else I1
        for i in range(1, N):
            tmp = Z1 if n == i else I1
            z = ncon([z, tmp], [(-0, -2), (-1, -3)]).fuse_legs(axes=[(0, 1), (2, 3)], mode='hard').drop_leg_history()
        if self._sym == 'Z2':
            return z
        return z.to_nonsymmetric()  # else: 'dense'


    def vec_z(self, val=(1,)) -> yastn.Tensor:
        r"""
        Normalized eigenvectors of :math:`\sigma_n^z`.
        len(val) == N. Translates 0 to -1.
        """
        val = [v if v != 0 else -1 for v in val]
        N = len(val)

        vs = {1: np.array([1, 0], dtype=int),
             -1: np.array([0, 1], dtype=int)}

        tmp = vs[val[0]]
        for i in range(1, N):
            l = 2 ** (i - 1)
            v = vs[val[i]]
            tmp = np.concatenate([tmp[:l] * v[0], tmp[l:] * v[1], tmp[:l] * v[1], tmp[l:] * v[0]], axis=0)

        if self._sym == 'dense':
            vec = Tensor(config=self.config, s=(1,))
            vec.set_block(Ds=(2**N,), val=tmp)
            return vec

        # if self._sym == 'Z2':
        l = 2 ** (N - 1)
        nn = sum(tmp[l:])
        vec = Tensor(config=self.config, s=(1,), n=(nn,))
        vec.set_block(ts=(nn,), Ds=(2 ** (N-1),), val=tmp[l * nn : l * (nn + 1)])
        return vec


    def vec_x(self, val=1) -> yastn.Tensor:
        r""" Normalized eigenvectors of :math:`\sigma^x`. """

        val = [v if v != 0 else -1 for v in val]
        N = len(val)

        vs = {1: np.array([1, 1], dtype=int),
             -1: np.array([1, -1], dtype=int)}

        tmp = vs[val[0]]
        for i in range(1, N):
            l = 2 ** (i - 1)
            v = vs[val[i]]
            tmp = np.concatenate([tmp[:l] * v[0], tmp[l:] * v[1], tmp[:l] * v[1], tmp[l:] * v[0]], axis=0)

        tmp = tmp * 2 ** (-N / 2)

        if self._sym == 'dense':
            vec = Tensor(config=self.config, s=(1,))
            vec.set_block(Ds=(2**N,), val=tmp)
            return vec

        raise ValueError("vec_x cannot be defined for Z2 symmetry")


    def all_vecs(self, N, which):
        vals = [*product(*[(1, 0)] *  N)]
        if which == 'X':
            vecs = [self.vec_x(val).conj() for val in vals]
        elif which == 'Z':
            vecs = [self.vec_z(val).conj() for val in vals]
        vals = {i: v for i, v in enumerate(vals)}
        return vals, vecs
