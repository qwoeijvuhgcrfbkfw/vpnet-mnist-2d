import math
import time
import datetime

import torch
import scipy
from typing import Any, Callable, List, Optional
import matplotlib.pyplot as plt

"""Function systems and Variable Projection operators."""


class FunSystem:
    """Abstract function system base class. Callable."""

    def __init__(self, num_samples: int, num_coeffs: int, num_params: int) -> None:
        """
        Initializes properties. Subclasses should call this in their constructor as of
            super().__init__(num_samples, num_coeffs, num_params)

        Input:
            num_samples: int    Number of time sampling points of the input
            num_coeffs: int     Number of output coefficients (number of functions)
            num_params: int     Number of nonlinear system parameters
        """
        self.num_samples = num_samples
        self.num_coeffs = num_coeffs
        self.num_params = num_params

    def __call__(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the sampled function system and its derivatives with respect to
        the nonlinear system parameters. Subclasses should implement this method.

        Input:
            params: torch.Tensor    Tensor of nonlinear system parameters.
                                    Expected size: (num_params)
        Output:
            Phi: torch.Tensor       Tensor of sampled function system.
                                    Size: (num_coeffs, num_samples)
                                    Phi[i,j] represents the ith basic function
                                    sampled at the jth time instance.
                                    Can be thought of as a transposed Phi matrix
                                    from the papers
            dPhi: torch.Tensor      Tensor of the function system derivatives.
                                    Size: (num_params, num_coeffs, num_samples)
                                    dPhi[p,i,j] represents the derivative of
                                    the ith basic function with respect to the
                                    pth system parameter param[p] sampled at
                                    the jth time instance.
        """
        raise NotImplementedError()


class HermiteSystem(FunSystem):
    """
    Adaptive Hermite functions: classical Hermite functions parametrized
    by dilation and translation.
    """

    def __init__(self, num_samples: int, num_coeffs: int) -> None:
        """
        Initializes properties. System parameters consist of dilation and
        translation, i.e. property num_params = 2 fixed.

        See also FunSystem.__init__. Property"""
        assert num_samples > 0
        assert num_coeffs > 1

        super().__init__(num_samples, num_coeffs, 2)

    def __call__(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes adaptive Hermite functions and derivatives. System parameters
        params is expected to contain dilation and translation in this order.
        The functions are uniformly sampled.

        See also FunSystem.__call__
        """
        dilation, translation = params[:2]

        m2 = self.num_samples // 2

        t = (
            torch.arange(-m2, m2 + 1, dtype=params.dtype, device=params.device)
            if self.num_samples % 2
            else torch.arange(-m2, m2, dtype=params.dtype, device=params.device)
        )

        x = dilation * (t - translation * m2)

        w = torch.exp(-0.5 * x**2)

        pi_sqrt = 1 / torch.sqrt(torch.sqrt(torch.tensor(math.pi, dtype=params.dtype, device=params.device)))
        n_sqrt = torch.sqrt(2 * torch.arange(0, self.num_coeffs, dtype=params.dtype, device=params.device))

        # stack to avoid gradient computation error on inplace modification
        Phi: List[Optional[torch.Tensor]] = self.num_coeffs * [None]  # type: ignore
        dPhi: List[Optional[torch.Tensor]] = self.num_coeffs * [None]  # type: ignore

        Phi[0] = torch.sqrt(dilation) * pi_sqrt * w
        dPhi[0] = -x * Phi[0]

        Phi[1] = n_sqrt[1] * x * Phi[0]
        dPhi[1] = -x * Phi[1] + n_sqrt[1] * Phi[0]

        for j in range(2, self.num_coeffs):
            Phi[j] = (2 * x * Phi[j - 1] - n_sqrt[j - 1] * Phi[j - 2]) / n_sqrt[j]  # type: ignore
            dPhi[j] = -x * Phi[j] + n_sqrt[j] * Phi[j - 1]  # type: ignore

        Phi = torch.stack(Phi)  # type: ignore
        dPhi = torch.stack(dPhi)  # type: ignore
        dPhi = torch.stack((dPhi * (t - translation * m2) + 0.5 * Phi / dilation, -dPhi * dilation * m2))  # type: ignore

        return Phi, dPhi  # type: ignore

    def __repr__(self):
        return f"HermiteSystem(num_samples={self.num_samples}, num_coeffs={self.num_coeffs})"


class RealMTSystem(FunSystem):
    """
    Real Malmquist--Takenaka system.

    The system is parametrized by a sequence of inverse poles with given
    multiplicities (mults). Every inverse pole is represented with its
    complex magnitude and argument.
    """

    def __init__(self, num_samples: int, mults: list[int]) -> None:
        """
        Initializes properties. Complex system parameters (inverse poles)
        are represented with their complex magnitude and argument. Number
        of coefficients and parameters depend on the multiplicities:
            num_params = 2 * len(mults)
            num_coeffs = 1 + 2 * sum(mults)

        Input:
            num_samples: int    Number of time sampling points of the input
            mults: list[int]    List of multiplicities of inverse poles.
        See also FunSystem.__init__
        """
        assert num_samples > 0
        assert len(mults) > 0
        assert all([m > 0 for m in mults])

        super().__init__(num_samples, 1 + 2 * sum(mults), 2 * len(mults))

        self.mults = mults

        # index precomputation
        num_coeffs_c = self.num_coeffs // 2  # sum(mults)
        i = 0

        k0: List[Optional[torch.Tensor]] = self.num_params * [None]  # type: ignore
        k1: List[Optional[torch.Tensor]] = self.num_params * [None]  # type: ignore
        k2: List[Optional[torch.Tensor]] = self.num_params * [None]  # type: ignore

        for j, m in enumerate(self.mults):
            k0_j = num_coeffs_c * [0]
            k1_j = num_coeffs_c * [0]
            k2_j = num_coeffs_c * [0]

            for k in range(m):
                k0_j[i] = 1
                k1_j[i] = k
                k2_j[i] = k + 1

                i += 1
            for ii in range(i, num_coeffs_c):
                k1_j[ii] = m
                k2_j[ii] = m

            k0_j = torch.tensor(k0_j)
            k1_j = torch.tensor(k1_j)
            k2_j = torch.tensor(k2_j)

            k0[2 * j] = k0_j
            k0[2 * j + 1] = torch.zeros_like(k0_j)

            k1[2 * j] = -k1_j
            k1[2 * j + 1] = -k1_j

            k2[2 * j] = k2_j
            k2[2 * j + 1] = -k2_j

        self._k0 = torch.stack(k0).unsqueeze(-1)  # (num_params, num_coeffs, 1) # type: ignore
        self._k1 = torch.stack(k1).unsqueeze(-1)  # (num_params, num_coeffs, 1) # type: ignore
        self._k2 = torch.stack(k2).unsqueeze(-1)  # (num_params, num_coeffs, 1) # type: ignore

    def __call__(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the real MT functions and derivatives. System parameters
        (inverse poles) are represented with their complex magnitude and
        argument, in this order, i.e. params is expected to have the
        structure
            [magnitude0, argument0, magnitude1, argument1, ...]
        The functions are uniformly sampled between -pi and pi.

        See also FunSystem.__call__
        """
        t = (
            2 * torch.arange(self.num_samples, dtype=params.dtype, device=params.device) / self.num_samples - 1
        ) * math.pi

        z = torch.exp(1j * t)

        sqrt2 = torch.sqrt(torch.tensor(2, dtype=params.dtype, device=params.device))

        # complex intermediate computation
        mults = torch.tensor(self.mults, device=params.device)

        # Phi
        r = torch.repeat_interleave(params[0::2], mults).unsqueeze(-1)
        a = torch.repeat_interleave(params[1::2], mults).unsqueeze(-1)

        eia = torch.exp(1j * a)

        R1 = z - r * eia  # (num_coeffs // 2, num_samples)
        R2 = 1 - r * eia.conj() * z  # (num_coeffs // 2, num_samples)

        B0 = sqrt2 * torch.sqrt(1 - r**2) / R2
        B1 = R1 / R2

        B = torch.cumprod(B1, dim=0)
        B = torch.cat((torch.ones((1, self.num_samples), dtype=params.dtype, device=params.device), B))
        B = B[:-1, :]

        Phi = B0 * B * z  # (num_coeffs // 2, num_samples)

        # dPhi
        r0 = params[0::2]
        r = torch.stack((r0, r0)).T.reshape(self.num_params).unsqueeze(-1).unsqueeze(-1)

        a0 = params[1::2]
        a = torch.stack((a0, a0)).T.reshape(self.num_params).unsqueeze(-1).unsqueeze(-1)

        eia = torch.exp(1j * a)

        R1 = z - r * eia  # (num_params, 1, num_samples)
        R2 = 1 - r * eia.conj() * z  # (num_params, 1, num_samples)

        d0 = -r / (1 - r**2)  # (num_params, 1, 1)

        d1 = eia / R1  # (num_params, 1, num_samples)
        d2 = eia.conj() * z / R2  # (num_params, 1, num_samples)

        dm = torch.stack((torch.ones_like(r0), 1j * r0)).T.reshape(self.num_params).unsqueeze(-1).unsqueeze(-1)

        k0 = self._k0.to(dtype=params.dtype, device=params.device)
        k1 = self._k1.to(dtype=params.dtype, device=params.device)
        k2 = self._k2.to(dtype=params.dtype, device=params.device)

        dPhi = (k0 * d0 + k1 * d1 + k2 * d2) * dm * Phi  # (num_params, num_coeffs // 2, num_samples)

        # Phi_0, dPhi_0
        Phi = torch.cat(
            (torch.ones((1, self.num_samples), dtype=params.dtype, device=params.device), Phi.real, Phi.imag)
        )  # (num_coeffs, num_samples)

        dPhi = torch.cat(
            (
                torch.zeros((self.num_params, 1, self.num_samples), dtype=params.dtype, device=params.device),
                dPhi.real,
                dPhi.imag,
            ),
            dim=1,
        )  # (num_params, num_coeffs, num_samples)

        return Phi, dPhi

    def __repr__(self):
        return f"RealMTSystem(num_samples={self.num_samples}, mults={self.mults})"


def bernstein_mtx_transposed(curve_degree: int, nodes: torch.Tensor) -> torch.Tensor:
    """
    This function computes the transposed Bernstein matrix, for the given
    curve degree and nodes.

    Parameters:
        curve_degree: int   The curve degree. Curve degree n implies n + 1 control points.
        nodes: torch.Tensor The nodes of the curve. Should be a tensor of size (nodes_count)

    Suppose the resulting matrix is B, then B is of size (curve_degree + 1, nodes_count).
    Thus, B[i, j] represents the i-th Bernstein polynomial evaluated at node j.
    """
    out = torch.Tensor()

    # Loop over the rows of the matrix, same polynomials evaluated at different nodes
    for j in range(0, curve_degree + 1):
        # Compute the leading Bernstein polynomial coefficient for the current row
        n_choose_j = scipy.special.comb(curve_degree, j, exact=True)

        # Computing (n choose j) * node^j * (1 - node)^(curve_degree - j) for each node
        col = n_choose_j * torch.pow(nodes, j) * torch.pow(torch.ones(nodes.shape) - nodes, curve_degree - j)

        # Transforming a column into a row and adding it to the output matrix
        col = torch.unsqueeze(col, dim=0)
        out = torch.cat((out, col), dim=0)

    return out

def bernstein_derivative(curve_degree: int, nodes: torch.Tensor) -> torch.Tensor:
    """
    This function computes the derivative of the Bernstein polynomials, at a given curve
    degree and nodes, with respect to the nodes.

    Parameters:
        curve_degree: int   The curve degree. Curve degree n implies n + 1 control points.
        nodes: torch.Tensor The nodes of the curve. Should be a tensor of size (nodes_count)

    The output should be a tensor of size (nodes_count, curve_degree + 1, nodes_count).
    In other words, for each node parameter a matrix output represents the partial derivative of
    the matrix-valued function.

    An observation is used from the paper, which allows us to compute all the nonzero pieces
    of the output tensor using a bernstein matrix of a lower curve degree.

    We only need to distribute the columns into their own otherwise zero matrices in their
    positions and return the result. Idk how the code below works
    """
    bpmat_t = bernstein_mtx_transposed(curve_degree - 1, nodes)

    zero_row = torch.zeros(1, bpmat_t.size(1))

    der_cols = torch.cat((zero_row, bpmat_t), dim=0) - torch.cat((bpmat_t, zero_row), dim=0)

    return torch.eye(der_cols.size(1)).unsqueeze(1) * der_cols.T.unsqueeze(-1)


class BezierCurveSystem(FunSystem):
    """
    This class represents a Bezier curve system, which fits a Bezier curve to a set of points.

    The set of points is represented by a tensor of size (num_fitpoints * 2), all the x coordinates
    should be first, and be followed by the y coordinates.

    Since the number of fitpoints matches with the number of nodes (nonlinear parameters as per
    the paper), the number of samples is initialized as num_fitpoints * 2 (for x and y cords),
    and the number of nonlinear parameters as num_fitpoints.

    The number of control points is represented by the curve_degree parameter. The curve degree n
    implies n + 1 control points, indexed from 0 to n. The Bezier curve is computed by a sum of
    products of Bernstein polynomials evaluated at the nodes with the control points.
    """

    def __init__(self, num_fitpoints: int, curve_degree: int, fix_first_last: bool = False) -> None:
        super().__init__(num_fitpoints * 2, (curve_degree + 1) * 2, num_fitpoints - 2 if fix_first_last else num_fitpoints)

        self.curve_degree = curve_degree
        self.fix_first_last = fix_first_last

    def __call__(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the transposed Bernstein matrix, and its derivative.

        The Phi matrix is of size (num_coeffs = (curve_degree + 1) * 2, num_samples = num_fitpoints * 2)
        and is just the Bernstein matrix transposed and put into a block diag. with itself.

        Its derivative is of size (num_params = num_fitpoints, num_coeffs = (curve_degree + 1) * 2,
        num_samples = num_fitpoints * 2), and is computed using a technique described in the above,
        and again put into a block diag. (for each matrix slab) with itself.
        """

        if self.fix_first_last:
            params = torch.cat((torch.Tensor([0]), params, torch.Tensor([1])))

        bmat_t = bernstein_mtx_transposed(self.curve_degree, params)

        phi_mat = torch.block_diag(bmat_t, bmat_t)
        dphi_mat_once = bernstein_derivative(self.curve_degree, params)

        mtxs, m, n = dphi_mat_once.shape
        dphi_mat = torch.zeros((mtxs, m * 2, n * 2))

        dphi_mat[:, :m, :n] = dphi_mat_once
        dphi_mat[:, m:, n:] = dphi_mat_once

        if self.fix_first_last:
            dphi_mat = dphi_mat[1:-1]

        return phi_mat, dphi_mat


class Hermite2DXMSystem(FunSystem):
    def __init__(self, num_samples_x: int, num_samples_y: int, num_directional_coeff: int):
        assert num_samples_x > 0
        assert num_samples_y > 0
        assert num_directional_coeff > 1

        super().__init__(num_samples_x * num_samples_y, num_directional_coeff ** 2, 4)

        self.system_x = HermiteSystem(num_samples_x, num_directional_coeff)
        self.system_y = HermiteSystem(num_samples_y, num_directional_coeff)

        self.num_samples_x = num_samples_x
        self.num_samples_y = num_samples_y

    def __call__(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dilation_x, translation_x, dilation_y, translation_y = params[:4]

        Phi_x, dPhi_x = self.system_x(torch.Tensor([dilation_x, translation_x]))
        Phi_y, dPhi_y = self.system_y(torch.Tensor([dilation_y, translation_y]))

        Phi_out = (Phi_x[:   , None, None, :   ] *
                   Phi_y[None, :   , :   , None]).reshape(self.num_coeffs, self.num_samples)

        dXPhi_out = (dPhi_x[:   , :   , None, None, :   ] *
                     Phi_y [None, None, :   , :   , None]).reshape(self.system_x.num_params, self.num_coeffs, self.num_samples)

        dYPhi_out = (Phi_x [None, :   , None, None, :   ] *
                     dPhi_y[:   , None, :   , :   , None]).reshape(self.system_y.num_params, self.num_coeffs, self.num_samples)

        return Phi_out, torch.cat([dXPhi_out, dYPhi_out], dim=0)



def bbmm(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    Batch-batch matrix multiplication.

    Input:
        t1: torch.Tensor    Input tensor of size (batch1,*,num_samples).
        t2: torch.Tensor    Input tensor of size (batch2,num_samples,**).
    Output:
        out: torch.Tensor   Output tensor of size (batch2,batch1,*,**) computed
                            as the product of t1 and t2 along the last
                            dimension of t1 and the second (first non-batch)
                            dimension of t2.
    """
    t2 = t2.permute(*range(1, t2.ndim), 0)  # (num_samples,**,batch2)

    out = torch.tensordot(t1, t2, dims=1)  # (batch1,*,**,batch2)

    out = out.permute(-1, *range(out.ndim - 1))  # (batch2,batch1,*,**)

    return out


class VPFun(torch.autograd.Function):
    """Variable Projection operators with analytic derivatives."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        params: torch.Tensor,
        fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the orthogonal projection of the input.

        Input:
            x: torch.Tensor         Input tensor of size (batch,*,num_samples).
            params: torch.Tensor    Tensor of nonlinear system parameters.
                                    Size: (num_params)
            fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
                                    Function system and derivative builder.
                                    Expected to return Phi and dPhi as of
                                    FunSystem.__call__
        Output:
            coeffs: torch.Tensor    Coefficients of orthogonal projection.
                                    coeffs = Phi^+ x
                                    Size: (batch,*,num_coeffs)
            x_hat: torch.Tensor     Orthogonal projection, VP approximation.
                                    x_hat = Phi coeffs
                                    Size: (batch,*,num_samples)
            res: torch.Tensor       Residual vector.
                                    res = x - x_hat
                                    Size: (batch,*,num_samples)
            r2: torch.Tensor        L2 error of approximation.
                                    r2 = ||res||^2
                                    Size: (batch,*)
        """
        phi, dphi = fun_system(params)  # phi: (num_coeffs,num_samples)
        phip = phi.T

        coeffs = x @ phip  # (batch,*,num_coeffs), coefficients

        x_hat = coeffs @ phi  # (batch,*,num_samples), approximations

        res = x - x_hat  # (batch,*,num_samples), residual vectors

        r2 = (res**2).sum(dim=-1)  # (batch,*), L2 errors

        ctx.save_for_backward(phi, phip, dphi, coeffs, res)

        return coeffs, x_hat, res, r2

    @staticmethod
    def backward(
        ctx: Any, d_coeff: torch.Tensor, d_x_hat: torch.Tensor, d_res: torch.Tensor, d_r2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Computes the backpropagation gradients.

        Input:
            d_coeffs: torch.Tensor  Backpropagated gradient of coeffs.
                                    Size: (batch,*,num_coeffs)
            d_x_hat: torch.Tensor   Backpropagated gradient of x_hat.
                                    Size: (batch,*,num_samples)
            d_res: torch.Tensor     Backpropagated gradient of res.
                                    Size: (batch,*,num_samples)
            d_r2: torch.Tensor      Backpropagated gradient of r2.
                                    Size: (batch,*)
        Output:
            dx: torch.Tensor        Gradient of input x.
                                    Size: (batch,*,num_samples)
            d_params: torch.Tensor  Gradient of params.
                                    Size: (num_params)
            None                    [Argument if not differentiable.]
        """
        phi, phip, dphi, coeffs, res = ctx.saved_tensors

        # Intermediate Jacobians:
        #   Jac1 = dPhi coeff
        #   Jac2 = Phi^+^T dPhi^T res
        #   Jac3 = dPhi^T Phi^+^T c
        jac1 = bbmm(coeffs, dphi)  # (num_params,batch,*,num_samples)
        jac2 = bbmm(res, dphi.mT) @ phip.T  # (num_params,batch,*,num_samples)
        jac3 = bbmm(coeffs @ phip.T, dphi.mT)  # (num_params,batch,*,num_coeffs)

        # Jacobians
        jac_coeff = jac3 + (-jac1 + jac2 - jac3 @ phi) @ phip  # (num_params,batch,*,num_coeffs)
        jac_x_hat = jac1 - jac1 @ phip @ phi + jac2  # (num_params,batch,*,num_samples)
        jac_res = -jac_x_hat  # (num_params,batch,*,num_samples)
        jac_r2 = -2 * (jac1 * res).sum(dim=-1)  # (num_params,batch,*)

        # gradients
        dx = d_coeff @ phip.T + d_x_hat @ phip @ phi + d_res - d_res @ phip @ phi + 2 * d_r2.unsqueeze(-1) * res
        d_params = (
            (jac_coeff * d_coeff).flatten(1).sum(dim=1)
            + (jac_x_hat * d_x_hat).flatten(1).sum(dim=1)
            + (jac_res * d_res).flatten(1).sum(dim=1)
            + (jac_r2 * d_r2).flatten(1).sum(dim=1)
        )

        return dx, d_params, None


class VPIteration:  # use similar to torch.autograd.Function
    """
    Variable Projection iteration step for higher level computation.
    Backpropagation gradients are automatically computed based on VPFun.
    Use similar to torch.autograd.Function.
    """

    @staticmethod
    def apply(
        x: torch.Tensor, params: torch.Tensor, fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Computes one step of the Variable Projection iteration.

        Input:
            x: torch.Tensor         Input tensor of size (batch,*,num_samples).
            params: torch.Tensor    Tensor of nonlinear system parameters.
                                    Size: (num_params)
            fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
                                    Function system and derivative builder.
                                    Expected to return Phi and dPhi as of
                                    FunSystem.__call__
        Output:
            iter: torch.Tensor      Iteration step.
                                    iter = -2 * res^T @ dPhi @ coeffs
                                    Size: (num_params)
        """
        phi, dphi = fun_system(params)

        fun_fixed: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = lambda params: (phi, dphi)

        coeffs, _, res, _ = VPFun.apply(x, params, fun_fixed)

        return -2 * (bbmm(coeffs, dphi) * res).sum()


class VPLayer(torch.nn.Module):
    """Variable Projection layer for neural networks."""

    def __init__(
        self, params_init: torch.Tensor, fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """
        Initializes parameters.

        Input:
            params_init: torch.Tensor   Initial values of nonlinear system parameters.
                                        Size: (num_params)
            fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
                                        Function system and derivative builder.
                                        Expected to return Phi and dPhi as of
                                        FunSystem.__call__
        """
        super().__init__()  # type: ignore

        self.params_init = params_init
        self.params = torch.nn.Parameter(params_init.clone().detach())
        self.fun_system = fun_system

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward operator, see VPFun"""
        return VPFun.apply(x, self.params, self.fun_system)

    def extra_repr(self) -> str:
        return f"params_init={self.params_init}, fun_system={self.fun_system}"
