#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Surface scattering models (BSDFs) for Monte Carlo ray tracing.

The models are written in direction cosine space: a direction leaving a
surface is described by beta, the projection of its unit vector onto the
tangent plane of the surface (abs(beta) = sin of the angle to the normal), and
the specular direction by beta_0. Scattering then shifts beta away from
beta_0, and the projected solid angle is the area element in that plane,
cos(theta_s) d(omega) = d^2 beta. The total integrated scatter (TIS), the
scattered fraction of the incident power, is the integral of the BSDF over
the unit disk abs(beta) <= 1 ::

    TIS(beta_0) = integral over |beta| <= 1 of BSDF(beta - beta_0) d^2 beta

All models here are shift invariant and isotropic (the BSDF depends only on
abs(beta - beta_0)), which is the form of Harvey's linear systems formulation
of surface scatter and of the ABg model used by commercial stray light
software:

- `Lambertian`: BSDF = rho/pi, a perfectly diffuse surface of reflectance
  rho (an exception: centred on the normal rather than on beta_0),
- `Gaussian`: BSDF = A exp(-abs(beta - beta_0)**2/(2 sigma**2)), a lobe
  around the specular direction, given by its TIS at normal incidence,
- `ABg`: BSDF = A/(B + abs(beta - beta_0)**g), the fit of measured scatter of
  polished optics (g between 1 and 3) and of black coatings,
- `Mixture`: a sum of models, e.g. a diffuse floor plus a specular lobe.

Each model gives its value (`bsdf()`), its TIS as a function of abs(beta_0)
(`tis()`), samples of beta distributed in proportion to the BSDF over the
unit disk (`sample()`), and the probability density of those samples
(`pdf()`, needed for importance sampling in `pyoptics.nsrt`). Near grazing
incidence part of a lobe around beta_0 falls outside the unit disk (into
evanescent directions); it is not scattered, so the TIS of a lobe drops
towards grazing incidence.

The values of the presets in `pyoptics.nsrt` (black anodized aluminium,
black paint, machined metal) are invented but of plausible size.

References
----------
J. E. Harvey, "Light-scattering characteristics of optical surfaces",
PhD thesis, University of Arizona (1976).

E. R. Freniere, G. G. Gregory and R. A. Hassler, "Edge diffraction in Monte
Carlo ray tracing", Proc. SPIE 3780, 151-157 (1999) (the ABg model as used in
stray light codes).

J. C. Stover, "Optical Scattering: Measurement and Analysis", 3rd ed., SPIE
Press (2012).

E. Veach and L. J. Guibas, "Optimally combining sampling techniques for Monte
Carlo rendering", SIGGRAPH 95, 419-428 (1995) (multiple importance sampling).
"""

import numpy as np

__all__ = ["BSDF", "Lambertian", "Gaussian", "ABg", "Mixture", "beta_to_direction",
           "direction_to_beta"]


def _rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _arc_inside(rho, b):
    """Angle (radians) of the circle of radius rho about a point at distance
    b from the origin that lies inside the unit disk."""

    rho, b = np.broadcast_arrays(np.asarray(rho, float), np.asarray(b, float))
    out = np.zeros(rho.shape)
    full = rho <= 1 - b
    out[full] = 2*np.pi
    part = ~full & (rho < 1 + b) & (rho > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (rho[part]**2 + b[part]**2 - 1)/(2*rho[part]*b[part])
    out[part] = 2*np.arccos(np.clip(c, -1.0, 1.0))
    return out


def direction_to_beta(d, normal, e1=None):
    """Direction cosines beta (projection onto the tangent plane) of unit
    directions d against the unit normals, in the tangent basis (e1, e2).

    Parameters
    ----------
    d, normal : (N, 3) arrays
    e1 : (N, 3) array, optional
        First tangent vector; default: from `tangent_basis()`.

    Returns
    -------
    beta : (N, 2) array
    e1, e2 : (N, 3) arrays
    """

    if e1 is None:
        e1, e2 = tangent_basis(normal)
    else:
        e2 = np.cross(normal, e1)
    return np.stack([np.sum(d*e1, -1), np.sum(d*e2, -1)], -1), e1, e2


def tangent_basis(normal):
    """Orthonormal tangent vectors (e1, e2) for unit normals (N, 3)."""

    a = np.where(np.abs(normal[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    e1 = np.cross(normal, a)
    e1 /= np.linalg.norm(e1, axis=-1)[:, None]
    e2 = np.cross(normal, e1)
    return e1, e2


def beta_to_direction(beta, normal, e1, e2):
    """Unit directions from direction cosines beta (N, 2) on the side of
    the unit normals (N, 3)."""

    bz = np.sqrt(np.clip(1 - np.sum(beta**2, -1), 0.0, None))
    return beta[:, :1]*e1 + beta[:, 1:]*e2 + bz[:, None]*normal


class BSDF(object):
    """Base class of the scattering models (see the module docstring)."""

    def bsdf(self, beta, beta0):
        """BSDF (1/sr) at the direction cosines beta (N, 2) for the specular
        direction cosines beta0 (N, 2)."""

        raise NotImplementedError

    def tis(self, b0):
        """Total integrated scatter for abs(beta_0) = b0 (array)."""

        raise NotImplementedError

    def sample(self, beta0, rng=None):
        """Direction cosines (N, 2) inside the unit disk, distributed in
        proportion to the BSDF, for the specular direction cosines beta0."""

        raise NotImplementedError

    def pdf(self, beta, beta0):
        """Probability density per unit area of beta (per projected solid
        angle) of `sample()`: bsdf/tis."""

        b0 = np.linalg.norm(beta0, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = self.bsdf(beta, beta0)/self.tis(b0)
        return np.where(np.isfinite(p), p, 0.0)

    def __add__(self, other):
        return Mixture([self, other])


class Lambertian(BSDF):
    """Perfectly diffuse scatter, BSDF = rho/pi.

    Parameters
    ----------
    rho : number
        Diffuse reflectance (or transmittance), the TIS at every angle.
    """

    def __init__(self, rho):
        self.rho = float(rho)

    def __repr__(self):
        return "Lambertian({})".format(self.rho)

    def bsdf(self, beta, beta0):
        inside = np.sum(beta**2, -1) <= 1.0
        return np.where(inside, self.rho/np.pi, 0.0)

    def tis(self, b0):
        return np.full(np.shape(b0), self.rho)

    def sample(self, beta0, rng=None):
        rng = _rng(rng)
        n = len(beta0)
        r = np.sqrt(rng.random(n))
        phi = 2*np.pi*rng.random(n)
        return np.stack([r*np.cos(phi), r*np.sin(phi)], -1)

    def pdf(self, beta, beta0):
        return np.where(np.sum(beta**2, -1) <= 1.0, 1/np.pi, 0.0)


class _ShiftInvariant(BSDF):
    """BSDF f(abs(beta - beta_0)) with tabulated radial sampling and TIS."""

    n_rho = 2048
    n_b0 = 129

    def profile(self, rho):
        raise NotImplementedError

    def _tables(self):
        if getattr(self, "_tab", None) is not None:
            return self._tab
        rho_min = self._rho_scale()*1e-4
        rho = np.concatenate([[0.0], np.geomspace(rho_min, 2.0, self.n_rho - 1)])
        f = self.profile(rho)
        # cumulative of f(rho) 2 pi rho drho (trapezoids):
        g = f*2*np.pi*rho
        cdf = np.concatenate([[0.0], np.cumsum(0.5*(g[1:] + g[:-1])*np.diff(rho))])
        # TIS(b0) = integral f(rho) arc(rho, b0) rho drho
        b0 = np.linspace(0.0, 1.0, self.n_b0)
        h = f[None, :]*_arc_inside(rho[None, :], b0[:, None])*rho[None, :]
        tis = np.sum(0.5*(h[:, 1:] + h[:, :-1])*np.diff(rho)[None, :], axis=-1)
        self._tab = rho, cdf, b0, tis
        return self._tab

    def _rho_scale(self):
        return 1.0

    def bsdf(self, beta, beta0):
        rho = np.linalg.norm(beta - beta0, axis=-1)
        return np.where(np.sum(beta**2, -1) <= 1.0, self.profile(rho), 0.0)

    def tis(self, b0):
        _, _, bt, tis = self._tables()
        return np.interp(np.clip(b0, 0.0, 1.0), bt, tis)

    def sample(self, beta0, rng=None, max_rounds=200):
        rng = _rng(rng)
        rho_t, cdf, _, _ = self._tables()
        beta0 = np.asarray(beta0, float)
        out = np.full(beta0.shape, np.nan)
        todo = np.arange(len(beta0))
        for _ in range(max_rounds):
            if not len(todo):
                break
            u = rng.random(len(todo))*cdf[-1]
            rho = np.interp(u, cdf, rho_t)
            phi = 2*np.pi*rng.random(len(todo))
            b = beta0[todo] + np.stack([rho*np.cos(phi), rho*np.sin(phi)], -1)
            ok = np.sum(b**2, -1) <= 1.0
            out[todo[ok]] = b[ok]
            todo = todo[~ok]
        if len(todo):  # (almost) no part of the lobe inside the disk
            out[todo] = beta0[todo]/np.maximum(1.0, np.linalg.norm(beta0[todo], axis=-1))[:, None]
        return out


class Gaussian(_ShiftInvariant):
    """Gaussian lobe around the specular direction,
    BSDF = A exp(-abs(beta - beta_0)**2/(2 sigma**2)).

    Parameters
    ----------
    tis : number
        Total integrated scatter at normal incidence (sets A).
    sigma : number
        Width of the lobe in direction cosines (about the angular width in
        radians near normal incidence).
    """

    def __init__(self, tis, sigma):
        self.tis0, self.sigma = float(tis), float(sigma)
        # TIS at b0 = 0: A 2 pi sigma**2 (1 - exp(-1/(2 sigma**2)))
        self.A = self.tis0/(2*np.pi*self.sigma**2*(1 - np.exp(-0.5/self.sigma**2)))
        self._tab = None

    def __repr__(self):
        return "Gaussian(tis={}, sigma={})".format(self.tis0, self.sigma)

    def _rho_scale(self):
        return self.sigma

    def profile(self, rho):
        return self.A*np.exp(-0.5*(np.asarray(rho)/self.sigma)**2)


class ABg(_ShiftInvariant):
    """Harvey's ABg model, BSDF = A/(B + abs(beta - beta_0)**g).

    Parameters
    ----------
    A, B, g : numbers
        A sets the level, B**(1/g) the width of the flat top of the lobe
        (in direction cosines), g the slope of its fall-off on a log-log plot.
    """

    def __init__(self, A, B, g):
        self.A, self.B, self.g = float(A), float(B), float(g)
        self._tab = None

    def __repr__(self):
        return "ABg(A={}, B={}, g={})".format(self.A, self.B, self.g)

    def _rho_scale(self):
        return min(1.0, self.B**(1/self.g)) if self.g > 0 else 1.0

    def profile(self, rho):
        return self.A/(self.B + np.asarray(rho, float)**self.g)


class Mixture(BSDF):
    """Sum of scattering models (e.g. a diffuse floor plus a lobe).

    Parameters
    ----------
    components : sequence of BSDF
    """

    def __init__(self, components):
        flat = []
        for c in components:
            flat.extend(c.components if isinstance(c, Mixture) else [c])
        self.components = flat

    def __repr__(self):
        return "Mixture({!r})".format(self.components)

    def bsdf(self, beta, beta0):
        return sum(c.bsdf(beta, beta0) for c in self.components)

    def tis(self, b0):
        return sum(c.tis(b0) for c in self.components)

    def sample(self, beta0, rng=None):
        rng = _rng(rng)
        b0 = np.linalg.norm(beta0, axis=-1)
        w = np.array([c.tis(b0) for c in self.components])  # (K, N)
        cum = np.cumsum(w, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = rng.random(len(beta0))*cum[-1]
        pick = np.sum(u[None, :] >= cum, axis=0).clip(0, len(self.components) - 1)
        out = np.empty_like(np.asarray(beta0, float))
        for k, c in enumerate(self.components):
            m = pick == k
            if np.any(m):
                out[m] = c.sample(beta0[m], rng)
        return out
