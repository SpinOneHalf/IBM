import numpy as np
from numba import njit, prange, jit
from scipy import interpolate

# Parameters
gamma = 1.4
CFL = 0.5


@njit
def update(Qnew, dt, dx, dy, nx, ny, Fh, Gh, fb):
    for j in prange(1, ny - 1):
        for i in prange(1, nx - 1):
            Qnew[i, j, :] = Qnew[i, j, :] - (dt / dx) * (Fh[i, j, :] - Fh[i - 1, j, :]) - (dt / dy) * (
                    Gh[i, j, :] - Gh[i, j - 1, :]) + fb[i, j, :] * dt
    return Qnew


def simulation(r, u, v, E, p, c,
               dx, dy, ny, nx, X0, tMax, dtheta,k=.001):
    # TODO: Redesign around generic force on fluid
    t = 0
    while t < tMax:
        print(t)
        dt = CFL * min(dx, dy) / np.max(c + np.sqrt(u * u + v * v))
        Qnew, F, G = generateQFG(r, u, v, E, p)
        Fh, Gh = generateFhGh(Qnew, F, G, c, u, v)

        X0 = X0 + dt / 2 * forcesBody(Qnew, X0, dx, dy, ny, nx)
        FB = forceFluid(dx, dy, nx, ny, X0, dtheta, t,k,u,v)
        Qnew = update(Qnew, dt, dx, dy, nx, ny, Fh, Gh, FB)
        X0 = X0 + dt / 2 * forcesBody(Qnew, X0, dx, dy, ny, nx)
        Qnew[:, 0, :] = Qnew[:, 1, :]
        Qnew[0, :, :] = Qnew[2, :, :]
        Qnew[:, ny - 1, :] = Qnew[:, ny - 2, :]
        Qnew[nx - 1, :, :] = Qnew[nx - 2, :, :]
        # Primitive variables
        r = Qnew[:, :, 0]
        u = Qnew[:, :, 1] / r
        v = Qnew[:, :, 2] / r
        E = Qnew[:, :, 3]
        p = (gamma - 1) * (E - 0.5 * r * (u * u + v * v))
        c = np.sqrt(gamma * p / r)
        t = t + dt  # Advance  time

    return r, u, v, p, E, c, t, X0


@njit(parallel=True)
def generateQFG(r, u, v, E, p):
    nx, ny = r.shape
    Q = np.zeros((nx, ny, 4))
    F = np.zeros((nx, ny, 4))
    G = np.zeros((nx, ny, 4))
    for j in prange(0, ny):
        for i in prange(0, nx):
            # Flux calculation
            Q[i, j, 0] = r[i, j]
            Q[i, j, 1] = r[i, j] * u[i, j]
            Q[i, j, 2] = r[i, j] * v[i, j]
            Q[i, j, 3] = E[i, j]

            F[i, j, 0] = r[i, j] * u[i, j]
            F[i, j, 1] = r[i, j] * u[i, j] ** 2 + p[i, j]
            F[i, j, 2] = r[i, j] * u[i, j] * v[i, j]
            F[i, j, 3] = u[i, j] * (E[i, j] + p[i, j])

            G[i, j, 0] = r[i, j] * v[i, j]
            G[i, j, 1] = r[i, j] * v[i, j] * u[i, j]
            G[i, j, 2] = r[i, j] * v[i, j] ** 2 + p[i, j]
            G[i, j, 3] = v[i, j] * (E[i, j] + p[i, j])
    return Q, F, G


@njit(parallel=True)
def generateFhGh(Q, F, G, c, u, v):
    nx, ny = c.shape

    ah1 = np.zeros((nx, ny))
    ah2 = np.zeros((nx, ny))

    Fh = np.zeros((nx - 1, ny, 4))
    Gh = np.zeros((nx, ny - 1, 4))

    for j in prange(0, ny - 1):
        for i in prange(0, nx - 1):
            ah1[i, j] = max(abs(u[i, j]) + c[i, j], abs(u[i + 1, j]) + c[i + 1, j])
            ah2[i, j] = max(abs(v[i, j]) + c[i, j], abs(v[i, j + 1]) + c[i, j + 1])

            Fh[i, j, :] = 0.5 * (F[i + 1, j, :] + F[i, j, :] - ah1[i, j] * (Q[i + 1, j, :] - Q[i, j, :]))
            Gh[i, j, :] = 0.5 * (G[i, j, :] + G[i, j + 1, :] - ah2[i, j] * (Q[i, j + 1, :] - Q[i, j, :]))
    return Fh, Gh

@njit
def forceFluid(dx, dy, nx, ny, X0, dtheta, t,k,u,v):
    Fk = _curve_force(X0, dtheta, t,k)
    p = X0.shape[0]
    Fb = np.zeros((nx, ny, 4))
    for i in range(nx):
        for j in range(ny):
            xp = i * dx
            yp = j * dy
            ftemp = np.zeros((3))
            # TODO: Final all local points, THEN COMPUTE FORCE
            for k in range(p):
                xtemp = X0[k, :]
                xpt = np.remainder(np.floor(xtemp[0] / dx), nx) * dx
                ypt = np.remainder(np.floor(xtemp[1] / dy), ny) * dy
                w = SixPointDelta((xpt - xp)/(dx))*SixPointDelta((ypt - yp)/dy) / (dx * dy)
                Fl = w * Fk[k, :] * dtheta
                ftemp[0:2] += Fl
                ftemp[2]+=Fl[0]*u[i,j]+Fl[1]*v[i,j]
            Fb[i, j, 1] = ftemp[0]
            Fb[i, j, 2] = ftemp[1]
            Fb[i,j,3]=ftemp[2]

    return Fb
@njit
def _curve_force(xs, dthe, t, K=.1):
    k = xs.shape[0]
    tempf = np.zeros((k, 2))
    for i in range(1, k - 1):
        tempf[i, :] = ((xs[i + 1] - 2 * xs[i, :] + xs[i - 1]) / (dthe ** 2)) *K
    tempf[0, :] = ((xs[-1, :] - 2 * xs[0, :] + xs[1, :]) / (dthe ** 2)) * K
    tempf[-1, :] = ((xs[-2, :] - 2 * xs[-1, :] + xs[0, :]) / (dthe ** 2)) * K
    return tempf


def forcesBody(Qnew, X0,
               dx, dy,
               nx, ny):
    x = np.arange(0, nx) * dx
    y = np.arange(0, ny) * dy
    r = Qnew[:, :, 0]
    u = Qnew[:, :, 1] / r
    v = Qnew[:, :, 2] / r
    points_n, _ = X0.shape

    vs = np.zeros((points_n, 2))
    for i in range(points_n):
        temp_x = X0[i, :]
        xp = np.remainder(np.floor(temp_x[0] / dx), nx) * dx
        yp = np.remainder(np.floor(temp_x[0] / dy), ny) * dy
        up = interpolate.interpn((x, y), u, (xp, yp))
        vp = interpolate.interpn((x, y), v, (xp, yp))
        vs[i, 1] = up
        vs[i, 0] = vp
    return vs


def RiamanProblemInit(nx=200,
                      ny=100):
    # Initialization
    c = np.zeros((nx, ny))
    r = np.zeros((nx, ny))
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    E = np.zeros((nx, ny))
    p = np.zeros((nx, ny))
    # Creating mesh

    dx = 1 / (nx - 1)
    dy = 1 / (ny - 1)
    x = np.zeros((nx, ny))
    y = np.zeros((nx, ny))
    for j in np.arange(0, ny):
        for i in np.arange(0, nx):
            x[i, j] = (i) * dx
            y[i, j] = (j) * dy
    # Initial condition
    for j in np.arange(0, ny):
        for i in np.arange(0, nx):
            # left up
            if x[i, j] < 0.5 and .5 < y[i, j]:
                p[i, j] = .3
                r[i, j] = .5323
                u[i, j] = 1.206
                v[i, j] = 0
            # left lower
            elif x[i, j] <= 0.5 and 0.5 >= y[i, j]:
                p[i, j] = .029
                r[i, j] = .138
                u[i, j] = 1.206
                v[i, j] = 1.206
            # right upper
            elif x[i, j] >= 0.5 and 0.5 <= y[i, j]:
                p[i, j] = 1.5
                r[i, j] = 1.5
                u[i, j] = 0
                v[i, j] = 0
            # right lower
            elif x[i, j] >= 0.5 and .5 >= y[i, j]:
                p[i, j] = .3
                r[i, j] = .5323
                u[i, j] = 0
                v[i, j] = 1.206
            E[i, j] = p[i, j] / (gamma - 1) + 0.5 * r[i, j] * (u[i, j] ** 2 + v[i, j] ** 2)
            c[i, j] = np.sqrt(gamma * p[i, j] / r[i, j])
    return r, u, v, p, E, c, dx, dy


# TODO: move all inits into seperate file... this looks bad
def static_case(nx=100,
                ny=100):
    # Initialization
    c = np.zeros((nx, ny))
    r = np.zeros((nx, ny))
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    E = np.zeros((nx, ny))
    p = np.zeros((nx, ny))
    dx = 1 / (nx - 1)
    dy = 1 / (ny - 1)
    # Creating mesh
    for i in range(nx):
        for j in range(ny):
            p[i, j] = .3
            r[i, j] = .5323
            u[i, j] = 0
            v[i, j] = 0
            E[i, j] = p[i, j] / (gamma - 1) + 0.5 * r[i, j] * (u[i, j] ** 2 + v[i, j] ** 2)
            c[i, j] = np.sqrt(gamma * p[i, j] / r[i, j])
    return r, u, v, p, E, c, dx, dy


def right_case(nx=100,
               ny=100):
    # Initialization
    c = np.zeros((nx, ny))
    r = np.zeros((nx, ny))
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    E = np.zeros((nx, ny))
    p = np.zeros((nx, ny))
    dx = 1 / (nx - 1)
    dy = 1 / (ny - 1)
    # Creating mesh
    for i in range(nx):
        for j in range(ny):
            p[i, j] = 3
            r[i, j] = .5323
            u[i, j] = 1
            v[i, j] = 0
            E[i, j] = p[i, j] / (gamma - 1) + 0.5 * r[i, j] * (u[i, j] ** 2 + v[i, j] ** 2)
            c[i, j] = np.sqrt(gamma * p[i, j] / r[i, j])
    return r, u, v, p, E, c, dx, dy


def generate_circle(r, n,d=.5):
    xs = np.zeros((n, 2))
    dtheta = (2 * np.pi) / n
    for i in range(n):
        xs[i, :] = np.array([np.cos(dtheta * i) * r, np.sin(dtheta * i) * r])+d
    return xs, dtheta


# TODO Move delta functions to utils
@njit
def beta(x):
    _beta = (9 / 4) - (3 / 2) * (np.power(x, 2)) + ((22 / 3)) * x - (7 / 3) * np.power(x, 3)
    return _beta


@njit
def _gamma(x):
    chunk = ((161 / 36)) * (1 / 2) * np.power(x, 2)
    chunk += (-(109 / 24)) * (1 / 3) * np.power(x, 4)
    chunk += (5 / 18) * np.power(x, 6)
    return chunk / 4


@njit
def pthree(b, d, a):
    return (-b + np.sqrt(d)) / (2 * a)


@njit
def SixPointDelta(x):
    alpha = 28

    if x > -3 and x <= -2:
        x = x + 3
        tempb = beta(x)
        tempgamma = _gamma(x)
        d = np.power(tempb, 2) - 4 * alpha * tempgamma
        pm3 = pthree(tempb, d, alpha)
        return pm3
    elif x > -2 and x <= -1:
        x = x + 2
        tempb = beta(x)
        tempgamma = _gamma(x)
        d = np.power(tempb, 2) - 4 * alpha * tempgamma
        pm3 = pthree(tempb, d, alpha)
        pm2 = -3 * pm3 - (1 / 16) + (1 / 8) * (np.power(x, 2)) + (1 / 12) * (- 1) * x + (1 / 12) * np.power(x, 3)
        return pm2
    elif x > -1 and x <= 0:
        x = x + 1
        tempb = beta(x)
        tempgamma = _gamma(x)
        d = np.power(tempb, 2) - 4 * alpha * tempgamma
        pm3 = pthree(tempb, d, alpha)
        pm1 = 2 * pm3 + (1 / 4) + (1 / 6) * (4) * x - (1 / 6) * np.power(x, 3)
        return pm1

    elif x > 0 and x <= 1:
        tempb = beta(x)
        tempgamma = _gamma(x)
        d = np.power(tempb, 2) - 4 * alpha * tempgamma
        pm3 = pthree(tempb, d, alpha)
        p = 2 * pm3 + (5 / 8) - (1 / 4) * (np.power(x, 2))
        return p
    elif x > 1 and x <= 2:
        x = x - 1
        tempb = beta(x)
        tempgamma = _gamma(x)
        d = np.power(tempb, 2) - 4 * alpha * tempgamma
        pm3 = pthree(tempb, d, alpha)
        pp1 = -3 * pm3 + (1 / 4) - (1 / 6) * (4) * x + (1 / 6) * np.power(x, 3)
        return pp1
    elif x > 2 and x <= 3:
        x = x - 2
        tempb = beta(x)
        tempgamma = _gamma(x)
        d = np.power(tempb, 2) - 4 * alpha * tempgamma
        pm3 = pthree(tempb, d, alpha)
        pp2 = pm3 - (1 / 16) + (1 / 8) * (np.power(x, 2)) - (1 / 12) * (- 1) * x - (1 / 12) * np.power(x, 3)
        return pp2
    return 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nx = 150
    ny = 150
    r0, u0, v0, p0, E0, c0, dx, dy = static_case(nx, ny)
    xs, dtheta = generate_circle(.1, 500)
    results = simulation(r0, u0, v0, E0, p0, c0,
                         dx, dy, ny, nx, xs, .2, dtheta)
    xf = results[-1]
    plt.scatter(xs[:, 0], xs[:, 1])
    plt.show()
    print("DONE")
