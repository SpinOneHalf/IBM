import numpy as np
import matplotlib.pyplot as plt

nx = 2000
ny = 2000

# Initialization
r = np.zeros((nx, ny))
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))
E = np.zeros((nx, ny))
p = np.zeros((nx, ny))

x = np.zeros((nx, ny))
y = np.zeros((nx, ny))

Q = np.zeros((nx, ny, 4))
F = np.zeros((nx, ny, 4))
G = np.zeros((nx, ny, 4))

Fh = np.zeros((nx , ny, 4))
Gh = np.zeros((nx, ny , 4))

Qnew = np.zeros((nx, ny, 4))

c = np.zeros((nx, ny))
ah1 = np.zeros((nx, ny))
ah2 = np.zeros((nx, ny))

# Parameters
gamma = 1.4
CFL = 0.5
tMax = 0.3
dx = 1 / (nx - 1)
dy = 1 / (ny - 1)

# Creating mesh
for j in np.arange(0, ny):
    for i in np.arange(0, nx):
        x[i, j] = (i) * dx
        y[i, j] = (j) * dy

# Initial condition
for j in np.arange(0, ny):
    for i in np.arange(0, nx):
        # left up
        if x[i, j] < 0.5 and .5<y[i,j]:
            p[i, j] = .3
            r[i, j] = .5323
            u[i, j] =  1.206
            v[i, j] = 0
        # left lower
        elif x[i, j] <= 0.5 and 0.5 >= y[i,j]:
            p[i, j] = .029
            r[i, j] = .138
            u[i, j] = 1.206
            v[i, j] = 1.206
        # right upper
        elif x[i, j] >= 0.5 and 0.5 <= y[i,j]:
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
        else:
            print("O NOOOO")
        E[i, j] = p[i, j] / (gamma - 1) + 0.5 * r[i, j] * (u[i, j] ** 2 + v[i, j] ** 2)
        c[i, j] = np.sqrt(gamma * p[i, j] / r[i, j])

# Q_new will be the solution vector
Qnew[:, :, 0] = r
Qnew[:, :, 1] = r * u
Qnew[:, :, 2] = r * v
Qnew[:, :, 3] = E

t = 0
while t < tMax:
    dt =CFL * min(dx, dy) / np.max(c + np.sqrt(u * u + v * v))
    for j in np.arange(0, ny):
        for i in np.arange(0, nx):
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

    # Rusanov method
    for j in range(0, ny - 1):
        for i in range(0, nx - 1):
            ah1[i, j] = max(abs(u[i, j]) + c[i, j], abs(u[i + 1, j]) + c[i + 1, j])
            ah2[i, j] = max(abs(v[i, j]) + c[i, j], abs(v[i, j + 1]) + c[i, j + 1])

            Fh[i, j, :] = 0.5 * (F[i + 1, j, :] + F[i, j, :] - ah1[i, j] * (Q[i + 1, j, :] - Q[i, j, :]))
            Gh[i, j, :] = 0.5 * (G[i, j, :] + G[i, j + 1, :] - ah2[i, j] * (Q[i, j + 1, :] - Q[i, j, :]))

    # Advance the solution in time
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            Qnew[i, j, :] = Qnew[i, j, :] - (dt / dx) * (Fh[i, j, :] - Fh[i - 1, j, :]) - (dt / dy) * (
                    Gh[i, j, :] - Gh[i, j - 1, :])

    Qnew[:, 0,:]  = Qnew[:, 1,:]
    Qnew[0,:,:]  = Qnew[2,:,:]
    Qnew[:, ny-1,:] = Qnew[:, ny - 2,:]
    Qnew[nx-1,:,:] = Qnew[nx - 2,:,:]
    # Primitive variables
    r = Qnew[:, :, 0]
    u = Qnew[:, :, 1] / r
    v = Qnew[:, :, 2] / r
    E = Qnew[:, :, 3]
    p = (gamma - 1) * (E - 0.5 * r * (u * u + v * v))
    c = np.sqrt(gamma * p / r)
    t = t + dt  # Advance  time
plt.figure(3)
plt.imshow(p)
plt.figure(1)
plt.imshow(r)
plt.figure(2)
plt.quiver(u,v)
print("DONE")
