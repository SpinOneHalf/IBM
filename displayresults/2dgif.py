from utils import simulation, RiamanProblemInit
import matplotlib.pyplot as plt
import glob
import contextlib
from PIL import Image
import numpy as np
import os


tf = .03
tlocal = 0.
nx = 5750
ny = 5750
r, u, v, p, E, c, dx, dy = RiamanProblemInit(nx, ny)

dtstep = .001
stepint = 0
while tlocal <= tf:
    tlocal += dtstep
    (r, u, v, p, E, c, tlocal) = simulation(r, u, v, E, p, c,
                                            dx, dy, ny, nx, tlocal)
    print(f'local {tlocal}')
    fig = plt.figure(1)
    plt.imshow(r)
    plt.savefig(f"temp/roe_step_{stepint}.png")
    stepint += 1
# filepaths
fp_in = "temp/roe_step_*.png"
fp_out = "image_roe.gif"

# use exit stack to automatically close opened images
with contextlib.ExitStack() as stack:
    stuff = glob.glob(fp_in)
    idxs = []
    for pic in stuff:
        idx = int(pic.split('step_')[-1].split('.')[0])
        idxs.append(idx)
    # lazily load images
    images=np.array(stuff)[np.argsort(idxs)]
    imgs = (stack.enter_context(Image.open(f))
            for f in images)

    # extract  first image from iterator
    img = next(imgs)

    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)
