Your task is to **write, by reasoning, every weight of a small neural network that does the light transport in a
volumetric cloud renderer**, so that its renders of one cumulus cloud look like the path-traced references: hyper-
realistic light scattering through the cloud, including the colour change from noon to sunset.

This is in the spirit of Disney's *Deep Scattering* (Kallweit et al., 2017), which trained a network on path-traced
data to predict the light arriving at points inside a cloud. Here nobody trains the network. You design it: which
inputs it reads, its layer sizes, each unit's activation, and every weight and bias, from the physics and from
looking at your own renders next to the references.

## Start here

Read `KIT.md` in this directory first. It gives the scene (cloud, medium, phase function, sun, sky), the renderer
and the exact role of the network's output, the probes the network can read, the `network.json` format and limits,
the 12 reference pairs, and the tools. Then look at the reference images in `references/`.

The renderer is fixed. Your whole output is the network and your account of it. You cannot change the renderer,
the probes or the tools.

## What you deliver

In `work/`:

1. `work/network.json`: your final network.
2. The script or scripts that generate it (for example `work/build_network.py`), so that running them with
   `./python3` rewrites `work/network.json` exactly. The script that builds `network.json` must not read any image
   (`.npy` or `.png`): every number in it is a literal you chose or a physics formula.
3. `work/NETWORK.md`: what every unit computes and why, layer by layer; the physical reasoning behind the design;
   the revisions you made after looking at renders, and why; what still differs from the references and why; the
   command that rebuilds `network.json`. Keep it up to date as you go, not only at the end.

## Tools

- `./render NET.json --shot PAIR_ID [--compare]` renders your network (also `--set reference`, and free cameras and
  sun positions); use the Read tool to look at the PNGs.
- `./inspect NET.json --shot PAIR_ID --pixel X Y` shows every probe value and every unit's activation along one
  pixel's ray.
- `./python3` runs your own scripts (numpy only).

`KIT.md` section 7 gives the details. The Bash tool runs only these three commands. Use Glob, Grep and Read to
list and read files. You can write only inside `work/`.

## Rules

1. **No training, fitting, optimisation or automated search of any kind.** In particular: no gradient descent or
   other iterative update of weights toward a target, however it is written; no least squares or regression in any
   form (including normal equations with `solve`/`inv`/`svd`, or `statistics.linear_regression`); no weight, scale
   or colour computed in code from reference pixels or from the difference between a render and a reference (for
   example `k = ref.mean()/render.mean()`): measure, think, then type the number you choose into your script with a
   comment saying why; no rendering units or terms as separate images and combining those images by computation to
   choose weights (the image is linear in the last layer, so that is a least-squares fit); no sweeps: rendering
   several values of one setting and keeping whichever scores best is a search, even by hand one call at a time, so
   change a value because you have a reason, then look; no making the network recognise a particular reference shot
   or preset (by view direction, sun elevation or height range) to correct it there: every term must make physical
   sense for any camera and any sun.
2. Computing values from physical formulas, measuring your renders and the references, and simulating light
   transport yourself with numpy on idealised shapes you build (for example Monte Carlo through a slab or a sphere)
   are all fine: read the numbers to inform your choices, but do not fit, regress or solve against them; type the
   numbers you choose. Your simulations take shapes you build as input, not the reference images, and not the
   cloud's density beyond what `./render` and `./inspect` show you.
3. No scipy, torch, sklearn or similar libraries; numpy only.
4. Don't modify or work around the tools or the `./python3` guard, and don't read anything outside this directory,
   except the file Claude Code itself names when it saves a long tool output.
5. Keep `work/NETWORK.md` up to date: what every unit computes.

## How to work

There is no hurry. The budget for this session is large, so take the time the problem deserves: think it through,
render often, look closely at your renders beside the references (all 12 pairs, and free views and sun positions
too, since the network will also be judged on views and sun positions you are not shown), and keep revising.
Quality is the goal.

Write each file with its own tool call as soon as it is settled, so that work is never lost if a response is cut
short.

## Ending

When `work/network.json` is your final network, `work/NETWORK.md` is complete and the generating script reproduces
the network, end your final message with this line, on its own:

NETWORK COMPLETE

Write that line only when you are finished. The session ends when you write it.
