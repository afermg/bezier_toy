#!/usr/bin/env jupyter
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from splines import Bernstein


# Minimal case of cubic bezier curves
def generate_segments(
    vertices: tuple[int, int], controls: tuple[tuple[tuple[int, int]]]
) -> np.ndarray:
    """
    Convert vertices and control points into a format usable for splines.Bernstein

    # To generate a circle
    vertices = ((-1,0), (1,1), (1,-1))
    controls = (((-2,1),(-1,2)), ((2,1), (2,0)), ((1,-2),(-2,-2)))
    generate_closed_cubic_bezier(vertices, controls)
    """
    vertices = np.array(vertices)
    controls = np.array(controls)

    assert len(vertices) == len(controls), "Vertices and control sizes do not match"
    segments = np.zeros((len(controls), controls.shape[1] + 2, 2))

    segments[:, 1:-1, :] = controls
    segments[:, 0, :] = vertices
    segments[:, -1, :] = np.roll(vertices, -1, axis=0)

    return segments


def plot_spline(
    segments: np.ndarray, ts: int = 50, vertices=True, controls=True
) -> np.ndarray:
    """
    For some reason Bernstein is only providing the first segment.
    To bypass this I loop over my segments to build multiple ones.
    """
    ts = np.linspace(0, 1, ts)
    splines = [
        Bernstein(
            [
                segment,
            ]
        )
        for segment in segments
    ]
    results = np.vstack([list(map(spline.evaluate, ts)) for spline in splines])
    plt.scatter(x=results[:, 0], y=results[:, 1])

    if vertices:
        plt.scatter(*segments[:, 0].T, color="red")

    if controls:
        plt.scatter(*segments[:, 1:-1, :].reshape((-1, 2)).T, color="orange")


def plot_bezier(vertices, controls, plot_kwargs):
    """

    > import numpy as np
    > vertices = ((0, 0), (0, 0.8))
    > controls = np.array(((1.8, 0.4), (1.5, 1.2)))  # Half-heart shape
    > M = np.array((-1, 1))  # Mirror
    > plot_bezier(
    >     vertices=vertices,
    >     controls=(controls, (M * controls)[::-1]),
    >     plot_kwargs={"ts": 200},
    > )
    """
    segments = generate_segments(vertices=vertices, controls=controls)
    plot_spline(segments, vertices=True, controls=True, **plot_kwargs)


# %%
# Heart bezier curve

vertices = ((0, 0), (0, 0.8))
controls = np.array(((1.8, 0.4), (1.5, 1.2)))  # Half-heart shape
M = np.array((-1, 1))  # Mirror
plot_bezier(
    vertices=vertices,
    controls=(controls, (M * controls)[::-1]),
    plot_kwargs={"ts": 200},
)
plt.savefig("heart_bezier.png")
