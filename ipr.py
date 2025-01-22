#!/usr/local/bin/python3
import numpy as np
import cv2
from typing import Optional
import matplotlib.pyplot as plt
from IPython import get_ipython


in_ipython = False
in_colab = False
try:
    get_ipython()
    in_ipython = True
    if "google.colab" in str(get_ipython()):
        in_colab = True
except:  # if get_ipython() is not available we do not run as script!
    pass


def pre_process_image(image: list, resize: float, flipRB: bool = False):
    image = np.asarray(image, dtype=np.uint8)

    # image.shape: Height, width, channels
    if flipRB and len(image.shape) == 3:  # Assume BGR, do a conversion
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize so pixels are larger
    if resize != 1:
        image = cv2.resize(
            image, np.array(image.shape[:2]) * resize, interpolation=cv2.INTER_NEAREST
        )

    return image


if in_ipython:
    import matplotlib.pyplot as plt

    def imshow(title: str, image: list, *args, resize=1, **kwargs):
        """Shows an image in a Jupyter Notebook.

        Args:
            title (str): the title of the window.
            image (list): the image to show.
            resize (float, optional): resize the image. Defaults to 1.
            *args: any additional args passed to matplotlib.pyplot.imshow.
            **kwargs: any additional kwargs passed to matplotlib.pyplot.imshow.
        """
        image = pre_process_image(image, resize)
        plt.figure(figsize=(10, 10))
        # Draw the image
        plt.imshow(image, *args, **kwargs)
        plt.title(title)
        # We'll also disable drawing the axes and tick marks in the plot, since it's actually an image
        plt.axis("off")
        # Make sure it outputs
        plt.show()


else:

    def imshow(title: str, img: list, *args, resize=1, **kwargs):
        """Shows an image in a window.

        Args:
            title (str): the title of the window.
            image (list): the image to show.
            resize (float, optional): resize the image. Defaults to 1.
            *args: any additional args passed to cv2.imshow.
            **kwargs: any additional kwargs passed to cv2.imshow.
        """
        img = pre_process_image(img, resize, flipRB=True)
        cv2.imshow(title, img, *args, **kwargs)


def draw_polygon(
    polygon: list, edge_color="purple", vertex_color="orange", show_indices: bool = True
):
    """Draws a polygon.

    Args:
        polygon (list): A list of vertices to draw.
        edge_color (str|color, optional): color for the edges. Defaults to 'purple'.
        vertex_color (str|color, optional): color for the vertices. Defaults to 'orange'.
        show_indices (bool, optional): draw the index number. Defaults to True.
    """
    if not polygon or len(polygon) < 2:
        return  # nothing to draw

    vertices = polygon.copy()
    vertices.append(vertices[0])
    xs, ys, zs = zip(*vertices)  # create lists of x and y values

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(xs, ys, color=edge_color)
    plt.plot(xs, ys, "o", color=vertex_color)
    if show_indices:
        for i, (x, y) in enumerate(zip(xs[:-1], ys[:-1])):
            plt.text(x, y, f"{i}")  # for i in range(len(xs))])

    ax.set_aspect("equal", adjustable="box")
    plt.show()  # show the plot


def draw_triangles(
    triangles_or_vertices: list,
    indices: list = None,
    edge_color="purple",
    vertex_color="orange",
    show_indices: bool = True,
):
    """Draws triangles by a list of vertices with and without indices.

    Args:
        triangles_or_vertices (list): A list of vertices
        indices (list, optional): If defined a list of indices for the triangles to draw. Defaults to None.
        edge_color (str|color, optional): color for the edges. Defaults to 'purple'.
        vertex_color (str|color, optional): color for the vertices. Defaults to 'orange'.
        show_indices (bool, optional): draw the index number. Defaults to True.
    """
    if not triangles_or_vertices or len(triangles_or_vertices) < 2:
        return  # nothing to draw

    vertices = triangles_or_vertices.copy()
    # vertices.append(vertices[0])
    xs, ys, zs = zip(*vertices)  # create lists of x and y values

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if indices:
        for i in range(0, len(indices), 3):
            idx = [*indices[i : i + 3], indices[i % len(indices)]]
            plt.plot([xs[ii] for ii in idx], [ys[ii] for ii in idx], color=edge_color)
            plt.plot(
                [xs[ii] for ii in idx[:-1]],
                [ys[ii] for ii in idx[:-1]],
                "o",
                color=vertex_color,
            )
    else:
        for i in range(0, len(xs), 3):
            plt.plot([*xs[i : i + 3], xs[i]], [*ys[i : i + 3], ys[i]], color=edge_color)
            plt.plot(xs[i : i + 3], ys[i : i + 3], "o", color=vertex_color)

    if show_indices:
        for i, (x, y) in enumerate(zip(xs, ys)):
            plt.text(x, y, f"{i}")  # for i in range(len(xs))])

    ax.set_aspect("equal", adjustable="box")
    plt.show()  # show the plot
