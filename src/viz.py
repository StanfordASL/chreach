import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import ConvexHull
import numpy as np

def plot_ellipse(ax, mu, Q, additional_radius=0., color='blue', alpha=0.1, **kwargs):
    """
    Based on
    http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.
    """
    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(Q)
    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))
    # Eigenvalues give length of ellipse along each eigenvector
    w, h =  2. * (np.sqrt(vals) + additional_radius)
    ellipse = patches.Ellipse(mu, w, h, theta, color=color, alpha=alpha)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ax.add_artist(ellipse) 

def set_axes_equal(ax: plt.Axes):
    # reference: 
    # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    # reference: 
    # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

class Faces():
    # reference:
    # https://stackoverflow.com/questions/49098466/plot-3d-convex-closed-regions-in-matplotlib
    def __init__(self,tri, sig_dig=12):
        self.tri = np.around(np.array(tri), sig_dig)
        self.grpinx = list(range(len(tri)))
        norms = np.around([self.norm(s) for s in self.tri], sig_dig)
        _, self.inv = np.unique(norms,return_inverse=True, axis=0)

    def norm(self,sq):
        cr = np.cross(sq[2]-sq[0],sq[1]-sq[0])
        return np.abs(cr/np.linalg.norm(cr))

    def isneighbor(self, tr1,tr2):
        a = np.concatenate((tr1,tr2), axis=0)
        return len(a) == len(np.unique(a, axis=0))+2

    def order(self, v):
        if len(v) <= 3:
            return v
        v = np.unique(v, axis=0)
        n = self.norm(v[:3])
        y = np.cross(n,v[1]-v[0])
        y = y/np.linalg.norm(y)
        c = np.dot(v, np.c_[v[1]-v[0],y])
        h = ConvexHull(c)
        return v[h.vertices]

    def simplify(self):
        for i, tri1 in enumerate(self.tri):
            for j,tri2 in enumerate(self.tri):
                if j > i: 
                    if self.isneighbor(tri1,tri2) and \
                       self.inv[i]==self.inv[j]:
                        self.grpinx[j] = self.grpinx[i]
        groups = []
        for i in np.unique(self.grpinx):
            u = self.tri[self.grpinx == i]
            u = np.concatenate([d for d in u])
            u = self.order(u)
            groups.append(u)
        return groups

def plot_3d_convex_hull(points_3d, ax, color):
    # points_3d - (M, 3)
    hull = ConvexHull(points_3d)
    simplices = hull.simplices
    org_triangles = [points_3d[s] for s in simplices]
    faces = Faces(org_triangles)
    faces = faces.simplify()
    ax.scatter(
        points_3d[:, 0], 
        points_3d[:, 1], 
        points_3d[:, 2], 
        alpha=0)
    pc = Poly3DCollection(faces, alpha=0.3)
    pc.set_facecolor(color)
    pc.set_alpha(0.1)
    lc = Line3DCollection(faces, 
        colors=color, linewidths=0.4, linestyles=':')
    lc.set_alpha(0.5)
    ax.add_collection3d(pc)
    ax.add_collection3d(lc)
    return ax