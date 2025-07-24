from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from math import pi
import torch
import numpy as np

class GaussianViewer(ShowBase):
    def __init__(self, gaussians):
        super().__init__()

        self.disableMouse()
        self.camera.setPos(0, -5, 1)
        self.camLens.setFov(60)
        self.camLens.setNearFar(0.01, 100)

        self.gaussians = gaussians.squeeze(0).detach().cpu().numpy()
        self.load_gaussians()

    def load_gaussians(self):
        for g in self.gaussians:
            x, y, z = g[0:3]
            sx, sy, sz = g[3:6]
            r, g_, b = g[6:9]
            alpha = g[9]

            if alpha < 0.1:
                continue  # Skip transparent ones

            # Create an ellipsoid
            sphere = self.loader.loadModel("models/misc/sphere")
            sphere.setScale(sx, sy, sz)
            sphere.setPos(x, y, z)
            sphere.setColor(r, g_, b, alpha)
            sphere.reparentTo(self.render)

        print(f"✅ Rendered {len(self.gaussians)} Gaussians")

def visualize_gaussians_panda(gaussians: torch.Tensor):
    app = GaussianViewer(gaussians)
    app.run()
