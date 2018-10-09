import skymapper as skm
import matplotlib.pyplot as plt
import inspect

if __name__ == "__main__":

    # cycle through all defined projections and show the full sky
    # with default graticules
    args = {"ra_0": 0}
    conic_args = {"ra_0":0,
                "dec_0": -10,
                "dec_1": -40,
                "dec_2": 10
                }

    for name, proj_cls in skm.projection_register.items():
        proj = None
        signature = inspect.signature(proj_cls.__init__)
        try:
            proj = proj_cls(**args)
        except TypeError:
            try:
                proj = proj_cls(**conic_args)
            except TypeError:
                pass
                
        if proj is not None:
            map = skm.Map(proj, interactive=False)
            map.grid()
            map.fig.suptitle(name)
            map.show()
