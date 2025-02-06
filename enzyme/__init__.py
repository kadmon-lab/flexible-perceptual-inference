import os
from pathlib import Path

import seaborn as sns
# from enzyme.colors import *

import numpy
numpy.set_printoptions(threshold=50)

try:
    import jax
    jax.numpy.set_printoptions(threshold=50)
except ImportError:
    pass

PRJ_ROOT = Path(__file__).parents[1]
SRC_ROOT = PRJ_ROOT / "enzyme"
TMPDIR = Path.home() / "Downloads"

import logging
import logging.config

logging.config.fileConfig(SRC_ROOT / 'logging.conf')
from matplotlib.colors import Normalize
 

diag = 27
aspect = 16 / 9
pix_width = 2560
pix_height = 1440
pix_diag = (pix_width ** 2 + pix_height ** 2) ** 0.5
dpi = pix_diag / diag
# print(f"Using monitor dpi of {dpi:.0f}")


MM = 0.0393701

INCH = 1
CM = 1 / 2.54 * INCH
PT = 1 / 72 * INCH

# 6-8 pt
FONTSIZE = 7 * 1

CLMN_1 = 85*MM
CLMN_1_5 = 114*MM
CLMN_FULL = 174*MM

INCH = 1
PAGEWIDTH = 8.5*INCH
PAGEHEIGHT = 11*INCH
TEXTHEIGHT = 9.5 * INCH # ballpark estimate
TEXTWIDTH = 5.5 * INCH


# cmap = sns.color_palette("crest", as_cmap=True)

TEXPATH = PRJ_ROOT / "tex_REFACTORIZED" 
FIGPATH = TEXPATH / "figures"
CACHE_DIR = PRJ_ROOT / "Data" / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# refine some settings
CMAP_CYCLIC = sns.color_palette("twilight", as_cmap=True)

# def init_mpl(usetex=False):
#     import matplotlib.pyplot as plt
#     cmap = plt.get_cmap("Greys_r")
#     MPLRCPATH = Path(__file__).parent / "matplotlibrc"
#     os.environ["MATPLOTLIBRC"] = str(MPLRCPATH)
#     import matplotlib as mpl
#     config = mpl.rc_params_from_file(MPLRCPATH, fail_on_error=True)
#     mpl.rcParams = config

#     from matplotlib.rcsetup import cycler

#     # my_cycler = cycler(color=colors)
#     mpl.rc('axes', prop_cycle=my_cycler)

#     # TeX support
#     mpl.rcParams["text.usetex"] = usetex
#     # mpl.rcParams["text.parse_math"] = usetex
#     if usetex:
#         mpl.rcParams["text.latex.preamble"]=(TEXPATH / "preamble.tex").read_text()


#     mpl.rcParams['figure.dpi'] = dpi
#     # set text size
#     mpl.rcParams["font.size"] = FONTSIZE

#     #mpl.rcParams["font.sans-serif"] = ["Open Sans"] 

#     mpl.rcParams['lines.markeredgewidth'] = mpl.rcParams['lines.markeredgewidth'] / 2

    

#     return plt

if __name__ == "__main__":
    print(PRJ_ROOT)
    plt = init_mpl()
    import matplotlib as mpl
    print(mpl.rcParams["text.latex.preamble"])
    fig, axes = plt.subplots(1,1)
    axes.set_title(r"$\E$")
    plt.show()
