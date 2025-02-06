import colorsys
import copy
from pathlib import Path
import pathlib
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
from matplotlib import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors


from seaborn import diverging_palette


    
def get_color_name(rgb_int_triplet):
    color_hex = rgb_to_hex(rgb_int_triplet)
    api_url = f"https://api.color.pizza/v1/?values={color_hex.lstrip('#')}&list=bestOf&noduplicates=true"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        color_data = response.json()
        color_name = color_data['colors'][0]['name']
        return color_name
    else:
        return "Error: Unable to fetch color name from the API."


def write_gpl(color_iterable, file, format="hex"):
    # if color_iterable is not a dict, get the keys as color names via get_color_names
    if not isinstance(color_iterable, dict):
        color_dict = dict(zip([get_color_name([int(c_*255) for c_ in mcolors.to_rgb(c)]) for c in color_iterable], color_iterable))
    else:
        color_dict = color_iterable

    def write_color_line(f, color, name, format, lightness=1.):
        rgb = color

        f.write(f"{int(rgb[0]*255)} {int(rgb[1]*255)} {int(rgb[2]*255)}        {name} {lightness}\n")

    # write palette to inkscape swatches .gpl
    name = file.stem
    with open(file, "w") as f:
        f.write(
f"""GIMP Palette
Name: {name}
Columns: 0
#
"""
        )
        for name, color in color_dict.items():
            if type(color) == dict:
                for lightness, color_ in color.items():
                    write_color_line(f, color_, name, format, lightness)
            else:
                write_color_line(f, color, name, format)       



import os
import sys
import platform

def get_inkscape_palettes_directory():
    if platform.system() == 'Windows':
        appdata = os.getenv('APPDATA')
        path = os.path.join(appdata, 'inkscape', 'palettes')
    elif platform.system() == 'Darwin':
        home = os.getenv('HOME')
        path = os.path.join(home, 'Library', 'Application Support', 'org.inkscape.Inkscape', 'config', 'inkscape', 'palettes')
    elif platform.system() == 'Linux':
        home = os.getenv('HOME')
        path = os.path.join(home, '.config', 'inkscape', 'palettes')
    else:
        raise Exception('Unsupported operating system.')
    
    return Path(path)

from pathlib import Path
import re

def write_color_macros_to_latex_preamble(color_dict, latex_preamble_path: Path):
    # Convert the path to a pathlib.Path object if it's not already one
    latex_preamble_path = Path(latex_preamble_path)
    
    # Create or open the LaTeX preamble file in read+write mode ('r+')
    with latex_preamble_path.open('a+') as f:
        # Move the read/write position to the start of the file
        f.seek(0)
        
        # Read the contents of the file
        contents = f.read()
        
        # Check if the xcolor package is already included in the file
        if "\\usepackage{xcolor}" not in contents:
            # Add the xcolor package include command
            contents += "\n\n% Include the xcolor package\n\\usepackage{xcolor}\n"
        
        # Initialize a list to keep track of new color definitions
        new_color_definitions = []
        
        # Loop through the color dictionary and update each color macro in the LaTeX file
        for color_name, hex_code in color_dict.items():
            # Check if the color macro already exists in the file
            pattern = f"\\definecolor{{{color_name}}}{{[^}}]*}}"
            if re.search(pattern, contents):
                # Replace the existing color macro with the new one
                contents = re.sub(pattern, f"\\definecolor{{{color_name}}}{{HTML}}{{{hex_code}}}", contents)
            else:
                # Store the new color macro for later addition
                new_color_definitions.append(f"\\definecolor{{{color_name}}}{{HTML}}{{{hex_code}}}")
        
        # Add any new color definitions to the end of the contents
        if new_color_definitions:
            contents += "\n" + "\n".join(new_color_definitions) + "\n"
        
        # Move the read/write position to the start of the file
        f.seek(0)
        
        # Truncate the file to 0 bytes (i.e., delete its contents)
        f.truncate()
        
        # Write the updated contents back to the file
        f.write(contents)

    
                
def adjust_colormap_hls(cmap, h_adjust=0, l_adjust=0, s_adjust=0):
    # Get the colormap's RGB values
    cmap_rgb = cmap(np.arange(cmap.N))
    
    # Convert RGB to HLS, adjust, and convert back to RGB
    new_colors = []
    for rgb in cmap_rgb:
        h, l, s = colorsys.rgb_to_hls(rgb[0], rgb[1], rgb[2])
        h = (h + h_adjust) % 1.0
        l = np.clip(l + l_adjust, 0, 1)
        s = np.clip(s + s_adjust, 0, 1)
        new_rgb = colorsys.hls_to_rgb(h, l, s)
        new_colors.append((new_rgb[0], new_rgb[1], new_rgb[2], rgb[3]))  # Preserve original alpha value
    
    # Create new colormap with modified colors
    new_cmap = mcolors.ListedColormap(new_colors)
    return new_cmap

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def hex_to_rgb(hex, norm1=False):
    hex = hex.replace("#", "")
    return tuple(int(hex[i: i + 2], 16) if not norm1 else float(int(hex[i: i + 2], 16)) / 255 for i in (0, 2, 4))



cmap_cool = adjust_colormap_hls(plt.get_cmap("cool"), h_adjust=0, l_adjust=-.03, s_adjust=-.45)
action_cmap = plt.get_cmap('YlOrRd')
cmap = cmap_cool

colors = [(1, 1, 1), (1, 1, 1), (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)]  # R -> G -> B
positions = [0, 0.4, 1]
cm_blue = mcolors.LinearSegmentedColormap.from_list("tab_blue", list(zip(positions, colors)), N=100)



# markers
ls_bayes = "--"
marker_bayes = "*"
c_bayes = "C0"
l_bayes = "Bayes"

ls_nw = "-"
marker_nw = "o"
c_nw = "C1"
l_nw = "RNN"

ls_fac = "dotted"
marker_fac = "x"
c_fac = "C7" #"C5"
l_fac = "RNN, untrained"

ls_mice = "-."
marker_mice = "d"
c_mice = "C3"
l_mice = "mice"

c_go = sns.color_palette("deep")[2]
c_nogo = sns.color_palette("deep")[3]

mec = "xkcd:dark grey"

c_Bs = "k"
c_Btheta = cmap(.5)

# write the cycler to a dict
my_cycler = plt.cycler(color=['k'])
# plt.rcParams.update({'axes.prop_cycle': my_cycler})

color_dict = {}
keys = copy.deepcopy(list(globals().keys()))
for k in keys:
    if k.startswith("c_"):
        v = globals()[k]
        color_dict[k[2:]] = mcolors.to_rgb(v)

# cycler colors
cycler_colors = {f"C{i}":c for i, c in enumerate(my_cycler.by_key()["color"])}
# get 10 steps from the colormap
cmap_colors = {f"cmap{i}": cmap(i/9) for i in range(10)}

color_dict.update(cycler_colors)
color_dict.update(cmap_colors)



if __name__ == "__main__":
    # Example usage
    from enzyme import TEXPATH
    from enzyme import init_mpl
    plt = init_mpl(usetex=False)

    # test_div_palette(CLASS_DIV)
    # # print(material_colors.keys())
    # print(open_colors.keys())
    # fig1 = plot_palette(colors)
    # fig2 = plot_palette(colors_muted)
    # fig1.show()
    # fig2.show()
    # if SHOW: show_plot()

    # Generate some data to plot
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.sin(Y)

    # Plot the data using the new colormap
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, Z, shading='auto', cmap=cmap_cool, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Modified RdBu_r Colormap")
    plt.show()

    
    latex_preamble_path = Path(TEXPATH / "preamble.tex")

    # write_color_macros_to_latex_preamble(color_dict, latex_preamble_path)

    file = TEXPATH / "enzyme.gpl"
    write_gpl(color_iterable=color_dict, file=file)

    # make a symlink to the Inkscape palletes directory
    inkscape_swatches_path = get_inkscape_palettes_directory()
    import os

    # Define the paths
    source_path = file
    target_path = inkscape_swatches_path / file.name

    # Create the symlink
    os.symlink(source_path, target_path)

