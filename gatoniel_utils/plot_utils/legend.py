from matplotlib.lines import Line2D


def custom_legend(
    ax,
    colors=[], colorlabels=[],
    markers=[], markerlabels=[],
    linestyles=[], linestylelabels=[],
):
    custom_lines = [
        Line2D([0], [0], color=color) for color in colors
    ] + [
        Line2D(
            [0], [0], color="gray", marker=marker
        ) for marker in markers
    ] + [
        Line2D(
            [0], [0], color="black", linestyle=linestyle
        ) for linestyle in linestyles
    ]
    custom_descr = colorlabels + markerlabels + linestylelabels
    ax.legend(custom_lines, custom_descr)
