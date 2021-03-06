
# Standard library imports
import json
import itertools
import time

# Third party imports
from bokeh.models import Label
# from bokeh.util.compiler import TypeScript
from bokeh.models import ColumnDataSource, Whisker
from bokeh.plotting import figure, show, output_file
import numpy as np


def size_product(tup):
    prod = 1
    for x in tup:
        prod *= x
    return prod


def base_agg():
    return {
        'float': {
            'cv2': [],
            'torch.flip': [],
            'flip (3000)': [],
            'flip (32768)': [],
            'indexing': []
        },
        'uint8': {
            'cv2': [],
            'torch.flip': [],
            'flip (3000)': [],
            'flip (32768)': [],
            'indexing': []
        }
    }


with open('benchmark_results.json', 'r') as f:
    results = json.load(f)

agg = {
    'horizontal': base_agg(),
    'vertical': base_agg()
}

sizes = []

for size_entry in results:
    size = size_entry['size']
    times = size_entry['timing']
    sizes.append(size_product(size))
    for direction in {'horizontal', 'vertical'}:
        direction_results = times[direction]
        for dtype in {'float', 'uint8'}:
            sources_results = direction_results[dtype]
            for source in sources_results:
                source_results = sources_results[source]
                time_mean = np.mean(source_results)
                time_std = np.std(source_results)
                agg[direction][dtype][source].append((time_mean, time_std))

colors = [('red', 'orangered', 'orange'), ('blue', 'dodgerblue', 'teal'),
          ('green',  'yellowgreen', 'limegreen'),
          ('chocolate', 'saddlebrown', 'brown'),
          ('indigo', 'darkslateblue', 'mediumslateblue')]
legends = ['(H ≪ W)', '', '(H ≫ W)']

# actual_sizes = [x[0] for x in itertools.groupby(sizes)]

for direction in agg:
    direction_agg = agg[direction]
    for dtype in direction_agg:
        sources_results = direction_agg[dtype]

        p = figure(title=f'{direction.capitalize()} flip ({dtype})',
                   height=600, width=800)
        p.xaxis.axis_label = 'Number of elements'
        p.yaxis.axis_label = 'Time (s)'
        p.xaxis.axis_label_text_font = 'DIN'
        p.xaxis.major_label_text_font = 'DIN'
        p.yaxis.axis_label_text_font = 'DIN'
        p.yaxis.major_label_text_font = 'DIN'
        p.yaxis.major_label_text_font = 'DIN'
        p.title.text_font = 'DIN'
        # p.output_backend = 'png'

        for source, group_colors in zip(sources_results, colors):
            source_results = sources_results[source]
            sized_results = zip(sizes, source_results)
            result_groups = itertools.groupby(
                sized_results, key=lambda x: x[0])
            groups = []
            for _, result_group in result_groups:
                result_group = tuple(result_group)
                groups.append(result_group)
            groups = list(zip(*groups))
            for group, color, legend in zip(groups, group_colors, legends):
                actual_sizes, points = zip(*group)
                mean_times, std_times = zip(*points)
                p.circle(actual_sizes, mean_times,
                         legend_label=f"{source} {legend}",
                         line_color=color, fill_color=color)
                p.line(actual_sizes, mean_times, line_color=color,
                       legend_label=f"{source} {legend}")
                err_xs = []
                err_ys = []

                # for x, y, yerr in zip(actual_sizes, mean_times, std_times):
                #     err_xs.append((x, x))
                #     err_ys.append((y - yerr, y + yerr))

                # p.multi_line(err_xs, err_ys, line_color=color,
                #              legend_label=f'{source} {legend}')

        p.legend.location = "top_left"
        p.legend.label_text_font = "DIN"
        p.legend.label_text_font_style = "normal"
        show(p)
        time.sleep(0.2)
