import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def generate_colors(num_colors):
    # Get the YGnBu colormap
    cmap = cm.get_cmap('summer')

    # Generate 10 equally spaced values between 0 and 1
    values = np.linspace(0, 1, num_colors)

    # Get the colors from the colormap
    colors = [cmap(value) for value in values]
    return colors

def draw_stacked_bars(dataframe: pd.DataFrame, colors = ['red', 'blue', 'green', 'yellow', 'oranges', 'pink'], title="Best AUC result component contribution"):
    columns = dataframe.columns
    colors = generate_colors(len(columns) * 10)
    r = list(range(len(columns)))

    # From raw value to percentage
    all_bars = []
    max_bar_length = 0
    for col in columns:
        model_total = len(dataframe[col])
        bars = np.array([(dataframe[col] == x).sum() for x in dataframe[col].unique()]) / model_total
        all_bars.append(bars.tolist())
        if len(bars) > max_bar_length:
            max_bar_length = len(bars)
    
    # fill other with zero
    for i, bars in enumerate(all_bars):
        reminder = max_bar_length - len(bars)
        all_bars[i] = bars + [0 for r in range(reminder)]
    all_bars = np.array(all_bars)

    # plot
    barWidth = 0.85

    # Create green Bars
    last_bar_items = np.zeros((all_bars.shape[0]))
    for i, bar_items in enumerate(all_bars.T):
        plt.bar(
            r,
            bar_items,
            bottom=last_bar_items,
            color=colors[i],
            edgecolor='white', width=barWidth)
        
        for j, item in enumerate(bar_items):
            if item > 0.01:
                print()
                plt.text(j, (last_bar_items[j] + item / 2), dataframe.iloc[:, j].unique()[i], horizontalalignment='center', size='small', color='black')
        
        last_bar_items += bar_items

    # Custom x axis
    plt.xticks(r, columns)
    plt.xlabel("group")
    plt.title(title)
    # Show graphic
    plt.show()