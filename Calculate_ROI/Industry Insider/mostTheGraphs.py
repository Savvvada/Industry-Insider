import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
import matplotlib.patches as mpatches
import seaborn as sea


# Visualizing 3-D numeric data with Scatter Plots
# length, breadth and depth
if __name__ == '__main__':

    val = input("There are 2 graphs to view in this section. \n Would you like to see the 3d graph  that has:revenue, us ranking, dates, and genres? \n Or perhaps the heat map? of the most correlative data? \n enter 1 for heatmap or 2 for 3d graph    ")
    if val == str(2) :
        color_source = pd.read_csv('outputting/output4.csv') # I needed this unprocessed csv to get the index of the genres in the processed one, they have the same index for data objects it's just that in this csv I hadn't yet created new columns for every genre (i.e this one doesn't have genres one-hot encoded)
        music_ROI = pd.read_csv('ProcessedData.csv')
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        xs = music_ROI['0']
        ys = music_ROI['5']
        zs = music_ROI['4']
        unique_genres = color_source['11']
        n = 16
        freq = unique_genres.value_counts()[:n].index.tolist()
        # print('these are the top 16 most popular genres, to reduce confusion we\'ll restrict our color key in the plot' + str(freq))
        unique_genres = unique_genres.drop_duplicates()
        unique_genres = unique_genres.sort_values()
        print(unique_genres.size)
        colormap = []

        for i in range(unique_genres.size):
            colormap.append('#%06X' % randint(0, 0xFFFFFF))

        colormap.sort() #There are 72 distinct genres, which are stored. some combinations include: rap, rap/hip hop, rap/pop. Therefore it made sense to me to sort the colors so similar named genres have a higher chance of getting similar colors
        # print(type(colormap)) most of these commented line are simply test values
        unique_genres.tolist()
        colors = dict(zip(unique_genres, colormap)) #The scatter method accepts a parameters that specifies the colors based on a received value. I passed a data dictionary appended to a randomizer for html color codes.
        # print(colors)
        color2 = []

        ax.scatter(xs, ys, zs, c=color_source['11'].apply(lambda x: colors[x]), s=50, alpha=0.8, edgecolors='none')

        ax.set_xlabel('Dates: 2000 - 2020, starting at 0')
        ax.set_ylabel('Peak U.S. Rankings, the lower the better')
        ax.set_zlabel('1-Month Revenue after release')

        color_key = []
        for i in colors:
            for j in freq:
                if j == i:
                    color_key.append(mpatches.Patch(color=colors[j], label=j))
                    color2.append(colors[j])
                    print(i,j)

        ax.legend(handles = color_key)

        ax.grid(True)
        plt.show() # Finally got it lol, I must say that the legend does not include all the colors as some genres only have 1 artist for them, it becomes MUCH easier to differentiate what is what when one zooms in (right click).

    if val == str(1):

        dataset = pd.read_csv('ProcessedData.csv', header=0)
        dataset = dataset[['4','5','6','7','8','9','10']]
        print(type(dataset))
        print(dataset.columns)
        A = (dataset.loc[:, ['4','5','6','7','8','9','10']]
                .applymap(lambda v: int(v) if float else np.nan)
                .dropna()
        ).corr()
        sea.heatmap(A, annot=True)
        plt.show()
