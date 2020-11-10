import lxml
from bs4 import BeautifulSoup as bs
from itertools import groupby
import pprint
import numpy as np
import matplotlib.pyplot as plt

content = []
with open('soph-ot-treebank.xml', 'r') as file:
     content = file.readlines()
     content = "".join(content)
     bs_content = bs(content, 'lxml')

words = bs_content.find_all("word")

lemmas_total = []
for x in words:
    if x.get("lemma"):
        lemmas_total.append(x.get("lemma"))

lemmas_alnum = [x for x in lemmas_total if x.isalnum()]

lemma_frequencies = [(key, len(list(group))) for key, group in groupby(sorted(lemmas_alnum))]
lemma_frequencies_greatest = lemma_frequencies.sort(key=lambda y:y[1])

# show top 40 frequencies
sample = lemma_frequencies[::-1][0:100]

lemmas = [x[0] for x in sample]
frequencies = [x[1] for x in sample]
x_pos = np.arange(len(lemmas))

plt.bar(x_pos, frequencies, align='center')
plt.xticks(x_pos, lemmas, rotation='vertical')
plt.ylabel('Number of Occurrences')
plt.show()



"""
https://www.kite.com/python/answers/how-to-sort-a-list-of-tuples-by-the-second-value-in-python

Plotting:
- vertical x labels: https://stackoverflow.com/questions/1221108/barchart-with-vertical-labels-in-python-matplotlib
- bar plot from list of tuples: https://stackoverflow.com/questions/13925251/python-bar-plot-from-list-of-tuples
- matplotlib, plot docs: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.plot.html
"""
