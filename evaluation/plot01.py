#!/usr/local/bin/python3

# import modules
#import matplotlib.pyplot as plt
#import numpy as np
#from sys import argv
#script, filename = argv
#txt = open(filename)
#print(txt.read())
#plt.plot([1,2,3,4])
#plt.ylabel('some numbers')
#plt.show();

import csv
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

script, filename = argv

f = open(filename, 'r')
reader = csv.reader(f, delimiter=',')
rows = [x for x in reader]
algos = [row[0] for row in rows]
nums = [row[1:] for row in rows]
#nums.pop() # remove total value
del nums[-1]
#print(algos)
#print(nums)

N = len(nums[0])
#print(N)

ind = np.arange(N)
data = np.array(nums, dtype='float')

p = plt.subplot(111)

rects = []
for i in range(data.shape[0]):
    r = p.barh(ind, data[i], height = 0.2, left = np.sum(data[:i], axis = 0), label = algos[i])
    rects.append(r)

count = 0
for rs in rects:
    for r in rs:
        width = r.get_width()
        p.text(r.get_x() + width / 2, r.get_y() + 0.05, '%d' % count,
                ha='center',va='bottom')
        count += 1


p.yaxis.set_visible(False)
ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1000))
p.xaxis.set_major_formatter(ticks)
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.spines['left'].set_visible(False)

box = p.get_position()
p.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.1])
handles, labels = p.get_legend_handles_labels()
# for h in handles:
#     print(h)
plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-1.5), frameon=False, shadow=False, ncol=4)

plSize = plt.gcf()
print(plSize)

plt.xlabel('Time [ms]')
plt.title(filename.replace('.txt', ''))
plt.show()
