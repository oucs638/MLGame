import os
from os import path

count = [0, 0, 0]
for file in os.listdir("./log"):
    filename = str(file)
    count[int(filename[10])-1] += 1
for level in range(1, 4):
    print(f"LEVEL {level}: {count[level-1]}")
total = count[0] + count[1] + count[2]
print(f"Total: {total}")
