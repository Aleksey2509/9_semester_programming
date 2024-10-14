from math import log, ceil

start = float(input())
end = float(input())
proc = float(input())

rel = end / start
months = int(ceil(log(rel) / log(1 + proc / 100)))
print(months)
