import random

rng = random.Random()
total = 0

for i in range(1000):
    total += rng.random()

print(total/1000)