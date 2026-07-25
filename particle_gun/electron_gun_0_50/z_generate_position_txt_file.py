import numpy

POS_X=-30.0     # position x [cm]. -10 -45
POS_Y=40.0      # position y [cm].   52.5, 15
POS_Z=340.0     # position z [cm].  280-350


N=10000
z = numpy.random.uniform(280, 350, N)

with open("z_position.txt", "w") as f:
    for i in range(N):
        f.write(f"{z[i]}\n")
print("z_position.txt file is created.")
