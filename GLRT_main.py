import json

from galahad import glrt
import numpy as np
import torch
import time

from problems.problems import problems
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: glrt")

start = time.time()
PROBLEM_TYPE = "softmax-cubic_None_MNISTsb"
LAMBDA = 0.1
n, func = problems(PROBLEM_TYPE, LAMBDA)
# set problem data

power = 3.0
v = torch.zeros(n,dtype = torch.double)
hv = torch.zeros(n,dtype = torch.double)
g = func(v,"1")

# set default options
options = glrt.initialize()

# set some non-default options
options['unitm'] = False
options['print_level'] = 0
#print("options:", options)

# load non-default options
glrt.load_options(options)

# first problem

print(" solve problem 1")
status = 1
weight = 1e3
r = g.numpy()
v = v.numpy()
i = 1
res = {}

while status > 0:
  status, x, r, v = glrt.solve_problem(status, n, power, weight, r, v)
  inform = glrt.information()
  res[i] = {"inform": inform, "x": x}
  i += 1
  if status == 2: # precondition v -> v/2
    for i in range(n):
      v[i] = v[i] / 2.0;
  elif status == 3: # form v -> H v
    hv[0] = 2.0 * v[0] + v[1]
    for i in range(1,n-1):
       hv[i] = v[i-1] + 2.0 * v[i] + v[i+1]
    hv[n-1] = v[n-2] + 2.0 * v[n-1]
    for i in range(n):
      v[i] = hv[i]
  elif status == 4: # reset r
    r = g.numpy()

# get information

end = time.time()
print(" time taken:", end-start)
print("Results:", res)
with open("Results/glrt"+PROBLEM_TYPE+".json", "w") as f:
    f.write(f"Results: {res}\n")
    f.write(f"Time taken: {end - start}\n")
