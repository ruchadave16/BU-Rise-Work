# this script implements a single unit with feedback. Modified
# from Tom Anastasio's mfile to Python code.
# Modified for BU's RISE Practicum Comp Neuro lab by mbezaire@bu.edu

# TO DO: Add the commands to import the necessary
#        packages for this script
import numpy as np
import matplotlib.pyplot as plt

###############################
# Set up simulation
###############################

# set input flag (1 for impulse, 2 for step)
inFlag=2 


# TO DO: Define the minimum and maximum
#        allowable values of the activity.
#        To start, set the minimum (cut) to negative
#        infinity and the maximum (sat) to positive
#        infinity. Hint: Infinity can be calculated
#        as float('inf')
cut = -float('inf')
sat = float('inf')

# TO DO: Define a parameter, tEnd, that 
#        gives the last time step in the
#        in the simulation (we want 100 steps):
tEnd = 100

# TO DO: Define a parameter, nTs, that 
#        gives the number of time steps in
#        the simulation, in terms of tEnd:
nTs = tEnd + 1

# TO DO: Define a parameter for the input
#        weight, and another for the feedback
#        weight, according to the values
#        suggested in the background of the
#        exercise.
v = 1
w = 1.1

###############################
# Define input stimulation
###############################

# TO DO: Create a placeholder input vector of 
#        zeros called x, where there is
#        one element for each time step that
#        will be simulated:
x = np.zeros((1, nTs))

start=11 # set a start time for the input

if inFlag==1: # if the input should be a pulse
# TO DO: then set the input at only one time point
	x[0,start] = 1
elif inFlag==2: # if the input instead should be a step, then 
# TO DO: then set the input and keep it up until the end
	x[0,start:] = 1

print(x)



###############################
# Run simulation
###############################

# TO DO: Create a placeholder output vector called y
#        with an element for each time step that will be
#        executed during the simulation:
y = np.zeros((1, nTs))


# TO DO: Run the simulation. Use a for loop and the range function
#        to iterate through each time step and compute the activity
#        of the model. Check whether the model activity is too high
#        or too low, and reset it to within the allowable bounds if
#        necessary:
for t in range(nTs):
	y[0,t] = w*y[0,t-1] + v*x[0,t-1]

	if y[0,t] < cut:
		y[0,t] = cut
	if y[0,t] > sat:
		y[0,t] = sat



# TO DO: Create a time vector called tBase that ranges from 0 ms
#        all the way through the tEnd-th time step (not stopping
#        one before the tEnd-th step)
tBase = range(0,tEnd + 1)


###############################
# Plot the results
###############################

fig = plt.figure()
ax1 = fig.add_subplot(211)
# TO DO: Plot the time vector as the
#        independent variable and the
#        x variable as the dependent
#        variable. Remember that x is
#        a numpy vector and so it has
#        two dimensions that must be
#        specified:
ax1.plot(tBase, x[0])
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Input')
ax1.set_xlim(0,tEnd)
ax1.set_ylim(0,1.1)
plt.title('A')

# TO DO: Set the xlabel to 'time step', 
#        the ylabel to 'input', and
#        the title to 'A':
# ...
# ...


# TO DO: Set the x axis range from 0 to tEnd, 
#        the y axis range from 0 to 1.1:
# ...
# ...


ax2 = fig.add_subplot(212)
ax2.plot(tBase, y[0])
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Output')
plt.title('B')

# TO DO: Plot the time vector as the
#        independent variable and the
#        y variable as the dependent
#        variable:
# ax2....


# TO DO: Set the xlabel to 'time step', 
#        the ylabel to 'output', and
#        the title to 'B':
# ...
# ...
# plt....
plt.show()

