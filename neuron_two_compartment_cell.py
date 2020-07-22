# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:13:03 2020

@author: rdave
"""
#Create two compart neuron with soma and dendrite - very basic cell

from neuron import h
from neuron.units import ms, mV
import matplotlib.pyplot as plt

h.load_file('stdrun.hoc')

#Created class

class BallAndStick:
    def __init__(self, gid):
        self._gid = gid
        self._setup_morphology()
        self._setup_biophysics()
    def _setup_morphology(self):
        self.soma = h.Section(name="soma", cell=self)
        self.dend = h.Section(name='dend', cell=self)
        self.all = [self.soma, self.dend]
        self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = 12.6157
        self.dend.L = 200
        self.dend.diam = 1
    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 100 # Axial resistance in Ohm * cm
            sec.cm = 1 # Membrane capacitance in micro Farads / cm^2
        self.soma.insert('hh')
        for seg in self.soma:
            seg.hh.gnabar = 0.12 # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036 # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003 # Leak conductance in S/cm2
            seg.hh.el = -54.3 # Reversal potential in mV
        #Insert passive current in dendrite
        self.dend.insert('pas')
        for seg in self.dend:
            seg.pas.g = 0.001 # passive conductance in S/cm2
            seg.pas.e = -65 # leak reverse potential in mV
    def __repr__(self):
        return('BallAndStick[{}]'.format(self._gid))
    
my_cell = BallAndStick(0)

#Stimulate the cell

stim = h.IClamp(my_cell.dend(1))
stim.delay = 5
stim.dur = 1
stim.amp = 0.1

#Record membrane potential at center (0.5) for soma

soma_v = h.Vector().record(my_cell.soma(0.5)._ref_v)
t = h.Vector().record(h._ref_t)

#Run stimulation
h.finitialize(-65 * mV) # Initialize membrane potential to -65 mV
h.continuerun(25 * ms) # Run for 25 ms

#Plot results for soma
plt.figure()
plt.plot(t, soma_v)
plt.xlabel('t (ms)')
plt.ylabel('v (mV)')

#Change amplitude of current and graph onto another plot
plt.figure()
amps = [0.075 * i for i in range(1, 5)]
colors = ['green', 'blue', 'red', 'black']
for amp, color in zip(amps, colors):
    stim.amp = amp
    h.finitialize(-65 * mV)
    h.continuerun(25 * ms)
    plt.plot(t, list(soma_v), label=amp, color=color)
plt.legend()
    

#Plot soma and dendrite:
dend_v = h.Vector().record(my_cell.dend(0.5)._ref_v)
plt.figure()
amps = [0.075 * i for i in range(1, 5)]
colors = ['green', 'blue', 'red', 'black']
for amp, color in zip(amps, colors):
    stim.amp = amp
    h.finitialize(-65 * mV)
    h.continuerun(25 * ms)
    plt.plot(t, list(soma_v), label=amp, color=color)
    plt.plot(t, list(dend_v), linestyle='dashed', color=color)
plt.legend()

#Look at how number of segments on dendrite affects graph:
plt.figure()
amps = [0.075 * i for i in range(1, 5)]
colors = ['green', 'blue', 'red', 'black']
for amp, color in zip(amps, colors):
    stim.amp = amp
    for my_cell.dend.nseg, width in [(1, 2), (101, 1)]: #making the 1 51 makes both lines the same
        h.finitialize(-65 * mV)
        h.continuerun(25 * ms)
        plt.plot(t, list(soma_v), label=amp if my_cell.dend.nseg == 1 else None,linewidth = width, color=color)
        plt.plot(t, list(dend_v), linestyle = 'dashed', linewidth = width, color=color)
plt.legend()

