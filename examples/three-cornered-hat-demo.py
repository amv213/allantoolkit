"""
 Allan deviation tools
 Anders Wallin (anders.e.e.wallin "at" gmail.com)
 v1.0 2014 January

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

  Three-cornered-hat test
 
  See http://www.wriley.com/3-CornHat.htm
"""

import numpy
import matplotlib.pyplot as plt # only for plotting, not required for calculations

import allantools
from allantools import noise

def plotallan_phase(plt,y,rate,taus, style, label):
    (t2, ad, ade,adn) = allantools.mdev(y,data_type='phase',rate=rate,taus=taus)
    plt.loglog(t2, ad, style, label=label)

# plot a line with the slope alpha
def plotline(plt, alpha, taus,style):
    y = [ pow(tt,alpha) for tt in taus]
    plt.loglog(taus,y,style)
    
if __name__ == "__main__":
    print("allatools three-cornered-hat demo")
    # we test ADEV etc. by calculations on synthetic data
    # with known slopes of ADEV

    t = [xx for xx in numpy.logspace( 0 ,4,50)] # tau values from 1 to 1000
    plt.subplot(111, xscale="log", yscale="log")

    N=10000
    rate = 1.0
    # white phase noise => 1/tau ADEV
    d = numpy.random.randn(4*N)
    phaseA = d[0:N] # numpy.random.randn(N) #pink(N)
    phaseA = [1*x for x in phaseA]
    phaseB = d[N:2*N] #numpy.random.randn(N) #noise.pink(N)
    phaseB = [5*x for x in phaseB]
    phaseC = d[2*N:3*N] #numpy.random.randn(N) #noise.pink(N)
    phaseC = [5*x for x in phaseC]

    phaseAB = [a-b for (a,b) in zip(phaseA,phaseB)]
    phaseBC = [b-c for (b,c) in zip(phaseB,phaseC)]
    phaseCA = [c-a for (c,a) in zip(phaseC,phaseA)]

    (taus,devA,err_a,ns_ab) = allantools.three_cornered_hat_phase(phaseAB,phaseBC,phaseCA,rate,t, allantools.mdev)
    print("TCH devA")
    
    plotallan_phase(plt, phaseA, 1, t, 'ro', 'true A phase')
    print("phaseA")
    plotallan_phase(plt, phaseB, 1, t, 'go', 'true B phase')
    print("phaseB")
    plotallan_phase(plt, phaseC, 1, t, 'bo', 'true C phase')
    print("phaseC")
    
    plotallan_phase(plt, phaseAB, 1, t, 'r.', 'AB measurement')    
    print("phaseAB")
    plotallan_phase(plt, phaseBC, 1, t, 'g.', 'BC measurement')
    print("phaseBC")
    plotallan_phase(plt, phaseCA, 1, t, 'b.', 'CA measurement')
    print("phaseCA")
    
    plt.loglog(taus, devA, 'rv', label='3-C-H estimate for A')
    plt.legend()
    plt.grid()
    plt.show()
