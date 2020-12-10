"""

##########################
#  test stable32plot.py  #
##########################

""" 
import allantoolkit
from pylab import figure,show,plot
from stable32plot import sigmaplot,dataplot#import 2 functions: sigmaplot,dataplot


"""#------------generate random data and cal adev-----------------"""
x1 = allantoolkit.noise.white(1000)
(taus, adevs, errors, ns) = allantoolkit.adev(x1)
(taust, adevst, errorst, nst) = allantoolkit.tdev(x1)
(tauso, adevso, errorso, nso) = allantoolkit.oadev(x1)

x2=allantoolkit.noise.white(1000, 0.6)
(taus2,adevs2,errors2,ns2)=allantoolkit.oadev(x2)

x3=allantoolkit.noise.white(1000, 0.5)
(taus3,adevs3,errors3,ns3)=allantoolkit.oadev(x3)

x4=allantoolkit.noise.white(1000, 0.4)
(taus4,adevs4,errors4,ns4)=allantoolkit.oadev(x4)

x5=allantoolkit.noise.white(1000, 0.3)
(taus5,adevs5,errors5,ns5)=allantoolkit.oadev(x5)

x6=allantoolkit.noise.white(1000, 0.2)
(taus6,adevs6,errors6,ns6)=allantoolkit.oadev(x6)

xn=allantoolkit.noise.white(1000, 0.1)
(tausn,adevsn,errorsn,nsn)=allantoolkit.oadev(xn)

xf=allantoolkit.noise.white(1000, 1e-9)

xp1=allantoolkit.noise.brown(1000, 1e-9)
xp2=allantoolkit.noise.brown(500, 1e-10)


"""#--------------eg1: plot single adev curve with text list----------------"""
figure()
sigmaplot(taus,adevs,errors)

"""#--------------eg2: plot single adev curve with noise floor of the test device--------"""
figure()
sigmaplot(taus,adevs,errors,"test 1",taun=taus,sigman=adevsn)

"""#--------------eg3: plot single tdev curve with text list----------------"""
figure()
sigmaplot(taust,adevst,errorst,"test 1",sigmatype="tdev",
          taulist=[1,16,32],textloc=3,legendloc=1)

"""#--------------eg4: plot multiple adev curve with noise floor of the test device"""
figure()
sigmaplot(tauso,adevso,errorso,"test 1",
         tau2=taus2,sigma2=adevs2,error2=errors2,legend2="test 2",
         tau3=taus3,sigma3=adevs3,error3=errors3,legend3="test 3",
         tau4=taus4,sigma4=adevs4,error4=errors4,legend4="test 4",
         tau5=taus5,sigma5=adevs5,error5=errors5,legend5="test 5",
         tau6=taus6,sigma6=adevs6,error6=errors6,legend6="test 6",
         taun=tausn,sigman=adevsn,legendn="test 1 noise floor",
         sigmatype="oadev",sigmatext=False)

"""#-----------------eg5: plot single frequency data-----------"""
figure()
dataplot(range(len(xf)),xf)

"""#--------------eg6: plot multiple phase data----------------"""
figure()
dataplot(range(len(xp1)),xp1,"test 1",
         sec2=range(len(xp2)),data2=xp2,legend2="test 2",
         datatype='phase')

show()


