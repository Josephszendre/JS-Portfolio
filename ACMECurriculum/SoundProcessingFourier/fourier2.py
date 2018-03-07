
# coding: utf-8

# In[1]:

from matplotlib import pyplot as plt
from scipy.io import wavfile
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.signal import fftconvolve
class Signal(object):
    def __init__(self, sample_rate, samples):
        self.rate=sample_rate
        self.samples=samples
    
    def plot(self):
        plt.subplot(2,1,1)
        plt.plot(np.arange(0,len(self.samples))*1./self.rate,self.samples,"b-", lw=1,markersize=1, label="Samples")
        print len(self.samples)
        plt.legend(loc="upper left")
        
        transform=fft(self.samples)
        x_values=self.rate*np.arange(1,len(self.samples)+1)*1./len(self.samples)
        plt.subplot(2,1,2)
        plt.plot(x_values,transform,"b-",lw=2,markersize=1)
        plt.show()
                 
    def export(self,filename):
        i=max((max(self.samples),max(-1*self.samples)))
        wavfile.write(filename,self.rate,np.int16((self.samples/i)*32767))

    def import_from_file(self,filename):
        self.rate,self.samples = wavfile.read(filename)
        
    def clean(self):
        transform=fft(self.samples)
        for i in xrange(14600,60000):
            transform[i]=0.
            transform[-i]=0.           
        self.samples=ifft(transform)
        
    def add_reverb(self):
        balr,balloondata=wavfile.read("balloon.wav")
        print balloondata.shape
        self.samples=np.hstack((self.samples,np.zeros((rate*4))))
        transform1=fft(self.samples)
        transform2=fft(balloondata)
        transform2=np.hstack((transform2[:len(transform2)/2],np.zeros((len(transform1)-len(transform2)+1,)),transform2[:len(transform2)/2]))
        signal=ifft(transform1*transform2)
        self.samples=signal.copy()
        self.plot()
        self.export("convolvednew.wav")
        
        
        
                 

def prob1():
    rate,data=wavfile.read("Noisysignal2.wav")
    sig=Signal(rate,data)
    #sig.plot()
    sig.clean()   
    sig.plot()
    sig.export("cleaned.wav")


# In[2]:

def A():
    wave_a = lambda x: np.sin(2*x*np.pi*440)
    sig=Signal(44100,wave_a(np.arange(44100*5)))

    sig.export("./asdf.wav")


# In[3]:

def prob4(samples):
    res=[]
    temp=0j
    for i in xrange(len(samples)):
        temp=0j
        for j in xrange(len(samples)):
            temp+=samples[j]*np.exp(-2.*np.pi*1j*j*i/len(samples))
        res+=[temp]
         
    return res
    
    #FFT Ballono 


# In[4]:

def prob6(samples):
    wave_a = lambda x: np.sin(2*x*np.pi*440)
    wave_c = lambda x: np.sin(2*x*np.pi*493.88)
    wave_e = lambda x: np.sin(2*x*np.pi*523.25)
    wave_g = lambda x: np.sin(2*x*np.pi*783.99)
    _2sec=np.arange(44100*2)
    chord1_2s=wave_a(_2sec)+wave_c(_2sec)+wave_e(_2sec)
    chord2_2s=wave_c(_2sec)+wave_e(_2sec)+wave_g(_2sec)
    music=np.hstack((chord1_2s,chord2_2s))
    sig2=Signal(44100,music[87000:89000])
    sig2.export("asdfasdf.wav")
    
    

if __name__=="__main__":
    rate,data=wavfile.read("chopin.wav")
    sig=Signal(rate,data)
    sig.add_reverb()
    


# In[ ]:



