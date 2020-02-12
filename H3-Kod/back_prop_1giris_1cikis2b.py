# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as grafik
from matplotlib.pyplot import plot

def linear(n):
    return n

def sigmoid(x):
    y=1/(1+np.exp(-x))
    return y

#VERI SETI ---------------------------------------------
X1=1;X2=2
Y1=2;Y2=5;
#Egitileck parametrelerin baslangic degerleri
W1=-0.5
b1=0.1
W2=-0.3
b2=0.2
print('W1=',W1,'b1=',b1,'W2=',W2,'b2=',b2)
#ogrenme oranı
alfa=0.1#learning rate
epoch=500
HATA=np.zeros((epoch,))
for i in range(epoch):
    X=X1;Y=Y1
    #1. katmanin cikisi
    y1=sigmoid( W1*X+b1)
    #2. katmanin cikisi
    y2=linear(W2*y1+b2) #W2*y1, linear: f(n)=n
    #print('y=',y2)
    
    #hata: 
    e=Y-y2    
    print('e1:',e)    
    HATA[i]=np.abs(e)       
    #GERI YAYILIM
    F2=1
    d2=-2*F2*e 
    # 2. Katmandaki parametreleri yenile
    W2=W2-alfa*d2*y1
    b2=b2-alfa*d2    
    
    F1= (1-y1)*y1
    d1= F1*W2*d2   
    #1. Katmandaki parametreler
    W1=W1-alfa*d1*X #X(i)' 
    b1=b1-alfa*d1    
    #print('W1=',W1,'b1=',b1,'W2=',W2,'b2=',b2)
    
    
    X=X2;Y=Y2
    #1. katmanin cikisi
    y1=sigmoid( W1*X+b1)
    #2. katmanin cikisi
    y2=linear(W2*y1+b2) #W2*y1, linear: f(n)=n
    #print('y=',y2)
    
    #hata: 
    e=Y-y2   
    HATA[i]=HATA[i]+np.abs(e) 
    print('e2:',e)    
            
    #GERI YAYILIM
    F2=1
    d2=-2*F2*e 
    # 2. Katmandaki parametreleri yenile
    W2=W2-alfa*d2*y1
    b2=b2-alfa*d2    
    
    F1= (1-y1)*y1
    d1= F1*W2*d2   
    #1. Katmandaki parametreler
    W1=W1-alfa*d1*X #X(i)' 
    b1=b1-alfa*d1    
    #print('W1=',W1,'b1=',b1,'W2=',W2,'b2=',b2)
print('W1=',W1,'b1=',b1,'W2=',W2,'b2=',b2)
X=X1;Y=Y1
    #1. katmanin cikisi
y1=sigmoid( W1*X+b1)
    #2. katmanin cikisi
y2=linear(W2*y1+b2) #W2*y1, linear: f(n)=n
print('X=',X1,'y=',y2)
print(HATA)
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(HATA)


