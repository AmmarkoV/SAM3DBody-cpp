
#Octave script using symbolic math
#sudo apt-get install octave octave-symbolic
pkg load symbolic

syms sinX cosX z
syms sinY cosY
syms sinZ cosZ
syms ZXY

X = [ [1, 0, 0, 0];
      [0, cosX, sinX, 0];
      [0, -sinX, cosX, 0];
      [0, 0, 0, 1];
      ]
      
Y = [ [cosY, 0, -sinY, 0];
      [0, 1, 0, 0];
      [sinY, 0, cosY, 0];
      [0, 0, 0, 1];
      ]      
      
Z = [ [cosZ, sinZ, 0, 0];
      [-sinZ, cosZ, 0, 0];
      [0, 0, 1, 0];
      [0, 0, 0, 1];
      ]      
      
ZXY = Z * X * Y    
