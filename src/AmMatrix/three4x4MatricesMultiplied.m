#Octave script using symbolic math
#sudo apt-get install octave octave-symbolic
pkg load symbolic

syms a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
syms b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
syms c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
syms R

A = [ a0 a1 a2 a3;
      a4 a5 a6 a7;
      a8 a9 a10 a11;
      a12 a13 a14 a15;
      ]
B = [ b0 b1 b2 b3;
      b4 b5 b6 b7;
      b8 b9 b10 b11;
      b12 b13 b14 b15;
      ]      
C = [ c0 c1 c2 c3;
      c4 c5 c6 c7;
      c8 c9 c10 c11;
      c12 c13 c14 c15;
      ]      
R = A * B * C    
