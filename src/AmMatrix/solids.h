
#define phi  1.6180339887498948482045868343656381177203091798057628621354486227
#define phi2 2.6180339887498948482045868343656381177203091798057628621354486226
#define phi3 4.2360679774997896964091736687312762354406183596115257242708972453
 

const float polyhedraCoordinates[]=
{
    0.0  , 0.0  , 0.0    ,  //<- this is not a valid coordinate triplet it is the center and only used to pad number to start from index 1..
    //---------------------------------
    
    0.0  ,  0.0  , 2*phi2, // Vertex 1  
    phi2 ,  0.0  ,   phi3, // Vertex 2
    phi  ,  phi2 ,   phi3, // Vertex 3
    0.0  ,  phi  ,   phi3, // Vertex 4
    -phi ,  phi2 ,   phi3, // Vertex 5
    -phi2,  0.0  ,   phi3, // Vertex 6
    -phi , -phi2 ,   phi3, // Vertex 7
    0.0  , -phi  ,   phi3, // Vertex 8
    phi  , -phi2 ,   phi3, // Vertex 9
    phi3 ,  phi  ,   phi2, // Vertex 10
    phi2 ,  phi2 ,   phi2, // Vertex 11
    0.0  ,  phi3 ,   phi2, // Vertex 12
    -phi2,  phi2 ,   phi2, // Vertex 13
    -phi3,  phi  ,   phi2, // Vertex 14
    -phi3, -phi  ,   phi2, // Vertex 15
    -phi2, -phi2 ,   phi2, // Vertex 16
    0.0  , -phi3 ,   phi2, // Vertex 17
    phi2 , -phi2 ,   phi2, // Vertex 18
    phi3 , -phi  ,   phi2, // Vertex 19
    
    phi3 , 0.0   ,   phi, // Vertex 20
    phi2 , phi3  ,   phi, // Vertex 21
   -phi2 , phi3  ,   phi, // Vertex 22
   -phi3 , 0.0   ,   phi, // Vertex 23
   -phi2 , -phi3 ,   phi, // Vertex 24
    phi2 , -phi3 ,   phi, // Vertex 25
    
  2*phi2 , 0.0   ,   0.0, // Vertex 26
    phi3 , phi2  ,   0.0, // Vertex 27
    phi  , phi3  ,   0.0, // Vertex 28
    0.0  ,2*phi2 ,   0.0, // Vertex 29
   -phi  , phi3  ,   0.0, // Vertex 30
   -phi3 , phi2  ,   0.0, // Vertex 31
  -2*phi2, 0.0   ,   0.0, // Vertex 32
   -phi3 ,-phi2  ,   0.0, // Vertex 33
   -phi  ,-phi3  ,   0.0, // Vertex 34
    0.0  ,-2*phi2,   0.0, // Vertex 35
    phi  ,-phi3  ,   0.0, // Vertex 36
    phi3 ,-phi2  ,   0.0, // Vertex 37
    
    phi3 , 0.0   ,  -phi, // Vertex 38
    phi2 , phi3  ,  -phi, // Vertex 39
   -phi2 , phi3  ,  -phi, // Vertex 40
   -phi3 , 0.0   ,  -phi, // Vertex 41
   -phi2 ,-phi3  ,  -phi, // Vertex 42
    phi2 ,-phi3  ,  -phi, // Vertex 43
    
    phi3 , phi   , -phi2, // Vertex 44
    phi2 , phi2  , -phi2, // Vertex 45
    0.0  , phi3  , -phi2, // Vertex 46
   -phi2 , phi2  , -phi2, // Vertex 47
   -phi3 , phi   , -phi2, // Vertex 48
   -phi3 ,-phi   , -phi2, // Vertex 49
   -phi2 ,-phi2  , -phi2, // Vertex 50
    0.0  ,-phi3  , -phi2, // Vertex 51
    phi2 ,-phi2  , -phi2, // Vertex 52
    phi3 ,-phi   , -phi2, // Vertex 53

    phi2 , 0.0   , -phi3, // Vertex 54
    phi  , phi2  , -phi3, // Vertex 55
    0.0  , phi   , -phi3, // Vertex 56
   -phi  , phi2  , -phi3, // Vertex 57
   -phi2 , 0.0   , -phi3, // Vertex 58
   -phi  , -phi2 , -phi3, // Vertex 59
    0.0  , -phi  , -phi3, // Vertex 60
    phi  ,-phi2  , -phi3, // Vertex 61
    0.0  , 0.0   ,-2*phi2 // Vertex 62
};

const char numberOfTetrahedronVertices = 4;
const char tetrahedronVertices[]=
{
 4, 34, 38, 47  
};

const char numberOfTetrahedronFaces = 12;
const char tetrahedronFaces[]=
{
    4, 34, 47, 
    4, 38, 34, 
    4, 47, 38, 
    34, 38, 47
};

 

const char numberOfCubeVertices = 8;
const char cubeVertices[]=
{
    4, 18, 23, 28, 34, 38, 47, 60 
};


const char numberOfCubeFaces = 36;
const char cubeFaces[]=
{ 
   4, 18, 38, 
   38, 28, 4, 
   4, 23, 18, 
   18, 23, 34,

   4, 28, 47,
   4, 47, 23,
   28, 38, 60,
   28, 60, 47,
   
   23, 47, 34,
   47, 60, 34, 
   38, 18, 60, 
   18, 34, 60
};
   
const char numberOfOctahedronVertices = 6;
const char octahedronVertices[]=
{ 
    7, 10, 22, 43, 49, 55
};
   
   
const char numberOfOctahedronFaces = 24;
const char octahedronFaces[]=
{
 7, 10, 43, 
 7, 22, 10, 
 7, 43, 49, 
 7, 49, 22,
 
 55, 10, 43,
 55, 22, 10, 
 55, 43, 49, 
 55, 49, 22
};
   
   
const char numberOfRhombicDodecahedronVertices = 14;
const char rhombicdodecahedronVertices[]=
{
  4, 7, 10, 18, 22, 23, 28, 34, 38, 43, 47, 49, 55, 60
};

const char numberOfRhombicDodecahedronFaces = 72;
const char rhombicdodecahedronFaces[]=
{
   7, 4, 18,
   10, 4, 18, 
   7, 18, 34, 
   43, 18, 34,

   7, 34, 23, 
   49, 34, 23, 
   7, 4, 23, 
   22, 4, 23,
   
   22, 4, 28,  
   10, 4, 28, 
   18, 10, 43, 
   38, 10, 43,

   34, 49, 43, 
   60, 49, 43, 
   23, 49, 22, 
   47, 49, 22,
   
   55, 38, 60,
   43, 38, 60, 
   55, 60, 47, 
   49, 60, 47,

   55, 47, 28,
   22, 47, 28, 
   55, 28, 38, 
   10, 28, 38
}; 


const char numberOfDodecahedronVertices = 20;
const char dodecahedronVertices[]=
{
  4, 8, 11, 13, 16, 18, 20, 23, 28, 30, 34, 36, 38, 41, 45, 47, 50, 52, 56, 60 
};

const char numberOfDodecahedronFaces = 108;
const char dodecahedronFaces[]=
{
 4, 8, 11, 
 11, 8, 18,
 11, 18, 20,
 
 4, 13, 23, 
 4, 23, 8, 
 8, 23, 16,

 4, 11, 28, 
 4, 28, 30, 
 4, 30, 13,
 
 8, 16, 34, 
 8, 34, 18, 
 18, 34, 36,
 
 11, 20, 28,
 20, 45, 28, 
 20, 38, 45,
 
 13, 30, 23, 
 23, 30, 41, 
 41, 30, 47,

 16, 23, 34,
 34, 23, 50, 
 50, 23, 41,

 18, 36, 52,
 18, 52, 38, 
 18, 38, 20,
 
 28, 45, 56,
 28, 56, 47,
 28, 47, 30,
 
 34, 50, 60, 
 34, 60, 36, 
 36, 60, 52,

 38, 52, 60,
 38, 60, 56, 
 38, 56, 45,
 
 41, 47, 56,
 41, 56, 60, 
 41, 60, 50   
};



const char numberOfIcosahedronVertices = 12;
const char icosahedronVertices[]=
{
  2, 6, 12, 17, 27, 31, 33, 37, 46, 51, 54, 58
};



const char numberOfIcosahedronFaces = 60;
const char icosahedronFaces[]=
{
 2, 6, 17, 
 2, 12, 6, 
 2, 17, 37, 
 2, 37, 27,
 
 2, 27, 12, 
 37, 54, 27, 
 27, 54, 46, 
 27, 46, 12,
 
 12, 46, 31,
 12, 31, 6,
 6, 31, 33, 
 6, 33, 17,
 
 17, 33, 51, 
 17, 51, 37, 
 37, 51, 54, 
 58, 54, 51,
 
 58, 46, 54, 
 58, 31, 46,
 58, 33, 31, 
 58, 51, 33
};


const char numberOfRhombicTriacontahedronVertices = 32;
const char rhombictriacontahedronVertices[]=
{
  2, 4, 6, 8, 11, 12, 13, 16, 17, 18, 20, 23,
 27, 28, 30, 31, 33, 34, 36, 37, 38, 41, 45, 46,
 47, 50, 51, 52, 54, 56, 58, 60
};



const char numberOfRhombicTriacontahedronFaces = 180;
const char rhombictriacontahedronFaces[]=
{
 2, 4, 6, 
 6, 8, 2, 
 2, 11, 4, 
 4, 11, 12,

 4, 12, 13, 
 4, 13, 6, 
 6, 16, 8, 
 8, 16, 17,

 8, 17, 18, 
 8, 18, 2, 
 2, 18, 37, 
 2, 37, 20,

 2, 20, 27, 
 2, 27, 11, 
11, 27, 28, 
11, 28, 12,

 6, 13, 31, 
 6, 31, 23, 
 6, 23, 33, 
 6, 33, 16,

54, 60, 58, 
58, 56, 54, 
54, 56, 45, 
45, 56, 46,

56, 58, 47, 
47, 46, 56, 
47, 58, 41, 
41, 31, 47,

58, 50, 33, 
33, 41, 58, 
58, 60, 51, 
51, 50, 58,

60, 54, 52, 
52, 51, 60, 
54, 38, 37, 
37, 52, 54,

45, 27, 38, 
38, 54, 45, 
20, 37, 38, 
38, 27, 20,

23, 31, 41, 
41, 33, 23, 
12, 28, 46, 
46, 30, 12,

12, 30, 31, 
31, 13, 12, 
31, 30, 46, 
46, 47, 31,

28, 27, 45, 
45, 46, 28, 
17, 34, 51, 
51, 36, 17,

18, 17, 36, 
36, 37, 18, 
37, 36, 51, 
51, 52, 37,

17, 16, 33, 
33, 34, 17, 
34, 33, 50, 
50, 51, 34
};


const char numberOfOnehundredandtwentyhedronVertices = 62;
const char onehundredandtwentyhedronVertices[]=
{
 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
57, 58, 59, 60, 61, 62
};

const int numberOfOnehundredandtwentyhedronFaces = 360;
const char onehundredandtwentyhedronFaces[]=
{
 1, 2, 4, 
 2, 3, 4, 
 2, 20, 10, 
 2, 10, 11, 
 2, 11, 3,

 3, 11, 12, 
 3, 12, 4, 
20, 26, 27, 
20, 27, 10, 
10, 27, 11,

11, 27, 21, 
11, 21, 12, 
21, 27, 28, 
12, 21, 28, 
12, 28, 29,

 1, 4, 6, 
 4, 12, 5, 
 4, 5, 6, 
 5, 12, 13, 
 5, 13, 6,

 6, 13, 14, 
 6, 14, 23, 
12, 29, 30, 
12, 30, 22, 
12, 22, 13,

13, 22, 31, 
22, 30, 31, 
13, 31, 14, 
14, 31, 23, 
23, 31, 32,

 1, 6, 8, 
 6, 23, 15, 
 6, 15, 16, 
 6, 16, 7, 
 6, 7, 8,

 8, 7, 17, 
 7, 16, 17, 
23, 32, 33, 
15, 23, 33, 
16, 15, 33,

24, 16, 33, 
34, 24, 33, 
17, 16, 24, 
17, 24, 34, 
17, 34, 35,

 1, 8, 2, 
 8, 17, 9, 
 8, 9, 2, 
 9, 17, 18, 
 9, 18, 2,

 2, 18, 19, 
 2, 19, 20, 
17, 35, 36, 
17, 36, 25, 
17, 25, 18,

18, 25, 37, 
25, 36, 37, 
19, 18, 37, 
20, 19, 37, 
20, 37, 26,

27, 26, 38, 
27, 38, 44, 
27, 44, 45, 
27, 45, 39, 
27, 39, 28,

28, 39, 46, 
28, 46, 29, 
39, 45, 46, 
38, 54, 44, 
55, 45, 54,

45, 44, 54, 
45, 55, 46, 
46, 55, 56, 
55, 54, 56, 
56, 54, 62,

30, 29, 46, 
30, 46, 40, 
31, 30, 40, 
40, 46, 47, 
31, 40, 47,

31, 47, 48, 
31, 48, 41, 
31, 41, 32, 
46, 56, 57, 
47, 46, 57,

47, 57, 58, 
48, 47, 58, 
41, 48, 58, 
57, 56, 58, 
58, 56, 62,

33, 32, 41, 
33, 41, 49, 
33, 49, 50, 
33, 50, 42, 
33, 42, 34,

34, 42, 51, 
42, 50, 51, 
35, 34, 51, 
49, 41, 58, 
50, 49, 58,

50, 58, 59, 
51, 50, 59, 
51, 59, 60, 
59, 58, 60, 
60, 58, 62,

36, 35, 51, 
36, 51, 43, 
37, 36, 43, 
43, 51, 52, 
37, 43, 52,

37, 52, 53, 
37, 53, 38, 
37, 38, 26, 
51, 60, 61, 
52, 51, 61,

52, 61, 54, 
53, 52, 54, 
38, 53, 54, 
54, 61, 60, 
54, 60, 62
};