Code for the Interpolated Discretized (ID) Embedding
+
Alpha version of the SVM learning with stochastic subgradient decent.



Original code was taken from: 
Ofir Pele, Alexey Kurbatsky
Contact: ofir.pele@g.ariel.ac.il


contains the source code for computing the Interpolated Discretized (ID) embedding.

Please cite this paper if you use this code:
 Interpolated Discretized Embedding of Single Vectors and Vector Pairs for Classification, Metric Learning and Distance Approximation
 Ofir Pele, Yakir Ben-Aliz
 arXiv 2016
bibTex:
@article{Pele-arXiv2016d,
  title={Interpolated Discretized Embedding of Single Vectors and Vector Pairs for Classification, Metric Learning and Distance Approximation},
  author = {Ofir Pele and Yakir Ben-Aliz},
  journal={arXiv},
  year={2016}
}


Pele et al. plan to publish algorithm with embedding on the fly in the future. Also hope to publish a python wrapper and a version that uses the GPU.

Startup
------------
I currently have LearningDebug as main. it runs a Learning demo of with a one dimensional hyper-axis, learning classification of the thresholding of euclidean/scalar distance. 

Compiling 
-----------------
In a linux shell:
>> make
>> LearningDebug




Licensing conditions
--------------------
See the file ID/LICENSE.txt for conditions of use.
