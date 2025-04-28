# Daa-Project
Dense Subgraph Computation Using Network Flow
This project implements Algorithm-1 and Algorithm-4 from the paper "[Insert Paper Title Here]" to find dense subgraphs in given networks using network flow techniques. The algorithms are implemented in C++ and tested on several datasets as mentioned in the paper's experiment section.
Installation
To run this project, Compile the code:
g++ -std=c++11 -o dense_subgraph densest_subgraph.cpp
Usage
To run the program, ensure that the dataset files are in the same directory as the executable. Then, simply execute:
./dense_subgraph
Datasets
The following datasets are used in this project:
•	AST33.txt
•	netscience.txt
•	Ca-HepTh.txt
These files should be placed in the same directory as the executable. 
Team Members and Contributions
•	2022A7PS0132H Sahiti Kasina: Exact Algorithm
•	2022A7PS0059H Valavala Charan Teja: Exact Algorithm
•	2022A7PS1323H Saksham Daga: CoreExact Algorithm
•	2022A7PS1796H Aryan Saini: CoreExact Algorithm
•	2022A7PS0227H Kunal Maheshwari: Analysis and WebPage
Project Webpage
For more information, visit our project webpage: https://charan119.github.io/
Additional Notes
•	The enumeration of k-cliques for h > 3 might take significant time for larger datasets due to the computational complexity.
•	Ensure that the dataset files are correctly formatted as per the code's expectations.

