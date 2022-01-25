# Binding-Affinity-Prediction

This is the project work done during my internship in Machine Learning. I have implemented K-deep paper and worked in the prediction of Binding affinity of protein-ligands using 3D convolutional neural network. 

# Data: PDBbind (v.2016) database. Worked on the refined dataset.
PDBbind(v.2016)database,containing13,308protein−ligandcomplexesandtheir corresponding experimentally determined binding affinities collected from literature and the Protein Data Bank (PDB), in terms of a dissociation (Kd), inhibition (Ki) or half- concentration (IC50) constant.

A smaller refined subset(4057Protein-ligandcomplexes)is extracted from it following quality protocols addressing structural resolution and experimental precision of the binding measurement.

# Overflow
![alt text](https://github.com/yashikesarwani/Binding-Affinity-Prediction/blob/master/Related%20documents/Overflow.png?raw=true)


# Model used is squeezenet.

Architectural Design Strategies
Strategy 1. Replace 3×3 filters with 1×1 filters
Strategy 2. Decrease the number of input channels to 3×3 filters
Strategy 3. Down-sample late in the network so that convolution layers have large activation maps
Strategies 1 and 2 are about judiciously decreasing the quantity of parameters in a CNN while attempting to preserve accuracy. Strategy 3 is about maximizing accuracy on a limited budget of parameters
