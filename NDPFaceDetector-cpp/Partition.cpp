#include "Partition.h"

int Find(arma::uvec &parent,int x){
	int root = parent(x);
	if (root != x){
		root = Find(parent,root);
	}
	return root;
}



bool Partition(arma::mat &A, arma::uvec &label, int &nGroups){

	// to do
	int N = A.n_rows;
	arma::uvec parent(N);
	arma::uvec rank;

	for (int i = 0; i < N; i++){
		parent(i) = i;
	}
	rank.zeros(N, 1);
	arma::uvec ori_parent(parent);

	for (int i = 0; i < N; i++){
		// check equal items
		for (int j = 0; j < N; j++){
			if (A(i, j) == 0){
				continue;
			}

			// find root of node i and compress path
			int root_i = Find(parent, i);

			// find root of node j and compress path
			int root_j = Find(parent, j);

			// union both trees
			if (root_j != root_i){
				if (rank(root_j) < rank(root_i)){
					parent(root_j) = root_i;
				}
				else if (rank(root_i) < rank(root_j)){
					parent(root_i) = root_j;
				}
				else{
					parent(root_j) = root_i;
					rank(root_i) = rank(root_i) + 1;
				}
			}
		}
	}

	//parent.print("parent:");
	//rank.print("rank:");

	// label each element
	arma::uvec flag = parent == ori_parent;
	nGroups = arma::sum(flag);
	label.zeros(N,1);

	// matlab: label(flag) = 1:nGroups
	int t = 1;
	for (int i = 0; i < N; i++){
		if (flag(i)){
			label(i) = t++;
		}
	}
	//label.print("label");


	int root_i;
	for (int i = 0; i < N; i++){
		if (parent(i) == i){
			continue;
		}

		// find root of node i
		root_i = Find(parent, i);
		label(i) = label(root_i);
	}

	//label.print("label:");
	//cout << "nGroups:" << nGroups << endl;

	return true;
}