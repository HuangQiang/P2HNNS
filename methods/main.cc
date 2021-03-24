#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstring>

#include "def.h"
#include "util.h"
#include "ap2h.h"

using namespace p2h;

// -----------------------------------------------------------------------------
void usage() 						// display the usage of this package
{
	printf("\n"
		"-------------------------------------------------------------------\n"
		" Usage of the Package for Point-to Hyperplane NNS                  \n"
		"-------------------------------------------------------------------\n"
		"    -alg   {integer}  options of algorithms (0 - 17)\n"
		"    -n     {integer}  cardinality of the dataset\n"
		"    -qn    {integer}  number of queries\n"
		"    -d     {integer}  dimensionality of the dataset\n"
		"    -I     {integer}  is normalized or not\n"
		"    -m     {integer}  #hash tables (FH, FH-, NH)\n"
		"                      #single hasher of compond hasher (EH, BH, MH)\n"
		"    -l     {integer}  #hash tables (EH, BH, MH)\n"
		"    -M     {integer}  #proj vector used for a single hasher (MH)\n"
		"    -s     {integer}  scale factor of dimension (FH, FH-, NH)\n"
		"    -b     {float}    interval ratio (FH)\n"
		"    -w     {float}    bucket width (NH)\n"
		"    -cf    {string}   name of configuration\n"
		"    -dn    {string}   name of data set\n"
		"    -ds    {string}   address of data set\n"
		"    -qs    {string}   address of query set\n"
		"    -ts    {string}   address of truth set\n"
		"    -of    {string}   output folder\n"
		"\n"
		"-------------------------------------------------------------------\n"
		" The Options of Algorithms                                         \n"
		"-------------------------------------------------------------------\n"
		"    0  - Ground-Truth & Histogram & Heatmap\n"
		"         Param: -alg 0 -n -qn -d -dn -ds -qs -ts -of\n"
		"\n"
		"    1  - Linear-Scan\n"
		"         Param: -alg 1 -n -qn -d -dn -ds -qs -ts -of\n"
		"\n"
		"    2  - Random-Scan (Random Selection and Scan)\n"
		"         Param: -alg 2 -n -qn -d -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    3  - Sorted-Scan (Sort and Scan)\n"
		"         Param: -alg 3 -n -qn -d -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    4  - EH (Embedding Hyperplane Hash)\n"
		"         Param: -alg 4 -n -qn -d -I -m -l -b -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    5  - BH (Bilinear Hyperplane Hash)\n"
		"         Param: -alg 5 -n -qn -d -I -m -l -b -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    6  - MH (Multilinear Hyperplane Hash)\n"
		"         Param: -alg 6 -n -qn -d -I -m -l -M -b -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    7  - FH (Furthest Hyperpalne Hash)\n"
		"         Param: -alg 7 -n -qn -d -m -s -b -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    8  - FH^- (Furthest Hyperpalne Hash without Multi-Partition)\n"
		"         Param: -alg 8 -n -qn -d -m -s -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    9  - NH (Nearest Hyperpalne Hash with LCCS-LSH)\n"
		"         Param: -alg 10 -n -qn -d -m -w -s -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    10 - NH_Counting (Nearest Hyperpalne Hash with QALSH)\n"
		"         Param: -alg 10 -n -qn -d -m -s -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    11 - FH without Sampling\n"
		"         Param: -alg 11 -n -qn -d -m -b -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    12 - FH^- without Sampling\n"
		"         Param: -alg 12 -n -qn -d -m -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    13 - NH without Sampling\n"
		"         Param: -alg 13 -n -qn -d -m -w -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    14 - NH_Counting without Sampling\n"
		"         Param: -alg 14 -n -qn -d -m -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    15 - Orig_EH (Original Embedding Hyperplane Hash)\n"
		"         Param: -alg 15 -n -qn -d -m -l -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    16 - Orig_BH (Original Bilinear Hyperplane Hash)\n"
		"         Param: -alg 16 -n -qn -d -m -l -cf -dn -ds -qs -ts -of\n"
		"\n"
		"    17 - Orig_MH (Original Multilinear Hyperplane Hash)\n"
		"         Param: -alg 17 -n -qn -d -m -l -M -cf -dn -ds -qs -ts -of\n"
		"\n"
		"-------------------------------------------------------------------\n"
		"\n\n\n");
}

// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
	srand(6); // srand((unsigned) time(NULL)); 	// 

	char conf_name[200];	// name of configuration
	char data_name[200];	// name of dataset
	char data_set[200];		// address of data set
	char query_set[200];	// address of query set
	char truth_set[200];	// address of ground truth file
	char output_folder[200];// output folder

	int    alg    = -1;		// which algorithm?
	int    n      = -1;		// cardinality
	int    qn     = -1;		// query number
	int    d      = -1;		// dimensionality
	int    I      = -1;		// is normalized or not (0 - No; 1 - Yes)
	int    M      = -1;		// #proj vectors for a single hasher (MH)
	int    m      = -1;		// #hash tables (FH,FH-,NH), #single hasher (EH,BH,MH)
	int    l      = -1;		// #hash tables (EH,BH,MH)
	int    s      = -1;		// scale factor of dimension (s > 0) (FH,FH-,NH)
	float  b      = -1.0f;	// interval ratio (0 < b < 1) (FH)
	float  w      = -1.0f;	// bucket width (NH)
	float  *data  = NULL;	// data set
	float  *query = NULL;	// query set
	Result *R     = NULL;	// k-NN ground truth
	int    cnt    = 1;
	
	while (cnt < nargs) {
		if (strcmp(args[cnt], "-alg") == 0) {
			alg = atoi(args[++cnt]); assert(alg >= 0);
			printf("alg       = %d\n", alg);
		}
		else if (strcmp(args[cnt], "-n") == 0) {
			n = atoi(args[++cnt]); assert(n > 0);
			printf("n         = %d\n", n);
		}
		else if (strcmp(args[cnt], "-qn") == 0) {
			qn = atoi(args[++cnt]); assert(qn > 0);
			printf("qn        = %d\n", qn);
		}
		else if (strcmp(args[cnt], "-d") == 0) {
			d = atoi(args[++cnt]) + 1; assert(d > 1); // add 1 
			printf("d         = %d\n", d);
		}
		else if (strcmp(args[cnt], "-I") == 0) {
			I = atoi(args[++cnt]); assert(I==0 || I==1);
			printf("I         = %d\n", I);
		}
		else if (strcmp(args[cnt], "-m") == 0) {
			m = atoi(args[++cnt]); assert(m > 0);
			printf("m         = %d\n", m);
		}
		else if (strcmp(args[cnt], "-l") == 0) {
			l = atoi(args[++cnt]); assert(l > 0);
			printf("l         = %d\n", l);
		}
		else if (strcmp(args[cnt], "-M") == 0) {
			M = atoi(args[++cnt]); assert(M > 2);
			printf("M         = %d\n", M);
		}
		else if (strcmp(args[cnt], "-s") == 0) {
			s = atoi(args[++cnt]); assert(s > 0);
			printf("s         = %d\n", s);
		}
		else if (strcmp(args[cnt], "-b") == 0) {
			b = atof(args[++cnt]); assert(b > 0.0f && b < 1.0f);
			printf("b         = %.2f\n", b);
		}
		else if (strcmp(args[cnt], "-w") == 0) {
			w = atof(args[++cnt]); assert(w > 0.0f);
			printf("w         = %.2f\n", w);
		}
		else if (strcmp(args[cnt], "-cf") == 0) {
			strncpy(conf_name, args[++cnt], sizeof(conf_name));
			printf("conf_name = %s\n", conf_name);
		}
		else if (strcmp(args[cnt], "-dn") == 0) {
			strncpy(data_name, args[++cnt], sizeof(data_name));
			printf("data_name = %s\n", data_name);
		}
		else if (strcmp(args[cnt], "-ds") == 0) {
			strncpy(data_set, args[++cnt], sizeof(data_set));
			printf("data_set  = %s\n", data_set);
		}
		else if (strcmp(args[cnt], "-qs") == 0) {
			strncpy(query_set, args[++cnt], sizeof(query_set));
			printf("query_set = %s\n", query_set);
		}
		else if (strcmp(args[cnt], "-ts") == 0) {
			strncpy(truth_set, args[++cnt], sizeof(truth_set));
			printf("truth_set = %s\n", truth_set);
		}
		else if (strcmp(args[cnt], "-of") == 0) {
			strncpy(output_folder, args[++cnt], sizeof(output_folder));
			printf("folder    = %s\n", output_folder);

			int len = (int) strlen(output_folder);
			if (output_folder[len - 1] != '/') {
				output_folder[len] = '/';
				output_folder[len + 1] = '\0';
			}
			create_dir(output_folder);
		}
		else {
			usage();
			exit(1);
		}
		++cnt;
	}
	printf("\n");

	// -------------------------------------------------------------------------
	//  read data set and query set
	// -------------------------------------------------------------------------
	data = new float[n * d];
	if (read_bin_data(n, d, data_set, data)) exit(1);

	query = new float[qn * d];
	if (read_bin_query(qn, d, query_set, query)) exit(1);

	if (alg > 0) {
		R = new Result[qn * MAXK];
		if (read_ground_truth(qn, truth_set, R)) exit(1);
	}

	// -------------------------------------------------------------------------
	//  methods
	// -------------------------------------------------------------------------
	switch (alg) {
	case 0:
		ground_truth(n, qn, d, data_name, truth_set, output_folder, 
			(const float*) data, (const float*) query);
		break;
	case 1:
		linear_scan(n, qn, d, data_name, "Linear_Scan", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 2:
		random_scan(n, qn, d, conf_name, data_name, "Random_Scan", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 3:
		sorted_scan(n, qn, d, conf_name, data_name, "Sorted_Scan", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 4:
		if (I == 0) {
			embed_hash(n, qn, d, m, l, b, conf_name, data_name, "EH", 
				output_folder, (const float*) data, (const float*) query, 
				(const Result*) R);
		} else {
			orig_embed_hash(n, qn, d, m, l, conf_name, data_name, "Orig_EH", 
				output_folder, (const float*) data, (const float*) query, 
				(const Result*) R);
		}
		break;
	case 5:
		if (I == 0) {
			bilinear_hash(n, qn, d, m, l, b, conf_name, data_name, "BH", 
				output_folder, (const float*) data, (const float*) query, 
				(const Result*) R);
		} else {
			orig_bilinear_hash(n, qn, d, m, l, conf_name, data_name, "Orig_BH", 
				output_folder, (const float*) data, (const float*) query, 
				(const Result*) R);
		}
		break;
	case 6:
		if (I == 0) {
			multilinear_hash(n, qn, d, M, m, l, b, conf_name, data_name, "MH", 
				output_folder, (const float*) data, (const float*) query, 
				(const Result*) R);
		} else {
			orig_multilinear_hash(n, qn, d, M, m, l, conf_name, data_name, 
				"Orig_MH", output_folder, (const float*) data, 
				(const float*) query, (const Result*) R);
		}
		break;
	case 7:
		fh(n, qn, d, m, s, b, conf_name, data_name, "FH", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 8:
		fh_minus(n, qn, d, m, s, conf_name, data_name, "FH_Minus", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 9:
		nh_lccs(n, qn, d, m, w, s, conf_name, data_name, "NH", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 10:
		nh_counting(n, qn, d, m, s, conf_name, data_name, "NH_Counting", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 11:
		fh_wo_sampling(n, qn, d, m, b, conf_name, data_name, "FH_wo_S", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 12:
		fh_minus_wo_sampling(n, qn, d, m, conf_name, data_name, "FH_Minus_wo_S",
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 13:
		nh_lccs_wo_sampling(n, qn, d, m, w, conf_name, data_name, "NH_wo_S", 
			output_folder, (const float*) data, (const float*) query, 
			(const Result*) R);
		break;
	case 14:
		nh_counting_wo_sampling(n, qn, d, m, conf_name, data_name, 
			"NH_Counting_wo_S", output_folder, (const float*) data, 
			(const float*) query, (const Result*) R);
		break;
	default:
		printf("Parameters error!\n");
		usage();
		break;
	}
	// -------------------------------------------------------------------------
	//  release space
	// -------------------------------------------------------------------------
	delete[] data;
	delete[] query; 
	if (alg > 0) delete[] R;

	return 0;
}
