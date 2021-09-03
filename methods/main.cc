#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstring>
#include <sys/types.h>

#include "def.h"
#include "util.h"
#include "ap2h.h"

using namespace p2h;

// -----------------------------------------------------------------------------
void usage()                        // display the usage of this package
{
    printf("\n"
        "-------------------------------------------------------------------\n"
        " Usage of the Package for Point-to Hyperplane NNS                  \n"
        "-------------------------------------------------------------------\n"
        " -alg  {integer}  options of algorithms\n"
        " -n    {integer}  number of data objects\n"
        " -qn   {integer}  number of queries\n"
        " -d    {integer}  data dimension\n"
        " -I    {integer}  is normalized or not\n"
        " -m    {integer}  #hash tables (FH, FH-, NH)\n"
        "                  #single hasher of compond hasher (EH, BH, MH)\n"
        " -l    {integer}  #hash tables (EH, BH, MH)\n"
        " -M    {integer}  #proj vector used for a single hasher (MH)\n"
        " -s    {integer}  scale factor of dimension (FH, FH-, NH)\n"
        " -b    {float}    interval ratio (FH)\n"
        " -w    {float}    bucket width (NH)\n"
        " -cf   {string}   name of configuration\n"
        " -dt   {string}   data type\n"
        " -dn   {string}   name of data set\n"
        " -ds   {string}   address of data set\n"
        " -qs   {string}   address of query set\n"
        " -ts   {string}   address of truth set\n"
        " -op   {string}   output path\n"
        "\n"
        "-------------------------------------------------------------------\n"
        " The Options of Algorithms                                         \n"
        "-------------------------------------------------------------------\n"
        " 0  - Ground-Truth & Histogram & Heatmap\n"
        "      -alg 0 -n -qn -d -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 1  - Linear-Scan\n"
        "      -alg 1 -n -qn -d -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 2  - Random-Scan (Random Selection and Scan)\n"
        "      -alg 2 -n -qn -d -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 3  - Sorted-Scan (Sort and Scan)\n"
        "      -alg 3 -n -qn -d -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 4  - EH based on Multi-Partition (or Original EH)\n"
        "      -alg 4 -n -qn -d -I -m -l -b -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 5  - BH based on Multi-Partition (or Original BH)\n"
        "      -alg 5 -n -qn -d -I -m -l -b -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 6  - MH based on Mutli-Partition (or Original MH)\n"
        "      -alg 6 -n -qn -d -I -m -l -M -b -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 7  - FH (Furthest Hyperpalne Hash)\n"
        "      -alg 7 -n -qn -d -m -s -b -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 8  - FH^- (Furthest Hyperpalne Hash without Multi-Partition)\n"
        "      -alg 8 -n -qn -d -m -s -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 9  - NH (Nearest Hyperpalne Hash with LCCS-LSH)\n"
        "      -alg 9 -n -qn -d -m -w -s -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 10 - FH without Randomized Sampling\n"
        "      -alg 10 -n -qn -d -m -b -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 11 - FH^- without Randomized Sampling\n"
        "      -alg 11 -n -qn -d -m -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 12 - NH without Randomized Sampling\n"
        "      -alg 12 -n -qn -d -m -w -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        "-------------------------------------------------------------------\n"
        " Author: Qiang Huang (huangq@comp.nus.edu.sg)                      \n"
        "-------------------------------------------------------------------\n"
        "\n\n\n");
}

// -----------------------------------------------------------------------------
template<class DType>
void interface(                     // interface for calling function
    int   alg,                          // which algorithm?
    int   n,                            // number of data points
    int   qn,                           // number of query hyperplanes
    int   d,                            // dimensionality
    int   I,                            // is normalized or not (0 - No; 1 - Yes)
    int   M,                            // #proj vectors for a single hasher (MH)
    int   m,                            // #tables (FH,NH), #single hasher (EH,BH,MH)
    int   l,                            // #tables (EH,BH,MH)
    int   s,                            // scale factor of dimension (s > 0) (FH,NH)
    float b,                            // interval ratio (0 < b < 1) (FH)
    float w,                            // bucket width (NH)
    const char *conf_name,              // configuration name
    const char *data_name,              // data name
    const char *data_set,               // address of data set
    const char *query_set,              // address of query set
    const char *truth_set,              // address of ground truth file
    const char *path)                   // output path
{
    // read data set, query set (and ground truth results)
    gettimeofday(&g_start_time, NULL);
    DType *data = new DType[(uint64_t) n*d];
    if (read_bin_data<DType>(n, d, data_set, data)) exit(1);

    DType *query = new DType[qn*d];
    if (read_bin_data<DType>(qn, d, query_set, query)) exit(1);

    Result *R = NULL; // ground truth results
    if (alg > 0) {
        R = new Result[qn*MAXK];
        if (read_ground_truth(qn, truth_set, R)) exit(1);
    }
    gettimeofday(&g_end_time, NULL);
    float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Read Data & Query: %f Seconds\n", running_time);

    // methods
    switch (alg) {
    case 0:
        ground_truth<DType>(n, qn, d, truth_set, path, (const DType*) data,
            (const DType*) query);
        break;
    case 1:
        linear_scan<DType>(n, qn, d, "Linear_Scan", path, (const DType*) data,
            (const DType*) query, (const Result*) R);
        break;
    case 2:
        random_scan<DType>(n, qn, d, conf_name, data_name, "Random_Scan", path,
            (const DType*) data, (const DType*) query, (const Result*) R);
        break;
    case 3:
        sorted_scan<DType>(n, qn, d, conf_name, data_name, "Sorted_Scan", path,
            (const DType*) data, (const DType*) query, (const Result*) R);
        break;
    case 4:
        if (I == 0) {
            eh<DType>(n, qn, d, m, l, b, conf_name, data_name, "EH", path,
                (const DType*) data, (const DType*) query, (const Result*) R);
        } else {
            orig_eh<DType>(n, qn, d, m, l, conf_name, data_name, "Orig_EH", path,
                (const DType*) data, (const DType*) query, (const Result*) R);
        }
        break;
    case 5:
        if (I == 0) {
            bh<DType>(n, qn, d, m, l, b, conf_name, data_name, "BH", path,
                (const DType*) data, (const DType*) query, (const Result*) R);
        } else {
            orig_bh<DType>(n, qn, d, m, l, conf_name, data_name, "Orig_BH", path, 
                (const DType*) data, (const DType*) query, (const Result*) R);
        }
        break;
    case 6:
        if (I == 0) {
            mh<DType>(n, qn, d, M, m, l, b, conf_name, data_name, "MH", path,
                (const DType*) data, (const DType*) query, (const Result*) R);
        } else {
            orig_mh<DType>(n, qn, d, M, m, l, conf_name, data_name, "Orig_MH", path,
                (const DType*) data, (const DType*) query, (const Result*) R);
        }
        break;
    case 7:
        fh<DType>(n, qn, d, m, s, b, conf_name, data_name, "FH", path, 
            (const DType*) data, (const DType*) query, (const Result*) R);
        break;
    case 8:
        fh_minus<DType>(n, qn, d, m, s, conf_name, data_name, "FH_Minus", path, 
            (const DType*) data, (const DType*) query, (const Result*) R);
        break;
    case 9:
        nh<DType>(n, qn, d, m, s, w, conf_name, data_name, "NH", path, 
            (const DType*) data, (const DType*) query, (const Result*) R);
        break;
    case 10:
        fh_wo_s<DType>(n, qn, d, m, b, conf_name, data_name, "FH_wo_S", path,
            (const DType*) data, (const DType*) query, (const Result*) R);
        break;
    case 11:
        fh_minus_wo_s<DType>(n, qn, d, m, conf_name, data_name, "FH_Minus_wo_S", 
            path, (const DType*) data, (const DType*) query, (const Result*) R);
        break;
    case 12:
        nh_wo_s<DType>(n, qn, d, m, w, conf_name, data_name, "NH_wo_S", path,
            (const DType*) data, (const DType*) query, (const Result*) R);
        break;
    default:
        printf("Parameters error!\n"); usage();
        break;
    }
    // release space
    delete[] data;
    delete[] query;
    if (alg > 0) delete[] R;
}

// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
    srand(RANDOM_SEED);   // use a fixed random seed
    
    int   cnt = 1;
    int   alg = -1;       // which algorithm?
    int   n   = -1;       // cardinality
    int   qn  = -1;       // query number
    int   d   = -1;       // dimensionality
    int   I   = -1;       // is normalized or not (0 - No; 1 - Yes)
    int   M   = -1;       // #proj vectors for a single hasher (MH)
    int   m   = -1;       // #tables (FH,FH-,NH), #single hasher (EH,BH,MH)
    int   l   = -1;       // #tables (EH,BH,MH)
    int   s   = -1;       // scale factor of dimension (s > 0) (FH,FH-,NH)
    float b   = -1.0f;    // interval ratio (0 < b < 1) (FH)
    float w   = -1.0f;    // bucket width (NH)

    char  conf_name[200]; // name of configuration
    char  data_type[20];  // data type
    char  data_name[200]; // name of dataset
    char  data_set[200];  // address of data set
    char  query_set[200]; // address of query set
    char  truth_set[200]; // address of ground truth file
    char  path[200];      // output path

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
            d = atoi(args[++cnt]); assert(d > 1);
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
        else if (strcmp(args[cnt], "-dt") == 0) {
            strncpy(data_type, args[++cnt], sizeof(data_type));
            printf("data_type = %s\n", data_type);
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
        else if (strcmp(args[cnt], "-op") == 0) {
            strncpy(path, args[++cnt], sizeof(path));
            printf("path      = %s\n", path);
            create_dir(path);
        }
        else {
            usage(); exit(1);
        }
        ++cnt;
    }
    printf("\n");

    // -------------------------------------------------------------------------
    //  methods
    // -------------------------------------------------------------------------
    if (strcmp(data_type, "int32") == 0) {
        interface<int>(alg, n, qn, d, I, M, m, l, s, b, w, conf_name, 
            data_name, data_set, query_set, truth_set, path);
    }
    else if (strcmp(data_type, "float32") == 0) {
        interface<float>(alg, n, qn, d, I, M, m, l, s, b, w, conf_name, 
            data_name, data_set, query_set, truth_set, path);
    }
    else {
        printf("Parameters error!\n"); usage();
    }
    return 0;
}
