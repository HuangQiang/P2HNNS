#pragma once

#include "baseline.h"
#include "fh.h"
#include "fh_minus.h"
#include "nh.h"

namespace p2h {

// -----------------------------------------------------------------------------
template<class DType>
int linear_scan(                    // Linear Scan
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    fprintf(fp, "%s:\n", method_name);
    float *norm_q = new float[d];
    float dp = -1.0f;

    HEAD(method_name)
    for (int top_k : TOPKs) {
        gettimeofday(&g_start_time, NULL);
        MinK_List* list = new MinK_List(top_k);
        init_global_metric();
        
        for (int i = 0; i < qn; ++i) {
            list->reset();
            get_normalized_query<DType>(d, &query[i*d], norm_q);
            for (int j = 0; j < n; ++j) {
                dp = fabs(calc_ip2<DType>(d, &data[(uint64_t) j*d], norm_q));
                list->insert(dp, j+1);
            }
            update_global_metric(top_k, n, n, &R[i*MAXK], list);
        }
        delete list;
        gettimeofday(&g_end_time, NULL);
        calc_and_write_global_metric(top_k, qn, fp);
    }
    FOOT(fp)

    delete[] norm_q;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int random_scan(                    // Random_Scan
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Random_Scan<DType> *random = new Random_Scan<DType>(n, d, data);
    random->display();
    float memory = random->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, NULL);

    fprintf(fp, "%s:\n", method_name);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, method_name, cand_list)) return 1;

    for (int cand : cand_list) {
        write_params(cand, fname, fp);
        HEAD(method_name)
        for (int top_k : TOPKs) {
            gettimeofday(&g_start_time, NULL);
            MinK_List* list = new MinK_List(top_k);
            init_global_metric();
            
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query<DType>(d, &query[i*d], norm_q);
                int check_k = random->nns(top_k, cand, norm_q, list);
                update_global_metric(top_k, check_k, n, &R[i*MAXK], list);
            }
            delete list;
            gettimeofday(&g_end_time, NULL);
            calc_and_write_global_metric(top_k, qn, fp);
        }
        FOOT(fp)
    }
    delete random;
    delete[] norm_q;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int sorted_scan(                    // Sorted_Scan
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Sorted_Scan<DType> *sorted = new Sorted_Scan<DType>(n, d, data);
    sorted->display();
    float memory = sorted->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, NULL);

    fprintf(fp, "%s:\n", method_name);
    write_index_info(memory, fp);
    
    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, method_name, cand_list)) return 1;

    for (int cand : cand_list) {
        write_params(cand, fname, fp);
        HEAD(method_name)
        for (int top_k : TOPKs) {
            gettimeofday(&g_start_time, NULL);
            MinK_List* list = new MinK_List(top_k);
            init_global_metric();
            
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query<DType>(d, &query[i*d], norm_q);
                int check_k = sorted->nns(top_k, cand, norm_q, list);
                update_global_metric(top_k, check_k, n, &R[i*MAXK], list);
            }
            delete list;
            gettimeofday(&g_end_time, NULL);
            calc_and_write_global_metric(top_k, qn, fp);
        }
        FOOT(fp)
    }
    delete sorted;
    delete[] norm_q;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int eh(                             // Embedding Hyperplane Hashing
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Angular_Hash<DType> *lsh = new Angular_Hash<DType>(n, d, 1, m, l, b, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, NULL);

    fprintf(fp, "%s: m=%d, l=%d, b=%.2f\n", method_name, m, l, b);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "EH", cand_list)) return 1;

    for (int cand : cand_list) {
        write_params(cand, fname, fp);
        HEAD(method_name)
        for (int top_k : TOPKs) {
            gettimeofday(&g_start_time, NULL);
            MinK_List* list = new MinK_List(top_k);
            init_global_metric();
            
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query<DType>(d, &query[i*d], norm_q);
                int check_k = lsh->nns(top_k, cand, norm_q, list);
                update_global_metric(top_k, check_k, n, &R[i*MAXK], list);
            }
            delete list;
            gettimeofday(&g_end_time, NULL);
            calc_and_write_global_metric(top_k, qn, fp);
        }
        FOOT(fp)
    }
    delete lsh;
    delete[] norm_q;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int bh(                             // Bilinear Hyperplane Hashing
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Angular_Hash<DType> *lsh = new Angular_Hash<DType>(n, d, 2, m, l, b, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, NULL);

    fprintf(fp, "%s: m=%d, l=%d, b=%.2f\n", method_name, m, l, b);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "BH", cand_list)) return 1;

    for (int cand : cand_list) {
        write_params(cand, fname, fp);
        HEAD(method_name)
        for (int top_k : TOPKs) {
            gettimeofday(&g_start_time, NULL);
            MinK_List* list = new MinK_List(top_k);
            init_global_metric();
            
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query<DType>(d, &query[i*d], norm_q);
                int check_k = lsh->nns(top_k, cand, norm_q, list);
                update_global_metric(top_k, check_k, n, &R[i*MAXK], list);
            }
            delete list;
            gettimeofday(&g_end_time, NULL);
            calc_and_write_global_metric(top_k, qn, fp);
        }
        FOOT(fp)
    }
    delete lsh;
    delete[] norm_q;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int mh(                             // Multilinear Hyperplane Hashing
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   M,                            // #proj vecotr used for a single hasher
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Angular_Hash<DType> *lsh = new Angular_Hash<DType>(n, d, M, m, l, b, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, NULL);
    
    fprintf(fp, "%s: M=%d, m=%d, l=%d, b=%.2f\n", method_name, M, m, l, b);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "MH", cand_list)) return 1;

    for (int cand : cand_list) {
        write_params(cand, fname, fp);
        HEAD(method_name)
        for (int top_k : TOPKs) {
            gettimeofday(&g_start_time, NULL);
            MinK_List* list = new MinK_List(top_k);
            init_global_metric();
            
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query<DType>(d, &query[i*d], norm_q);
                int check_k = lsh->nns(top_k, cand, norm_q, list);
                update_global_metric(top_k, check_k, n, &R[i*MAXK], list);
            }
            delete list;
            gettimeofday(&g_end_time, NULL);
            calc_and_write_global_metric(top_k, qn, fp);
        }
        FOOT(fp)
    }
    delete lsh;
    delete[] norm_q;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int fh(                             // Furthest Hyperplane Hashing
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    int   s,                            // scale factor of dimension
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    FH<DType> *lsh = new FH<DType>(n, d, m, s, b, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, NULL);
    
    fprintf(fp, "%s: m=%d, s=%d, b=%.2f\n", method_name, m, s, b);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> l_list;    // a list of separation threshold
    std::vector<int> cand_list; // a list of #candidates
    if (get_conf(conf_name, data_name, "FH", l_list, cand_list)) return 1;

    for (int l : l_list) {
        if (l >= m) continue;
        for (int cand : cand_list) {
            write_params(l, cand, fname, fp);
            HEAD(method_name)
            for (int top_k : TOPKs) {
                gettimeofday(&g_start_time, NULL);
                MinK_List* list = new MinK_List(top_k);
                init_global_metric();
            
                for (int i = 0; i < qn; ++i) {
                    list->reset();
                    get_normalized_query<DType>(d, &query[i*d], norm_q);
                    int check_k = lsh->nns(top_k, l, cand, norm_q, list);
                    update_global_metric(top_k, check_k, n, &R[i*MAXK], list);
                }
                delete list;
                gettimeofday(&g_end_time, NULL);
                calc_and_write_global_metric(top_k, qn, fp);
            }
            FOOT(fp)
        }
    }
    delete lsh;
    delete[] norm_q;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int fh_minus(                       // FH without Multi-Partition
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    int   s,                            // scale factor of dimension
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    FH_Minus<DType> *lsh = new FH_Minus<DType>(n, d, m, s, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, NULL);

    fprintf(fp, "%s: m=%d, s=%d\n", method_name, m, s);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> l_list;    // a list of separation threshold
    std::vector<int> cand_list; // a list of #candidates
    if (get_conf(conf_name, data_name, "FH-", l_list, cand_list)) return 1;

    for (int l : l_list) {
        if (l >= m) continue;
        for (int cand : cand_list) {
            write_params(l, cand, fname, fp);
            HEAD(method_name)
            for (int top_k : TOPKs) {
                gettimeofday(&g_start_time, NULL);
                MinK_List* list = new MinK_List(top_k);
                init_global_metric();
            
                for (int i = 0; i < qn; ++i) {
                    list->reset();
                    get_normalized_query<DType>(d, &query[i*d], norm_q);
                    int check_k = lsh->nns(top_k, l, cand, norm_q, list);
                    update_global_metric(top_k, check_k, n, &R[i*MAXK], list);
                }
                delete list;
                gettimeofday(&g_end_time, NULL);
                calc_and_write_global_metric(top_k, qn, fp);
            }
            FOOT(fp)
        }
    }
    delete lsh;
    delete[] norm_q;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int nh(                             // Nearest Hyperplane Hashing
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    int   s,                            // scale factor of dimension
    float w,                            // bucket width
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    NH<DType> *lsh = new NH<DType>(n, d, m, s, w, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, NULL);
    
    fprintf(fp, "%s: m=%d, s=%d, w=%.2f\n", method_name, m, w, s);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "NH", cand_list)) return 1;

    for (int cand : cand_list) {
        write_params(cand, fname, fp);
        HEAD(method_name)
        for (int top_k : TOPKs) {
            gettimeofday(&g_start_time, NULL);
            MinK_List* list = new MinK_List(top_k);
            init_global_metric();
            
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query<DType>(d, &query[i*d], norm_q);
                int check_k = lsh->nns(top_k, cand, norm_q, list);
                update_global_metric(top_k, check_k, n, &R[i*MAXK], list);
            }
            delete list;
            gettimeofday(&g_end_time, NULL);
            calc_and_write_global_metric(top_k, qn, fp);
        }
        FOOT(fp)
    }
    delete lsh;
    delete[] norm_q;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int fh_wo_s(                        // FH without Randomized Sampling
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    FH_wo_S<DType> *lsh = new FH_wo_S<DType>(n, d, m, b, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, NULL);
    
    fprintf(fp, "%s: m=%d, b=%.2f\n", method_name, m, b);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> l_list;    // a list of separation threshold
    std::vector<int> cand_list; // a list of #candidates
    if (get_conf(conf_name, data_name, "FH", l_list, cand_list)) return 1;

    for (int l : l_list) {
        if (l >= m) continue;
        for (int cand : cand_list) {
            write_params(l, cand, fname, fp);
            HEAD(method_name)
            for (int top_k : TOPKs) {
                gettimeofday(&g_start_time, NULL);
                MinK_List* list = new MinK_List(top_k);
                init_global_metric();
            
                for (int i = 0; i < qn; ++i) {
                    list->reset();
                    get_normalized_query<DType>(d, &query[i*d], norm_q);
                    int check_k = lsh->nns(top_k, l, cand, norm_q, list);
                    update_global_metric(top_k, check_k, n, &R[i*MAXK], list);
                }
                delete list;
                gettimeofday(&g_end_time, NULL);
                calc_and_write_global_metric(top_k, qn, fp);
            }
            FOOT(fp)
        }
    }
    delete lsh;
    delete[] norm_q;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int fh_minus_wo_s(                  // FH- without Randomized Sampling
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    FH_Minus_wo_S<DType> *lsh = new FH_Minus_wo_S<DType>(n, d, m, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, NULL);
    
    fprintf(fp, "%s: m=%d\n", method_name, m);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> l_list;    // a list of separation threshold
    std::vector<int> cand_list; // a list of #candidates
    if (get_conf(conf_name, data_name, "FH-", l_list, cand_list)) return 1;

    for (int l : l_list) {
        if (l >= m) continue;
        for (int cand : cand_list) {
            write_params(l, cand, fname, fp);
            HEAD(method_name)
            for (int top_k : TOPKs) {
                gettimeofday(&g_start_time, NULL);
                MinK_List* list = new MinK_List(top_k);
                init_global_metric();
                
                for (int i = 0; i < qn; ++i) {
                    list->reset();
                    get_normalized_query<DType>(d, &query[i*d], norm_q);
                    int check_k = lsh->nns(top_k, l, cand, norm_q, list);
                    update_global_metric(top_k, check_k, n, &R[i*MAXK], list);
                }
                delete list;
                gettimeofday(&g_end_time, NULL);
                calc_and_write_global_metric(top_k, qn, fp);
            }
            FOOT(fp)
        }
    }
    delete lsh;
    delete[] norm_q;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int nh_wo_s(                        // NH without Randomized Sampling
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    float w,                            // bucket width
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    NH_wo_S<DType> *lsh = new NH_wo_S<DType>(n, d, m, w, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, NULL);
    
    fprintf(fp, "%s: m=%d, w=%.2f\n", method_name, m, w);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "NH", cand_list)) return 1;

    for (int cand : cand_list) {
        write_params(cand, fname, fp);
        HEAD(method_name)
        for (int top_k : TOPKs) {
            gettimeofday(&g_start_time, NULL);
            MinK_List* list = new MinK_List(top_k);
            init_global_metric();
            
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query<DType>(d, &query[i*d], norm_q);
                int check_k = lsh->nns(top_k, cand, norm_q, list);
                update_global_metric(top_k, check_k, n, &R[i*MAXK], list);
            }
            delete list;
            gettimeofday(&g_end_time, NULL);
            calc_and_write_global_metric(top_k, qn, fp);
        }
        FOOT(fp)
    }
    delete lsh;
    delete[] norm_q;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int orig_eh(                        // Original Embedding Hyperplane Hashing
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Orig_EH<DType> *lsh = new Orig_EH<DType>(n, d, m, l, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, NULL);
    
    fprintf(fp, "%s: m=%d, l=%d\n", method_name, m, l);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "EH", cand_list)) return 1;

    for (int cand : cand_list) {
        write_params(cand, fname, fp);
        HEAD(method_name)
        for (int top_k : TOPKs) {
            gettimeofday(&g_start_time, NULL);
            MinK_List* list = new MinK_List(top_k);
            init_global_metric();
            
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query<DType>(d, &query[i*d], norm_q);
                int check_k = lsh->nns(top_k, cand, norm_q, list);
                update_global_metric(top_k, check_k, n, &R[i*MAXK], list);
            }
            delete list;
            gettimeofday(&g_end_time, NULL);
            calc_and_write_global_metric(top_k, qn, fp);
        }
        FOOT(fp)
    }
    delete lsh;
    delete[] norm_q;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int orig_bh(                        // Original Bilinear Hyperplane Hashing
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Orig_BH<DType> *lsh = new Orig_BH<DType>(n, d, m, l, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, NULL);
    
    fprintf(fp, "%s: m=%d, l=%d\n", method_name, m, l);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "BH", cand_list)) return 1;

    for (int cand : cand_list) {
        write_params(cand, fname, fp);
        HEAD(method_name)
        for (int top_k : TOPKs) {
            gettimeofday(&g_start_time, NULL);
            MinK_List* list = new MinK_List(top_k);
            init_global_metric();
            
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query<DType>(d, &query[i*d], norm_q);
                int check_k = lsh->nns(top_k, cand, norm_q, list);
                update_global_metric(top_k, check_k, n, &R[i*MAXK], list);
            }
            delete list;
            gettimeofday(&g_end_time, NULL);
            calc_and_write_global_metric(top_k, qn, fp);
        }
        FOOT(fp)
    }
    delete lsh;
    delete[] norm_q;
    return 0;
}


// -----------------------------------------------------------------------------
template<class DType>
int orig_mh(                        // Original Multilinear Hyperplane Hashing
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   M,                            // #proj vecotr used for a single hasher
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const DType *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s_%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Orig_MH<DType> *lsh = new Orig_MH<DType>(n, d, M, m, l, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, NULL);
    
    fprintf(fp, "%s: M=%d, m=%d, l=%d\n", method_name, M, m, l);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "MH", cand_list)) return 1;

    for (int cand : cand_list) {
        write_params(cand, fname, fp);
        HEAD(method_name)
        for (int top_k : TOPKs) {
            gettimeofday(&g_start_time, NULL);
            MinK_List* list = new MinK_List(top_k);
            init_global_metric();
            
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query<DType>(d, &query[i*d], norm_q);
                int check_k = lsh->nns(top_k, cand, norm_q, list);
                update_global_metric(top_k, check_k, n, &R[i*MAXK], list);
            }
            delete list;
            gettimeofday(&g_end_time, NULL);
            calc_and_write_global_metric(top_k, qn, fp);
        }
        FOOT(fp)
    }
    delete lsh;
    delete[] norm_q;
    return 0;
}

} // end namespace p2h
