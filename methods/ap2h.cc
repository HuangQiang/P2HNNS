#include "ap2h.h"

namespace p2h {

// -----------------------------------------------------------------------------
int linear_scan(                    // P2HNNS using Linear_Scan
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    fprintf(fp, "%s:\n", method_name);
    float *norm_q = new float[d];

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
        "Fraction (%%)\n");
    for (int round = 0; round < MAX_ROUND; ++round) {
        gettimeofday(&g_start_time, NULL);
        int top_k = TOPK[round];
        MinK_List* list = new MinK_List(top_k);

        g_ratio     = 0.0f;
        g_recall    = 0.0f;
        g_precision = 0.0f;
        g_fraction  = 0.0f;
        for (int i = 0; i < qn; ++i) {
            list->reset();
            get_normalized_query(d, &query[i*d], norm_q);
            for (int j = 0; j < n; ++j) {
                float dp = fabs(calc_inner_product(d, &data[j*d], norm_q));
                list->insert(dp, j + 1);
            }
            g_recall    += calc_recall(top_k, &R[i*MAXK], list);
            g_precision += top_k * 100.0f / n;
            g_fraction  += n * 100.0f / n;
            g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
        }
        delete list; list = NULL;
        gettimeofday(&g_end_time, NULL);
        g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
            (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

        g_ratio     = g_ratio     / qn;
        g_recall    = g_recall    / qn;
        g_precision = g_precision / qn;
        g_fraction  = g_fraction  / qn;
        g_runtime   = (g_runtime * 1000.0f) / qn;

        printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
            g_ratio, g_runtime, g_recall, g_precision, g_fraction);
        fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
            g_recall, g_precision, g_fraction);
    }
    printf("\n");
    fprintf(fp, "\n");
    fclose(fp);
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int random_scan(                    // P2HNNS using Random_Scan
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Random_Scan *random = new Random_Scan(n, d, data);
    random->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = random->get_memory_usage() / 1048576.0f;

    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: \n", method_name);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, method_name, cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int cand : cand_list) {
        fp = fopen(output_set, "a+");
        fprintf(fp, "cand=%d\n", cand);

        printf("cand=%d\n", cand);
        printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
            "Fraction (%%)\n");
        for (int round = 0; round < MAX_ROUND; ++round) {
            gettimeofday(&g_start_time, NULL);
            int top_k = TOPK[round];
            MinK_List* list = new MinK_List(top_k);

            g_ratio     = 0.0f;
            g_recall    = 0.0f;
            g_precision = 0.0f;
            g_fraction  = 0.0f;
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query(d, &query[i*d], norm_q);
                int   check_k = random->nns(top_k, cand, norm_q, list);
                float recall = 0.0f;
                float precision = 0.0f;
                calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);
    
                g_recall    += recall;
                g_precision += precision;
                g_fraction  += check_k * 100.0f / n;
                g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
            }
            delete list; list = NULL;
            gettimeofday(&g_end_time, NULL);
            g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

            g_ratio     = g_ratio     / qn;
            g_recall    = g_recall    / qn;
            g_precision = g_precision / qn;
            g_fraction  = g_fraction  / qn;
            g_runtime   = (g_runtime * 1000.0f) / qn;

            printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                g_ratio, g_runtime, g_recall, g_precision, g_fraction);
            fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                g_recall, g_precision, g_fraction);
        }
        printf("\n");
        fprintf(fp, "\n");
        fclose(fp);
    }
    delete   random;
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int sorted_scan(                    // P2HNNS using Sorted_Scan
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Sorted_Scan *sorted = new Sorted_Scan(n, d, data);
    sorted->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = sorted->get_memory_usage() / 1048576.0f;

    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: \n", method_name);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, method_name, cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int cand : cand_list) {
        fp = fopen(output_set, "a+");
        fprintf(fp, "cand=%d\n", cand);

        printf("cand=%d\n", cand);
        printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
            "Fraction (%%)\n");
        for (int round = 0; round < MAX_ROUND; ++round) {
            gettimeofday(&g_start_time, NULL);
            int top_k = TOPK[round];
            MinK_List* list = new MinK_List(top_k);

            g_ratio     = 0.0f;
            g_recall    = 0.0f;
            g_precision = 0.0f;
            g_fraction  = 0.0f;
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query(d, &query[i*d], norm_q);
                int   check_k = sorted->nns(top_k, cand, norm_q, list);
                float recall = 0.0f;
                float precision = 0.0f;
                calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);
    
                g_recall    += recall;
                g_precision += precision;
                g_fraction  += check_k * 100.0f / n;
                g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
            }
            delete list; list = NULL;
            gettimeofday(&g_end_time, NULL);
            g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

            g_ratio     = g_ratio     / qn;
            g_recall    = g_recall    / qn;
            g_precision = g_precision / qn;
            g_fraction  = g_fraction  / qn;
            g_runtime   = (g_runtime * 1000.0f) / qn;

            printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                g_ratio, g_runtime, g_recall, g_precision, g_fraction);
            fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                g_recall, g_precision, g_fraction);
        }
        printf("\n");
        fprintf(fp, "\n");
        fclose(fp);
    }
    delete   sorted;
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int embed_hash(                     // P2HNNS using EH
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Angular_Hash *lsh = new Angular_Hash(n, d, 1, m, l, b, data);
    lsh->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = lsh->get_memory_usage() / 1048576.0f;

    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: m=%d, l=%d, b=%.2f\n", method_name, m, l, b);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "EH", cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int cand : cand_list) {
        fp = fopen(output_set, "a+");
        fprintf(fp, "cand=%d\n", cand);

        printf("cand=%d\n", cand);
        printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
            "Fraction (%%)\n");
        for (int round = 0; round < MAX_ROUND; ++round) {
            gettimeofday(&g_start_time, NULL);
            int top_k = TOPK[round];
            MinK_List* list = new MinK_List(top_k);

            g_ratio     = 0.0f;
            g_recall    = 0.0f;
            g_precision = 0.0f;
            g_fraction  = 0.0f;
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query(d, &query[i*d], norm_q);
                int   check_k = lsh->nns(top_k, cand, norm_q, list);
                float recall = 0.0f;
                float precision = 0.0f;
                calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);
    
                g_recall    += recall;
                g_precision += precision;
                g_fraction  += check_k * 100.0f / n;
                g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
            }
            delete list; list = NULL;
            gettimeofday(&g_end_time, NULL);
            g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

            g_ratio     = g_ratio     / qn;
            g_recall    = g_recall    / qn;
            g_precision = g_precision / qn;
            g_fraction  = g_fraction  / qn;
            g_runtime   = (g_runtime * 1000.0f) / qn;

            printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                g_ratio, g_runtime, g_recall, g_precision, g_fraction);
            fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                g_recall, g_precision, g_fraction);
        }
        printf("\n");
        fprintf(fp, "\n");
        fclose(fp);
    }
    delete   lsh;
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int bilinear_hash(                  // P2HNNS using BH
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Angular_Hash *lsh = new Angular_Hash(n, d, 2, m, l, b, data);
    lsh->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = lsh->get_memory_usage() / 1048576.0f;

    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: m=%d, l=%d, b=%.2f\n", method_name, m, l, b);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "BH", cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int cand : cand_list) {
        fp = fopen(output_set, "a+");
        fprintf(fp, "cand=%d\n", cand);

        printf("cand=%d\n", cand);
        printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
            "Fraction (%%)\n");
        for (int round = 0; round < MAX_ROUND; ++round) {
            gettimeofday(&g_start_time, NULL);
            int top_k = TOPK[round];
            MinK_List* list = new MinK_List(top_k);

            g_ratio     = 0.0f;
            g_recall    = 0.0f;
            g_precision = 0.0f;
            g_fraction  = 0.0f;
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query(d, &query[i*d], norm_q);
                int   check_k = lsh->nns(top_k, cand, norm_q, list);
                float recall = 0.0f;
                float precision = 0.0f;
                calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);
    
                g_recall    += recall;
                g_precision += precision;
                g_fraction  += check_k * 100.0f / n;
                g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
            }
            delete list; list = NULL;
            gettimeofday(&g_end_time, NULL);
            g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

            g_ratio     = g_ratio     / qn;
            g_recall    = g_recall    / qn;
            g_precision = g_precision / qn;
            g_fraction  = g_fraction  / qn;
            g_runtime   = (g_runtime * 1000.0f) / qn;

            printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                g_ratio, g_runtime, g_recall, g_precision, g_fraction);
            fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                g_recall, g_precision, g_fraction);
        }
        printf("\n");
        fprintf(fp, "\n");
        fclose(fp);
    }
    delete   lsh;
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int multilinear_hash(               // P2HNNS using MH
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
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Angular_Hash *lsh = new Angular_Hash(n, d, M, m, l, b, data);
    lsh->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = lsh->get_memory_usage() / 1048576.0f;
    
    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: M=%d, m=%d, l=%d, b=%.2f\n", method_name, M, m, l, b);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "MH", cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int cand : cand_list) {
        fp = fopen(output_set, "a+");
        fprintf(fp, "cand=%d\n", cand);

        printf("cand=%d\n", cand);
        printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
            "Fraction (%%)\n");
        for (int round = 0; round < MAX_ROUND; ++round) {
            gettimeofday(&g_start_time, NULL);
            int top_k = TOPK[round];
            MinK_List* list = new MinK_List(top_k);

            g_ratio     = 0.0f;
            g_recall    = 0.0f;
            g_precision = 0.0f;
            g_fraction  = 0.0f;
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query(d, &query[i*d], norm_q);
                int   check_k = lsh->nns(top_k, cand, norm_q, list);
                float recall = 0.0f;
                float precision = 0.0f;
                calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);
    
                g_recall    += recall;
                g_precision += precision;
                g_fraction  += check_k * 100.0f / n;
                g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
            }
            delete list; list = NULL;
            gettimeofday(&g_end_time, NULL);
            g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

            g_ratio     = g_ratio     / qn;
            g_recall    = g_recall    / qn;
            g_precision = g_precision / qn;
            g_fraction  = g_fraction  / qn;
            g_runtime   = (g_runtime * 1000.0f) / qn;

            printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                g_ratio, g_runtime, g_recall, g_precision, g_fraction);
            fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                g_recall, g_precision, g_fraction);
        }
        printf("\n");
        fprintf(fp, "\n");
        fclose(fp);
    }
    delete   lsh;
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int fh(                             // P2HNNS using FH (Furthest_Hash)
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    int   s,                            // scale factor of dimension
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Furthest_Hash *lsh = new Furthest_Hash(n, d, m, s, b, data);
    lsh->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = lsh->get_memory_usage() / 1048576.0f;

    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: m=%d, s=%d, b=%.2f\n", method_name, m, s, b);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> l_list;    // a list of separation threshold
    std::vector<int> cand_list; // a list of #candidates
    if (get_conf(conf_name, data_name, "FH", l_list, cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int l : l_list) {
        if (l >= m) continue;
        for (int cand : cand_list) {
            fp = fopen(output_set, "a+");
            fprintf(fp, "l=%d, cand=%d\n", l, cand);
            
            printf("l=%d, cand=%d\n", l, cand);
            printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
                "Fraction (%%)\n");
            for (int round = 0; round < MAX_ROUND; ++round) {
                gettimeofday(&g_start_time, NULL);
                int top_k = TOPK[round];
                MinK_List* list = new MinK_List(top_k);

                g_ratio     = 0.0f;
                g_recall    = 0.0f;
                g_precision = 0.0f;
                g_fraction  = 0.0f;
                for (int i = 0; i < qn; ++i) {
                    list->reset();
                    get_normalized_query(d, &query[i*d], norm_q);
                    int   check_k = lsh->nns(top_k, l, cand, norm_q, list);
                    float recall = 0.0f;
                    float precision = 0.0f;
                    calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);

                    g_recall    += recall;
                    g_precision += precision;
                    g_fraction  += check_k * 100.0f / n;
                    g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
                }
                delete list; list = NULL;
                gettimeofday(&g_end_time, NULL);
                g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                    (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

                g_ratio     = g_ratio     / qn;
                g_recall    = g_recall    / qn;
                g_precision = g_precision / qn;
                g_fraction  = g_fraction  / qn;
                g_runtime   = (g_runtime * 1000.0f) / qn;

                printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                    g_ratio, g_runtime, g_recall, g_precision, g_fraction);
                fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                    g_recall, g_precision, g_fraction);
            }
            printf("\n");
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    delete   lsh; 
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int fh_minus(                       // P2HNNS using FH_Minus
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    int   s,                            // scale factor of dimension
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    FH_Minus *lsh = new FH_Minus(n, d, m, s, data);
    lsh->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = lsh->get_memory_usage() / 1048576.0f;
    
    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: m=%d, s=%d\n", method_name, m, s);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> l_list;    // a list of separation threshold
    std::vector<int> cand_list; // a list of #candidates
    if (get_conf(conf_name, data_name, "FH-", l_list, cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int l : l_list) {
        if (l >= m) continue;
        for (int cand : cand_list) {
            fp = fopen(output_set, "a+");
            fprintf(fp, "l=%d, cand=%d\n", l, cand);
            
            printf("l=%d, cand=%d\n", l, cand);
            printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
                "Fraction (%%)\n");
            for (int round = 0; round < MAX_ROUND; ++round) {
                gettimeofday(&g_start_time, NULL);
                int top_k = TOPK[round];
                MinK_List* list = new MinK_List(top_k);

                g_ratio     = 0.0f;
                g_recall    = 0.0f;
                g_precision = 0.0f;
                g_fraction  = 0.0f;
                for (int i = 0; i < qn; ++i) {
                    list->reset();
                    get_normalized_query(d, &query[i*d], norm_q);
                    int   check_k = lsh->nns(top_k, l, cand, norm_q, list);
                    float recall = 0.0f;
                    float precision = 0.0f;
                    calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);
        
                    g_recall    += recall;
                    g_precision += precision;
                    g_fraction  += check_k * 100.0f / n;
                    g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
                }
                delete list; list = NULL;
                gettimeofday(&g_end_time, NULL);
                g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                    (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

                g_ratio     = g_ratio     / qn;
                g_recall    = g_recall    / qn;
                g_precision = g_precision / qn;
                g_fraction  = g_fraction  / qn;
                g_runtime   = (g_runtime * 1000.0f) / qn;

                printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                    g_ratio, g_runtime, g_recall, g_precision, g_fraction);
                fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                    g_recall, g_precision, g_fraction);
            }
            printf("\n");
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    delete   lsh;
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int nh_lccs(                        // P2HNNS using NH (Nearest_Hash)
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    float w,                            // bucket width
    float s, 
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    NH_LCCS *lsh = new NH_LCCS(n, d, m, w, s, data);
    lsh->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = lsh->get_memory_usage() / 1048576.0f;

    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: m=%d, w=%.2f, s=%d\n", method_name, m, w, int(s));
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "NH", cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int cand : cand_list) {
        fp = fopen(output_set, "a+");
        fprintf(fp, "cand=%d\n", cand);

        printf("cand=%d\n", cand);
        printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
            "Fraction (%%)\n");
        for (int round = 0; round < MAX_ROUND; ++round) {
            gettimeofday(&g_start_time, NULL);
            int top_k = TOPK[round];
            MinK_List* list = new MinK_List(top_k);

            g_ratio     = 0.0f;
            g_recall    = 0.0f;
            g_precision = 0.0f;
            g_fraction  = 0.0f;
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query(d, &query[i*d], norm_q);
                int   check_k = lsh->nns(top_k, cand, norm_q, list);
                float recall = 0.0f;
                float precision = 0.0f;
                calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);
    
                g_recall    += recall;
                g_precision += precision;
                g_fraction  += check_k * 100.0f / n;
                g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
            }
            delete list; list = NULL;
            gettimeofday(&g_end_time, NULL);
            g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

            g_ratio     = g_ratio     / qn;
            g_recall    = g_recall    / qn;
            g_precision = g_precision / qn;
            g_fraction  = g_fraction  / qn;
            g_runtime   = (g_runtime * 1000.0f) / qn;

            printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                g_ratio, g_runtime, g_recall, g_precision, g_fraction);
            fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                g_recall, g_precision, g_fraction);
        }
        printf("\n");
        fprintf(fp, "\n");
        fclose(fp);
    }
    delete   lsh;
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int nh_counting(                    // P2HNNS using NH_Counting
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    int   s,                            // scale factor of dimension
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    NH_Counting *lsh = new NH_Counting(n, d, m, s, data);
    lsh->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = lsh->get_memory_usage() / 1048576.0f;
    
    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: m=%d, s=%d\n", method_name, m, s);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> l_list;    // a list of separation threshold
    std::vector<int> cand_list; // a list of #candidates
    if (get_conf(conf_name, data_name, "FH-", l_list, cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int l : l_list) {
        if (l >= m) continue;
        for (int cand : cand_list) {
            fp = fopen(output_set, "a+");
            fprintf(fp, "l=%d, cand=%d\n", l, cand);
            
            printf("l=%d, cand=%d\n", l, cand);
            printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
                "Fraction (%%)\n");
            for (int round = 0; round < MAX_ROUND; ++round) {
                gettimeofday(&g_start_time, NULL);
                int top_k = TOPK[round];
                MinK_List* list = new MinK_List(top_k);

                g_ratio     = 0.0f;
                g_recall    = 0.0f;
                g_precision = 0.0f;
                g_fraction  = 0.0f;
                for (int i = 0; i < qn; ++i) {
                    list->reset();
                    get_normalized_query(d, &query[i*d], norm_q);
                    int   check_k = lsh->nns(top_k, l, cand, norm_q, list);
                    float recall = 0.0f;
                    float precision = 0.0f;
                    calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);
        
                    g_recall    += recall;
                    g_precision += precision;
                    g_fraction  += check_k * 100.0f / n;
                    g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
                }
                delete list; list = NULL;
                gettimeofday(&g_end_time, NULL);
                g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                    (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

                g_ratio     = g_ratio     / qn;
                g_recall    = g_recall    / qn;
                g_precision = g_precision / qn;
                g_fraction  = g_fraction  / qn;
                g_runtime   = (g_runtime * 1000.0f) / qn;

                printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                    g_ratio, g_runtime, g_recall, g_precision, g_fraction);
                fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                    g_recall, g_precision, g_fraction);
            }
            printf("\n");
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    delete   lsh;
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int fh_wo_sampling(                 // P2HNNS using FH_wo_S
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    FH_wo_S *lsh = new FH_wo_S(n, d, m, b, data);
    lsh->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = lsh->get_memory_usage() / 1048576.0f;

    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: m=%d, b=%.2f\n", method_name, m, b);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> l_list;    // a list of separation threshold
    std::vector<int> cand_list; // a list of #candidates
    if (get_conf(conf_name, data_name, "FH", l_list, cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int l : l_list) {
        if (l >= m) continue;
        for (int cand : cand_list) {
            fp = fopen(output_set, "a+");
            fprintf(fp, "l=%d, cand=%d\n", l, cand);
            
            printf("l=%d, cand=%d\n", l, cand);
            printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
                "Fraction (%%)\n");
            for (int round = 0; round < MAX_ROUND; ++round) {
                gettimeofday(&g_start_time, NULL);
                int top_k = TOPK[round];
                MinK_List* list = new MinK_List(top_k);

                g_ratio     = 0.0f;
                g_recall    = 0.0f;
                g_precision = 0.0f;
                g_fraction  = 0.0f;
                for (int i = 0; i < qn; ++i) {
                    list->reset();
                    get_normalized_query(d, &query[i*d], norm_q);
                    int   check_k = lsh->nns(top_k, l, cand, norm_q, list);
                    float recall = 0.0f;
                    float precision = 0.0f;
                    calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);

                    g_recall    += recall;
                    g_precision += precision;
                    g_fraction  += check_k * 100.0f / n;
                    g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
                }
                delete list; list = NULL;
                gettimeofday(&g_end_time, NULL);
                g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                    (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

                g_ratio     = g_ratio     / qn;
                g_recall    = g_recall    / qn;
                g_precision = g_precision / qn;
                g_fraction  = g_fraction  / qn;
                g_runtime   = (g_runtime * 1000.0f) / qn;

                printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                    g_ratio, g_runtime, g_recall, g_precision, g_fraction);
                fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                    g_recall, g_precision, g_fraction);
            }
            printf("\n");
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    delete   lsh; 
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int fh_minus_wo_sampling(           // P2HNNS using FH_Minus_wo_S
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    FH_Minus_wo_S *lsh = new FH_Minus_wo_S(n, d, m, data);
    lsh->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = lsh->get_memory_usage() / 1048576.0f;
    
    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: m=%d\n", method_name, m);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> l_list;    // a list of separation threshold
    std::vector<int> cand_list; // a list of #candidates
    if (get_conf(conf_name, data_name, "FH-", l_list, cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int l : l_list) {
        if (l >= m) continue;
        for (int cand : cand_list) {
            fp = fopen(output_set, "a+");
            fprintf(fp, "l=%d, cand=%d\n", l, cand);
            
            printf("l=%d, cand=%d\n", l, cand);
            printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
                "Fraction (%%)\n");
            for (int round = 0; round < MAX_ROUND; ++round) {
                gettimeofday(&g_start_time, NULL);
                int top_k = TOPK[round];
                MinK_List* list = new MinK_List(top_k);

                g_ratio     = 0.0f;
                g_recall    = 0.0f;
                g_precision = 0.0f;
                g_fraction  = 0.0f;
                for (int i = 0; i < qn; ++i) {
                    list->reset();
                    get_normalized_query(d, &query[i*d], norm_q);
                    int   check_k = lsh->nns(top_k, l, cand, norm_q, list);
                    float recall = 0.0f;
                    float precision = 0.0f;
                    calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);
        
                    g_recall    += recall;
                    g_precision += precision;
                    g_fraction  += check_k * 100.0f / n;
                    g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
                }
                delete list; list = NULL;
                gettimeofday(&g_end_time, NULL);
                g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                    (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

                g_ratio     = g_ratio     / qn;
                g_recall    = g_recall    / qn;
                g_precision = g_precision / qn;
                g_fraction  = g_fraction  / qn;
                g_runtime   = (g_runtime * 1000.0f) / qn;

                printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                    g_ratio, g_runtime, g_recall, g_precision, g_fraction);
                fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                    g_recall, g_precision, g_fraction);
            }
            printf("\n");
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    delete   lsh;
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int nh_lccs_wo_sampling(            // P2HNNS using NH_wo_S
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    float w,                            // bucket width
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    NH_LCCS_wo_S *lsh = new NH_LCCS_wo_S(n, d, m, w, data);
    lsh->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = lsh->get_memory_usage() / 1048576.0f;

    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: m=%d, w=%.2f\n", method_name, m, w);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "NH", cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int cand : cand_list) {
        fp = fopen(output_set, "a+");
        fprintf(fp, "cand=%d\n", cand);

        printf("cand=%d\n", cand);
        printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
            "Fraction (%%)\n");
        for (int round = 0; round < MAX_ROUND; ++round) {
            gettimeofday(&g_start_time, NULL);
            int top_k = TOPK[round];
            MinK_List* list = new MinK_List(top_k);

            g_ratio     = 0.0f;
            g_recall    = 0.0f;
            g_precision = 0.0f;
            g_fraction  = 0.0f;
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query(d, &query[i*d], norm_q);
                int   check_k = lsh->nns(top_k, cand, norm_q, list);
                float recall = 0.0f;
                float precision = 0.0f;
                calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);
    
                g_recall    += recall;
                g_precision += precision;
                g_fraction  += check_k * 100.0f / n;
                g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
            }
            delete list; list = NULL;
            gettimeofday(&g_end_time, NULL);
            g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

            g_ratio     = g_ratio     / qn;
            g_recall    = g_recall    / qn;
            g_precision = g_precision / qn;
            g_fraction  = g_fraction  / qn;
            g_runtime   = (g_runtime * 1000.0f) / qn;

            printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                g_ratio, g_runtime, g_recall, g_precision, g_fraction);
            fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                g_recall, g_precision, g_fraction);
        }
        printf("\n");
        fprintf(fp, "\n");
        fclose(fp);
    }
    delete   lsh;
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int nh_counting_wo_sampling(        // P2HNNS using NH_Counting_wo_S
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    NH_Counting_wo_S *lsh = new NH_Counting_wo_S(n, d, m, data);
    lsh->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = lsh->get_memory_usage() / 1048576.0f;
    
    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: m=%d\n", method_name, m);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> l_list;    // a list of separation threshold
    std::vector<int> cand_list; // a list of #candidates
    if (get_conf(conf_name, data_name, "FH-", l_list, cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int l : l_list) {
        if (l >= m) continue;
        for (int cand : cand_list) {
            fp = fopen(output_set, "a+");
            fprintf(fp, "l=%d, cand=%d\n", l, cand);
            
            printf("l=%d, cand=%d\n", l, cand);
            printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
                "Fraction (%%)\n");
            for (int round = 0; round < MAX_ROUND; ++round) {
                gettimeofday(&g_start_time, NULL);
                int top_k = TOPK[round];
                MinK_List* list = new MinK_List(top_k);

                g_ratio     = 0.0f;
                g_recall    = 0.0f;
                g_precision = 0.0f;
                g_fraction  = 0.0f;
                for (int i = 0; i < qn; ++i) {
                    list->reset();
                    get_normalized_query(d, &query[i*d], norm_q);
                    int   check_k = lsh->nns(top_k, l, cand, norm_q, list);
                    float recall = 0.0f;
                    float precision = 0.0f;
                    calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);
        
                    g_recall    += recall;
                    g_precision += precision;
                    g_fraction  += check_k * 100.0f / n;
                    g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
                }
                delete list; list = NULL;
                gettimeofday(&g_end_time, NULL);
                g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                    (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

                g_ratio     = g_ratio     / qn;
                g_recall    = g_recall    / qn;
                g_precision = g_precision / qn;
                g_fraction  = g_fraction  / qn;
                g_runtime   = (g_runtime * 1000.0f) / qn;

                printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                    g_ratio, g_runtime, g_recall, g_precision, g_fraction);
                fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                    g_recall, g_precision, g_fraction);
            }
            printf("\n");
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    delete   lsh;
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int orig_embed_hash(                // P2HNNS using Orig_EH
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Orig_EH *lsh = new Orig_EH(n, d, m, l, data);
    lsh->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = lsh->get_memory_usage() / 1048576.0f;

    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: m=%d, l=%d\n", method_name, m, l);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "EH", cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int cand : cand_list) {
        fp = fopen(output_set, "a+");
        fprintf(fp, "cand=%d\n", cand);

        printf("cand=%d\n", cand);
        printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
            "Fraction (%%)\n");
        for (int round = 0; round < MAX_ROUND; ++round) {
            gettimeofday(&g_start_time, NULL);
            int top_k = TOPK[round];
            MinK_List* list = new MinK_List(top_k);

            g_ratio     = 0.0f;
            g_recall    = 0.0f;
            g_precision = 0.0f;
            g_fraction  = 0.0f;
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query(d, &query[i*d], norm_q);
                int   check_k = lsh->nns(top_k, cand, norm_q, list);
                float recall = 0.0f;
                float precision = 0.0f;
                calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);
    
                g_recall    += recall;
                g_precision += precision;
                g_fraction  += check_k * 100.0f / n;
                g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
            }
            delete list; list = NULL;
            gettimeofday(&g_end_time, NULL);
            g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

            g_ratio     = g_ratio     / qn;
            g_recall    = g_recall    / qn;
            g_precision = g_precision / qn;
            g_fraction  = g_fraction  / qn;
            g_runtime   = (g_runtime * 1000.0f) / qn;

            printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                g_ratio, g_runtime, g_recall, g_precision, g_fraction);
            fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                g_recall, g_precision, g_fraction);
        }
        printf("\n");
        fprintf(fp, "\n");
        fclose(fp);
    }
    delete   lsh;
    delete[] norm_q;

    return 0;
}

// -----------------------------------------------------------------------------
int orig_bilinear_hash(             // P2HNNS using Orig_BH
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Orig_BH *lsh = new Orig_BH(n, d, m, l, data);
    lsh->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = lsh->get_memory_usage() / 1048576.0f;

    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: m=%d, l=%d\n", method_name, m, l);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "BH", cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int cand : cand_list) {
        fp = fopen(output_set, "a+");
        fprintf(fp, "cand=%d\n", cand);

        printf("cand=%d\n", cand);
        printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
            "Fraction (%%)\n");
        for (int round = 0; round < MAX_ROUND; ++round) {
            gettimeofday(&g_start_time, NULL);
            int top_k = TOPK[round];
            MinK_List* list = new MinK_List(top_k);

            g_ratio     = 0.0f;
            g_recall    = 0.0f;
            g_precision = 0.0f;
            g_fraction  = 0.0f;
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query(d, &query[i*d], norm_q);
                int   check_k = lsh->nns(top_k, cand, norm_q, list);
                float recall = 0.0f;
                float precision = 0.0f;
                calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);
    
                g_recall    += recall;
                g_precision += precision;
                g_fraction  += check_k * 100.0f / n;
                g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
            }
            delete list; list = NULL;
            gettimeofday(&g_end_time, NULL);
            g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

            g_ratio     = g_ratio     / qn;
            g_recall    = g_recall    / qn;
            g_precision = g_precision / qn;
            g_fraction  = g_fraction  / qn;
            g_runtime   = (g_runtime * 1000.0f) / qn;

            printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                g_ratio, g_runtime, g_recall, g_precision, g_fraction);
            fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                g_recall, g_precision, g_fraction);
        }
        printf("\n");
        fprintf(fp, "\n");
        fclose(fp);
    }
    delete   lsh;
    delete[] norm_q;

    return 0;
}


// -----------------------------------------------------------------------------
int orig_multilinear_hash(          // P2HNNS using Orig_MH
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   M,                            // #proj vecotr used for a single hasher
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char output_set[200];
    sprintf(output_set, "%s%s_%s.out", output_folder, data_name, method_name);

    FILE *fp = fopen(output_set, "a+");
    if (!fp) { printf("Could not create %s\n", output_set); return 1; }

    // -------------------------------------------------------------------------
    //  preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, NULL);
    Orig_MH *lsh = new Orig_MH(n, d, M, m, l, data);
    lsh->display();

    gettimeofday(&g_end_time, NULL);
    g_indextime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec 
        - g_start_time.tv_usec) / 1000000.0f;
    g_memory = lsh->get_memory_usage() / 1048576.0f;
    
    printf("Indexing Time:    %f Seconds\n", g_indextime);
    printf("Estimated Memory: %f MB\n\n", g_memory);

    fprintf(fp, "%s: M=%d, m=%d, l=%d\n", method_name, M, m, l);
    fprintf(fp, "Indexing Time: %f Seconds\n", g_indextime);
    fprintf(fp, "Estimated Memory: %f MB\n", g_memory);
    fclose(fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    float *norm_q = new float[d];
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, "MH", cand_list)) return 1;

    printf("%s for Point-to-Hyperplane NNS:\n", method_name);
    for (int cand : cand_list) {
        fp = fopen(output_set, "a+");
        fprintf(fp, "cand=%d\n", cand);

        printf("cand=%d\n", cand);
        printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
            "Fraction (%%)\n");
        for (int round = 0; round < MAX_ROUND; ++round) {
            gettimeofday(&g_start_time, NULL);
            int top_k = TOPK[round];
            MinK_List* list = new MinK_List(top_k);

            g_ratio     = 0.0f;
            g_recall    = 0.0f;
            g_precision = 0.0f;
            g_fraction  = 0.0f;
            for (int i = 0; i < qn; ++i) {
                list->reset();
                get_normalized_query(d, &query[i*d], norm_q);
                int   check_k = lsh->nns(top_k, cand, norm_q, list);
                float recall = 0.0f;
                float precision = 0.0f;
                calc_pre_recall(top_k, check_k, &R[i*MAXK], list, recall, precision);
    
                g_recall    += recall;
                g_precision += precision;
                g_fraction  += check_k * 100.0f / n;
                g_ratio     += calc_ratio(top_k, &R[i*MAXK], list);
            }
            delete list; list = NULL;
            gettimeofday(&g_end_time, NULL);
            g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
                (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

            g_ratio     = g_ratio     / qn;
            g_recall    = g_recall    / qn;
            g_precision = g_precision / qn;
            g_fraction  = g_fraction  / qn;
            g_runtime   = (g_runtime * 1000.0f) / qn;

            printf("%3d\t\t%.4f\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
                g_ratio, g_runtime, g_recall, g_precision, g_fraction);
            fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
                g_recall, g_precision, g_fraction);
        }
        printf("\n");
        fprintf(fp, "\n");
        fclose(fp);
    }
    delete   lsh;
    delete[] norm_q;

    return 0;
}

} // end namespace p2h
