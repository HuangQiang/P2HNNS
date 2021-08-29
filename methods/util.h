#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>

#include "def.h"
#include "pri_queue.h"

namespace p2h {

extern timeval g_start_time;        // global param: start time
extern timeval g_end_time;          // global param: end time

extern float g_memory;              // global param: memory usage (megabytes)
extern float g_indextime;           // global param: indexing time (seconds)

extern float g_runtime;             // global param: running time (ms)
extern float g_ratio;               // global param: overall ratio
extern float g_recall;              // global param: recall (%)
extern float g_precision;           // global param: precision (%)
extern float g_fraction;            // global param: fraction (%)

// -----------------------------------------------------------------------------
void create_dir(                    // create dir if the path exists
    char *path);                        // input path

// -----------------------------------------------------------------------------
int read_ground_truth(              // read ground truth results from disk
    int    qn,                          // number of query objects
    const  char *fname,                 // address of truth set
    Result *R);                         // ground truth results (return)

// -----------------------------------------------------------------------------
void get_csv_from_line(             // get an array with csv format from a line
    std::string str_data,               // a string line
    std::vector<int> &csv_data);        // csv data (return)
    
// -----------------------------------------------------------------------------
int get_conf(                       // get cand list from configuration file
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    std::vector<int> &cand);            // candidates list (return)

// -----------------------------------------------------------------------------
int get_conf(                       // get nc and cand list from config file
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    std::vector<int> &l,                // a list of separation threshold (return)
    std::vector<int> &cand);            // a list of #candidates (return)

// -----------------------------------------------------------------------------
float uniform(                      // r.v. from Uniform(min, max)
    float min,                          // min value
    float max);                         // max value

// -----------------------------------------------------------------------------
float gaussian(                     // r.v. from Gaussian(mean, sigma)
    float mean,                         // mean value
    float sigma);                       // std value

// -----------------------------------------------------------------------------
template<class DType>
void init_prob_vector(              // init probability vector
    int   dim,                          // data dimension
    const DType *data,                  // input data
    float *prob)                        // probability vector (return)
{
    prob[0] = (float) data[0] * data[0];
    for (int i = 1; i < dim; ++i) {
        prob[i] = prob[i-1] + (float) data[i] * data[i];
    }
}

// -----------------------------------------------------------------------------
int coord_sampling(                 // sampling coordinate based on prob vector
    int   d,                            // dimension
    const float *prob);                 // probability vector

// -----------------------------------------------------------------------------
float calc_ratio(                   // calc overall ratio
    int   k,                            // top-k value
    const Result *R,                    // ground truth results 
    MinK_List *list);                   // results returned by algorithms
    
// -----------------------------------------------------------------------------
float calc_recall(                  // calc recall (percentage)
    int   k,                            // top-k value
    const Result *R,                    // ground truth results 
    MinK_List *list);                   // results returned by algorithms

// -----------------------------------------------------------------------------
void calc_pre_recall(               // calc precision and recall (percentage)
    int   top_k,                        // top-k value
    int   check_k,                      // number of checked objects
    const Result *R,                    // ground truth results 
    MinK_List *list,                    // results returned by algorithms
    float &recall,                      // recall value (return)
    float &precision);                  // precision value (return)

// -----------------------------------------------------------------------------
template<class DType>
int read_bin_data(                  // read data set (binary) from disk
    int   n,                            // number of data points
    int   d,                            // dimensionality
    const char *fname,                  // address of data set
    DType *data)                        // data (return)
{
    gettimeofday(&g_start_time, NULL);
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Could not open %s\n", fname); return 1; }
    
    // TODO need to modify the format of data sets
    for (int i = 0; i < (uint64_t) n*d; i += d) {
        fread(&data[i], sizeof(DType), d-1, fp);
        data[i+d-1] = 1.0f;
    }
    fclose(fp);
    gettimeofday(&g_end_time, NULL);

    float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Read Data:  %f Seconds\n", running_time);

    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int read_bin_query(                 // read query set (binary) from disk
    int   qn,                           // number of queries
    int   d,                            // dimensionality
    const char *fname,                  // address of query set
    DType *query)                       // query (return)
{
    gettimeofday(&g_start_time, NULL);
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Could not open %s\n", fname); return 1; }

    fread(query, sizeof(DType), (uint64_t) qn*d, fp);
    fclose(fp);
    gettimeofday(&g_end_time, NULL);

    float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Read Query: %f Seconds\n", running_time);

    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
float calc_inner_product(           // calc inner product
    int   dim,                          // dimension
    const DType *p1,                    // 1st point
    const DType *p2)                    // 2nd point
{
    float ret = 0.0f;
    for (int i = 0; i < dim; ++i) ret += (float) p1[i] * p2[i];
    return ret;
}

// -----------------------------------------------------------------------------
template<class DType>
float calc_inner_product2(          // calc inner product
    int   dim,                          // dimension
    const DType *p1,                    // 1st point
    const float *p2)                    // 2nd point
{
    float ret = 0.0f;
    for (int i = 0; i < dim; ++i) ret += (float) p1[i] * p2[i];
    return ret;
}

// -----------------------------------------------------------------------------
template<class DType>
float calc_cosine_angle(            // calc cosine angle, [-1,1]
    int   dim,                          // dimension
    const DType *p1,                    // 1st point
    const float *p2)                    // 2nd point
{
    float ip    = calc_inner_product2<DType>(dim, p1, p2);
    float norm1 = sqrt(calc_inner_product<DType>(dim, p1, p1));
    float norm2 = sqrt(calc_inner_product<float>(dim, p2, p2));

    return ip / (norm1 * norm2);
}

// -----------------------------------------------------------------------------
template<class DType>
void get_normalized_query(          // get normalized query
    int   d,                            // dimension
    const DType *query,                 // input query
    float *norm_q)                      // normalized query (return)
{
    float norm = sqrt(calc_inner_product<DType>(d-1, query, query));
    for (int i = 0; i < d; ++i) norm_q[i] = query[i] / norm;
}

// -----------------------------------------------------------------------------
template<class DType>
void norm_distribution(             // histogram of l2-norm of data
    int   n,                            // number of data objects
    int   d,                            // dimensionality
    const char *path,                   // output path
    const DType *data)                  // data objects
{
    gettimeofday(&g_start_time, NULL);

    // find l2-norm of data objects, and calc min, max, mean, and std
    std::vector<float> norm(n, 0.0f);
    float max_norm  = MINREAL;
    float min_norm  = MAXREAL;
    float mean_norm = 0.0f;
    for (int i = 0; i < n; ++i) {
        norm[i] = sqrt(calc_inner_product<DType>(d-1, &data[i*d], &data[i*d]));
        
        mean_norm += norm[i];
        if (norm[i] > max_norm) max_norm = norm[i];
        else if (norm[i] < min_norm) min_norm = norm[i];
    }
    mean_norm /= n;

    float std = 0.0f;
    for (int i = 0; i < n; ++i) {
        std += SQR(mean_norm - norm[i]);
    }
    std = sqrt(std / n);
    printf("min = %f, max = %f, mean = %f, std = %f\n", min_norm, max_norm, 
        mean_norm, std);

    // get the percentage of frequency of norm
    int m = M;
    float interval = (max_norm - min_norm) / m;
    printf("n = %d, m = %d, interval = %f\n", n, m, interval);

    std::vector<int> freq(m, 0);
    for (int i = 0; i < n; ++i) {
        int id = (int) ceil((norm[i] - min_norm) / interval) - 1;
        if (id < 0)  id = 0;
        if (id >= m) id = m - 1;
        ++freq[id];
    }

    // write norm distribution
    char fname[200]; sprintf(fname, "%s_%d_L2_Norm_Distr.out", path, m);
    FILE *fp = fopen(fname, "w");
    if (!fp) { printf("Could not create %s\n", fname); exit(1); }

    float num  = 0.5f / m;
    float step = 1.0f / m;
    for (int i = 0; i < m; ++i) {
        fprintf(fp, "%.2f\t%f\n", (num+step*i)*100.0f, freq[i]*100.0f/n);
    }
    fprintf(fp, "\n");
    fclose(fp);

    gettimeofday(&g_end_time, NULL);
    float runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("l2-norm distribution: %.6f Seconds\n\n", runtime);
}

// -----------------------------------------------------------------------------
template<class DType>
void angle_distribution(            // histogram of angle between data and query
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    const char *path,                   // output path
    const DType *data,                  // data  objects
    const DType *query)                 // query objects
{
    gettimeofday(&g_start_time, NULL);

    // calc the angle between data and query, and min, max, mean, std
    uint64_t N = n * qn;
    float *norm_q = new float[d];
    std::vector<std::vector<double> > angle(qn, std::vector<double>(n, 0.0));
    
    double max_angle  = MINREAL;
    double min_angle  = MAXREAL;
    double this_angle, mean_angle = 0.0;
    for (int i = 0; i < qn; ++i) {
        get_normalized_query<DType>(d, &query[i*d], norm_q);
        for (int j = 0; j < n; ++j) {
            this_angle = fabs(calc_cosine_angle<DType>(d, &data[j*d], norm_q));
            assert(this_angle >= 0 && this_angle <= 1);

            angle[i][j]  = this_angle;
            mean_angle  += this_angle;
            if (this_angle > max_angle) max_angle = this_angle;
            else if (this_angle < min_angle) min_angle = this_angle;
        }
    }
    mean_angle = mean_angle / N;

    double std = 0.0;
    for (int i = 0; i < qn; ++i) {
        for (int j = 0; j < n; ++j) {
            std += SQR(mean_angle - angle[i][j]);
        }
    }
    std = sqrt(std / N);
    printf("min = %lf, max = %lf, mean = %lf, std = %lf\n", min_angle,
        max_angle, mean_angle, std);

    // get the percentage of frequency of angle
    int m = M; // split m partitions in [0, 1]
    double interval = (max_angle - min_angle) / m;
    printf("n = %d, qn = %d, m = %d, interval = %lf\n", n, qn, m, interval);

    std::vector<uint64_t> freq(m, 0);
    for (int i = 0; i < qn; ++i) {
        for (int j = 0; j < n; ++j) {
            int id = (int) ceil((angle[i][j] - min_angle) / interval) - 1;
            if (id < 0)  id = 0;
            if (id >= m) id = m - 1;
            ++freq[id];
        }
    }

    // write angle distribution
    char fname[200]; sprintf(fname, "%s_%d_Cos_Angle_Distr.out", path, m);
    FILE *fp = fopen(fname, "w");
    if (!fp) { printf("Could not create %s\n", fname); exit(1); }

    double num  = 0.5 / m;
    double step = 1.0 / m;
    for (int i = 0; i < m; ++i) {
        fprintf(fp, "%.2lf\t%lf\n", (num+step*i)*100.0, freq[i]*100.0/N);
    }
    fprintf(fp, "\n");
    fclose(fp);
    delete[] norm_q;

    gettimeofday(&g_end_time, NULL);
    float runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("|cos(angle)| distribution: %.6f Seconds\n\n", runtime);
}

// -----------------------------------------------------------------------------
template<class DType>
void heatmap(                       // heatmap between l2-norm and |cos \theta|
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    const char *path,                   // output path
    const DType *data,                  // data  objects
    const DType *query)                 // query objects
{
    gettimeofday(&g_start_time, NULL);

    // calc the angle between data and query, and min, max, mean, std
    uint64_t N = n * qn;
    float *norm_q = new float[d];
    std::vector<double> norm(n, 0.0);
    std::vector<std::vector<double> > angle(n, std::vector<double>(qn, 0.0));

    double max_norm   = MINREAL;
    double min_norm   = MAXREAL;
    double mean_norm  = 0.0;
    double max_angle  = MINREAL;
    double min_angle  = MAXREAL;
    double mean_angle = 0.0;
    
    for (int i = 0; i < n; ++i) {
        // calc mean, min, max of l2-norm
        norm[i] = sqrt(calc_inner_product<DType>(d-1, &data[i*d], &data[i*d]));
        
        mean_norm += norm[i];
        if (norm[i] > max_norm) max_norm = norm[i];
        else if (norm[i] < min_norm) min_norm = norm[i];

        for (int j = 0; j < qn; ++j) {
            // calc mean, min, max of |cos(angle)|
            get_normalized_query<DType>(d, &query[j*d], norm_q);

            angle[i][j] = fabs(calc_cosine_angle<DType>(d, &data[i*d], norm_q));
            assert(angle[i][j] >= 0 && angle[i][j] <= 1);

            mean_angle  += angle[i][j];
            if (angle[i][j] > max_angle) max_angle = angle[i][j];
            else if (angle[i][j] < min_angle) min_angle = angle[i][j];
        }
    }
    mean_norm /= n; mean_angle /= N;

    // calc std
    double std_norm  = 0.0, std_angle = 0.0;
    for (int i = 0; i < n; ++i) {
        std_norm += SQR(mean_norm - norm[i]);
        for (int j = 0; j < qn; ++j) {
            std_angle += SQR(mean_angle - angle[i][j]);
        }
    }
    std_norm  = sqrt(std_norm  / n);
    std_angle = sqrt(std_angle / N);
    printf("l2-norm:      min = %lf, max = %lf, mean = %lf, std = %lf\n", 
        min_norm, max_norm, mean_norm, std_norm);
    printf("|cos(angle)|: min = %lf, max = %lf, mean = %lf, std = %lf\n", 
        min_angle, max_angle, mean_angle, std_angle);

    // get the percentage of frequency of angle
    int    m = M; // split m partitions in [0, 1]
    double interval_norm  = (max_norm  - min_norm)  / m;
    double interval_angle = (max_angle - min_angle) / m;
    printf("n = %d, qn = %d, m = %d, interval_norm = %lf, "
        "interval_angle = %lf\n", n, qn, m, interval_norm, interval_angle);

    std::vector<std::vector<uint64_t> > freq(m, std::vector<uint64_t>(m, 0));
    for (int i = 0; i < n; ++i) {
        int x = (int) ceil((norm[i]-min_norm)/interval_norm) - 1;
        if (x < 0)  x = 0;
        if (x >= m) x = m - 1;
        for (int j = 0; j < qn; ++j) {
            int y = (int) ceil((angle[i][j]-min_angle)/interval_angle) - 1;
            if (y < 0)  y = 0;
            if (y >= m) y = m - 1;
            ++freq[y][x];
        }
    }
    // write heatmap
    char fname[200]; sprintf(fname, "%s_%d_Heatmap.out", path, m);
    FILE *fp = fopen(fname, "w");
    if (!fp) { printf("Could not create %s\n", fname); exit(1); }

    // x-axis is Euclidean norm (j); y-axis is |cos \theta| (i)
    for (int i = 0; i < m; ++i) {
        fprintf(fp, "%lf", freq[i][0]*100.0/N);
        for (int j = 1; j < m; ++j) {
            fprintf(fp, ",%lf", freq[i][j]*100.0/N);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    delete[] norm_q;

    gettimeofday(&g_end_time, NULL);
    float runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Heatmap: %.6f Seconds\n\n", runtime);
}

// -----------------------------------------------------------------------------
template<class DType>
int ground_truth(                   // find ground truth results
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    const char *truth_set,              // address of truth set
    const char *path,                   // output path
    const DType *data,                  // data  objects
    const DType *query)                 // query objects
{
    gettimeofday(&g_start_time, NULL);
    FILE *fp = fopen(truth_set, "w");
    if (!fp) { printf("Could not create %s\n", truth_set); return 1; }

    float *norm_q = new float[d];
    MinK_List *list = new MinK_List(MAXK);

    fprintf(fp, "%d,%d\n", qn, MAXK);
    for (int i = 0; i < qn; ++i) {
        // find grouth-truth results by linear scan
        list->reset();
        get_normalized_query<DType>(d, &query[i*d], norm_q);
        for (int j = 0; j < n; ++j) {
            float dp = fabs(calc_inner_product2<DType>(d, &data[j*d], norm_q));
            list->insert(dp, j + 1);
        }
        // write groound truth results for each query
        fprintf(fp, "%d", i + 1);
        for (int j = 0; j < MAXK; ++j) {
            fprintf(fp, ",%d,%.7f", list->ith_id(j), list->ith_key(j));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    delete list;
    delete[] norm_q;
    
    gettimeofday(&g_end_time, NULL);
    float truth_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Ground Truth: %f Seconds\n\n", truth_time);

    // write the histogram and heatmap between l2-norm and cos(angle)
    norm_distribution<DType>(n, d, path, data);
    angle_distribution<DType>(n, qn, d, path, data, query);
    heatmap<DType>(n, qn, d, path, data, query);

    return 0;
}

} // end namespace p2h
