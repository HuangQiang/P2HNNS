#include <cstdint>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>

const float MIN_DIST = 0.00005F;
const float MAX_DIST = 0.1F;

timeval g_start_time;
timeval g_end_time;

// -----------------------------------------------------------------------------
void create_dir(                    // create directory
    char *path)                         // input path
{
    int len = (int) strlen(path);
    for (int i = 0; i < len; ++i) {
        if (path[i] != '/') continue;
        
        char ch = path[i+1]; path[i+1] = '\0';
        if (access(path, F_OK) != 0) {
            if (mkdir(path, 0755) != 0) { 
                printf("Could not create %s\n", path); exit(1);
            }
        }
        path[i+1] = ch;
    }
}

// -----------------------------------------------------------------------------
int read_bin_data_and_normalize(    // read bin data & normalize
    int   n,                            // number of data points
    int   d,                            // data dimension
    const char *fname,                  // address of data set
    float *data)                        // data (return)
{
    gettimeofday(&g_start_time, NULL);

    // read bin data
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Could not open %s\n", fname); return 1; }

    int i = 0;
    while (!feof(fp) && i < n) {
        uint64_t shift = (uint64_t) i*(d+1);
        fread(&data[shift], sizeof(float), d, fp);
        data[shift+d] = 1.0f;
        ++i;
    }
    assert(i == n);
    fclose(fp);

    // calc the min & max coordinates for d dimensions
    float *min_coord = new float[d];
    float *max_coord = new float[d];

    for (int i = 0; i < n; ++i) {
        const float *point = (const float*) &data[(uint64_t)i*(d+1)];
        for (int j = 0; j < d; ++j) {
            if (i == 0 || point[j] < min_coord[j]) min_coord[j] = point[j];
            if (i == 0 || point[j] > max_coord[j]) max_coord[j] = point[j];
        }
    }

    // calc the data center
    float *center = new float[d];
    for (int i = 0; i < d; ++i) center[i] = (min_coord[i]+max_coord[i])/2.0f;

    float norm = -1.0f, val = -1.0f;
    for (int i = 0; i < n; ++i) {
        float *point = &data[(uint64_t)i*(d+1)];
        // shift data by the center & calc the l2-norm to the center
        norm = 0.0f;
        for (int j = 0; j < d; ++j) {
            val = point[j] - center[j];
            point[j] = val; norm += val * val;
        }
        norm = sqrt(norm);
        // normalization
        for (int j = 0; j < d; ++j) point[j] /= norm;
    }
    // release space
    delete[] max_coord;
    delete[] min_coord;
    delete[] center;

    gettimeofday(&g_end_time, NULL);
    float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Read & Normalize Data: %f Seconds\n", running_time);

    return 0;
}

// -----------------------------------------------------------------------------
int read_bin_data(                  // read bin data from disk
    int   n,                            // number of data points
    int   d,                            // data dimension
    const char *fname,                  // address of data set
    float *min_coord,                   // min coordinates (return)
    float *max_coord,                   // max coordinates (return)
    float *data)                        // data (return)
{
    gettimeofday(&g_start_time, NULL);

    // read bin data
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Could not open %s\n", fname); return 1; }

    int i = 0;
    while (!feof(fp) && i < n) {
        uint64_t shift = (uint64_t) i*(d+1);
        fread(&data[shift], sizeof(float), d, fp);
        data[shift+d] = 1.0f;
        ++i;
    }
    assert(i == n);
    fclose(fp);

    // shift data by the position of center
    for (int i = 0; i < n; ++i) {
        const float *point = (const float*) &data[(uint64_t)i*(d+1)];
        for (int j = 0; j < d; ++j) {
            if (i == 0 || point[j] < min_coord[j]) min_coord[j] = point[j];
            if (i == 0 || point[j] > max_coord[j]) max_coord[j] = point[j];
        }
    }

    // calc the data center
    float *center = new float[d];
    for (int i = 0; i < d; ++i) center[i] = (min_coord[i]+max_coord[i])/2.0f;

    // shift the data by the center & find the max l2-norm to the center
    float max_norm = -1.0f, norm = -1.0f, val = -1.0f;
    for (int i = 0; i < n; ++i) {
        float *point = &data[(uint64_t)i*(d+1)];
        // shift the data by the center
        norm = 0.0f;
        for (int j = 0; j < d; ++j) {
            val = point[j] - center[j];
            point[j] = val; norm += val * val;
        }
        norm = sqrt(norm);
        // find the max l2-norm to the center
        if (max_norm < norm) max_norm = norm;
    }

    // max normalization: rescale the data by the max l2-norm
    for (int i = 0; i < n; ++i) {
        float *point = &data[(uint64_t)i*(d+1)];
        for (int j = 0; j < d; ++j) {
            val = point[j] / max_norm;
            point[j] = val;
            if (i == 0 || val < min_coord[j]) min_coord[j] = val;
            if (i == 0 || val > max_coord[j]) max_coord[j] = val;
        }
    }
    for (int j = 0; j < d; ++j) {
        printf("min[%d]=%f, max[%d]=%f\n", j, min_coord[j], j, max_coord[j]);
    }
    delete[] center;

    gettimeofday(&g_end_time, NULL);
    float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Read Data: %f Seconds\n", running_time);
    
    return 0;
}

// -----------------------------------------------------------------------------
float uniform(                      // r.v. from Uniform(min, max)
    float min,                          // min value
    float max)                          // max value
{
    int   num  = rand();
    float base = (float) RAND_MAX - 1.0F;
    float frac = ((float) num) / base;

    return (max - min) * frac + min;
}

// -----------------------------------------------------------------------------
float calc_dot_plane(               // calc dist from a point to a hyperplane
    int   d,                            // dimension of input data object
    const float *data,                  // input data object
    const float *query)                 // input query
{
    float ip = 0.0f, norm = 0.0f;
    for (int i = 0; i < d; ++i) {
        ip   += data[i]  * query[i];
        norm += query[i] * query[i];
    }
    ip += query[d];

    return fabs(ip) / sqrt(norm);
}

// -----------------------------------------------------------------------------
int write_bin_data(                 // write binary data to disk
    int   n,                            // number of data points
    int   d,                            // data dimension + 1
    bool  sign,                         // data or query
    char  *fname,                       // output file name
    const float *data)                  // output data
{
    gettimeofday(&g_start_time, NULL);

    //  write binary data
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s\n", fname); return 1; }
    
    fwrite(data, sizeof(float), (uint64_t) n*d, fp);
    fclose(fp);

    gettimeofday(&g_end_time, NULL);
    float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Write Bin %s: %f Seconds\n\n", sign?"Data":"Query", running_time);
    
    return 0;
}

// -----------------------------------------------------------------------------
int write_txt_data(                 // write text data to disk
    int   n,                            // number of data points
    int   d,                            // data dimension + 1
    bool  sign,                         // data or query
    char  *fname,                       // output file name
    const float *data)                  // output data
{
    gettimeofday(&g_start_time, NULL);

    // write text data
    std::ofstream fp;
    fp.open(fname, std::ios::trunc);
    for (int i = 0; i < n; ++i) {
        const float *point = &data[(uint64_t)i*d];
        fp << i+1;
        for (int j = 0; j < d; ++j) fp << " " << point[j];
        fp << "\n";
    }
    fp.close();

    gettimeofday(&g_end_time, NULL);
    float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Write Text %s: %f Seconds\n\n", sign?"Data":"Query", running_time);

    return 0;
}

// -----------------------------------------------------------------------------
void generate_query(                // generate query
    int   n,                            // number of data points
    int   qn,                           // number of hyperplane queries
    int   d,                            // data dimension + 1
    const char *fname,                  // file name
    const float *data,                  // input data
    float *query)                       // query (return)
{
    gettimeofday(&g_start_time, NULL);
    
    // generate query set
    float ip, val, rnd, dist, min_dist;
    for (int i = 0; i < qn; ++i) {
        float *record = &query[(uint64_t) i*(d+1)];
        do {
            std::cout << fname << " " << i+1 << ": "; 
            ip = 0.0f;
            for (int j = 0; j < d; ++j) {
                val = uniform(-1.0f, 1.0f);
                rnd = uniform(-1.0f, 1.0f);
                record[j] = val; ip += val * rnd;
            }
            record[d] = -ip;
            
            min_dist = -1.0f;
            for (int j = 0; j < n; ++j) {
                dist = calc_dot_plane(d, &data[(uint64_t)j*(d+1)], record);
                if (min_dist < 0 || dist < min_dist) min_dist = dist;
            }
            printf("min_dist = %f, ip = %f\n", min_dist, ip);
        } while (min_dist < MIN_DIST || min_dist > MAX_DIST);
    }
    gettimeofday(&g_end_time, NULL);
    float running_time = g_end_time.tv_sec - g_start_time.tv_sec +
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Generate Query: %f Seconds\n", running_time);
}

// -----------------------------------------------------------------------------
void generate_query(                // generate query
    int   n,                            // number of data points
    int   qn,                           // number of hyperplane queries
    int   d,                            // data dimension + 1
    const char *fname,                  // file name
    const float *min_coord,             // min coordinates in d-dimensions
    const float *max_coord,             // max coordinates in d-dimensions
    const float *data,                  // input data
    float *query)                       // query (return)
{
    gettimeofday(&g_start_time, NULL);
    
    // generate query set
    float ip, val, rnd, dist, min_dist;
    for (int i = 0; i < qn; ++i) {
        float *record = &query[(uint64_t) i*(d+1)];
        do {
            std::cout << fname << " " << i+1 << ": "; 
            ip = 0.0f;
            for (int j = 0; j < d; ++j) {
                val = uniform(min_coord[j], max_coord[j]);
                rnd = uniform(min_coord[j], max_coord[j]);
                record[j] = val; ip += val * rnd;
            }
            record[d] = -ip;

            min_dist = -1.0f;
            for (int j = 0; j < n; ++j) {
                dist = calc_dot_plane(d, &data[(uint64_t)j*(d+1)], record);
                if (min_dist < 0 || dist < min_dist) min_dist = dist;
            }
            printf("min_dist = %f, ip = %f\n", min_dist, ip);
        } while (min_dist < MIN_DIST || min_dist > MAX_DIST);
    }
    gettimeofday(&g_end_time, NULL);
    float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Generate Query: %f Seconds\n", running_time);
}

// -----------------------------------------------------------------------------
int main(int nargs, char** args)
{
    srand(666); // set up random seed 

    // read parameters
    int  n    = atoi(args[1]); // cardinality
    int  d    = atoi(args[2]); // dimensionality
    int  qn   = atoi(args[3]); // number of queries
    bool norm = atoi(args[4]) == 0 ? false : true; // 0: normalized
    char infile[200]; strncpy(infile, args[5], sizeof(infile));
    char folder[200]; strncpy(folder, args[6], sizeof(folder));
    create_dir(folder);

    printf("n           = %d\n", n);
    printf("d           = %d\n", d);
    printf("qn          = %d\n", qn);
    printf("normalized? = %s\n", norm ? "true" : "false");
    printf("input file  = %s\n", infile);
    printf("out folder  = %s\n", folder);
    printf("\n");

    float *min_coord = new float[d];
    float *max_coord = new float[d];
    float *data      = new float[(uint64_t) n*(d+1)];

    // read dataset
    if (norm) {
        read_bin_data_and_normalize(n, d, infile, data);
    } else {
        read_bin_data(n, d, infile, min_coord, max_coord, data);
    }

    // write data to disk
    char bin_fname[200]; sprintf(bin_fname, "%s.ds", folder);
    char txt_fname[200]; sprintf(txt_fname, "%s.dt", folder);
    write_bin_data(n, d+1, true, bin_fname, data);
    write_txt_data(n, d+1, true, txt_fname, data);

    // generate query
    float *query = new float[(uint64_t) qn*(d+1)];
    if (norm) {
        generate_query(n, qn, d, infile, data, query);
    } else {
        generate_query(n, qn, d, infile, min_coord, max_coord, data, query);
    }

    // write query to disk
    sprintf(bin_fname, "%s.q",  folder);
    sprintf(txt_fname, "%s.qt", folder);
    write_bin_data(qn, d+1, false, bin_fname, query);
    write_txt_data(qn, d+1, false, txt_fname, query);

    // release space
    delete[] min_coord;
    delete[] max_coord;
    delete[] data;
    delete[] query;

    return 0;
}
