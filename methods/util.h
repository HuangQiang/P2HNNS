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

extern timeval g_start_time;		// global parameter: start time
extern timeval g_end_time;			// global parameter: end time

extern float g_memory;				// global parameter: memory usage (MB)
extern float g_indextime;			// global parameter: indexing time (seconds)

extern float g_runtime;				// global parameter: running time (ms)
extern float g_ratio;				// global parameter: overall ratio
extern float g_recall;				// global parameter: recall (%)
extern float g_precision;			// global parameter: precision (%)
extern float g_fraction;			// global parameter: fraction (%)

// -----------------------------------------------------------------------------
void create_dir(					// create dir if the path exists
	char *path);						// input path

// -----------------------------------------------------------------------------
int read_bin_data(					// read data set (binary) from disk
	int   n,							// number of data points
	int   d,							// dimensionality
	const char *fname,					// address of data set
	float *data);						// data (return)

// -----------------------------------------------------------------------------
int read_bin_query(					// read query set (binary) from disk
	int   qn,							// number of queries
	int   d,							// dimensionality
	const char *fname,					// address of query set
	float *query);						// query (return)

// -----------------------------------------------------------------------------
int read_ground_truth(				// read ground truth results from disk
	int    qn,							// number of query objects
	const  char *fname,					// address of truth set
	Result *R);							// ground truth results (return)

// -----------------------------------------------------------------------------
void get_csv_from_line(				// get an array with csv format from a line
	std::string str_data,				// a string line
	std::vector<int> &csv_data);		// csv data (return)
	
// -----------------------------------------------------------------------------
int get_conf(						// get cand list from configuration file
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	std::vector<int> &cand);			// candidates list (return)

// -----------------------------------------------------------------------------
int get_conf(						// get nc and cand list from config file
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	std::vector<int> &l,				// a list of separation threshold (return)
	std::vector<int> &cand);			// a list of #candidates (return)

// -----------------------------------------------------------------------------
float uniform(						// r.v. from Uniform(min, max)
	float min,							// min value
	float max);							// max value

// -----------------------------------------------------------------------------
float gaussian(						// r.v. from Gaussian(mean, sigma)
	float mean,							// mean value
	float sigma);						// std value

// -----------------------------------------------------------------------------
float calc_dot_plane(				// calc distance from a point to a plane
	int   dim,							// dimension
	const float *point,					// input point
	const float *plane);				// input plane

// -----------------------------------------------------------------------------
float calc_inner_product(			// calc inner product
	int   dim,							// dimension
	const float *p1,					// 1st point
	const float *p2);					// 2nd point
	
// -----------------------------------------------------------------------------
float calc_l2_sqr(					// calc L2 square distance
	int   dim,							// dimension
	const float *p1,					// 1st point
	const float *p2);					// 2nd point

// -----------------------------------------------------------------------------
float calc_l2_dist(					// calc L2 distance between two points
	int   dim,							// dimension
	const float *p1,					// 1st point
	const float *p2);					// 2nd point

// -----------------------------------------------------------------------------
float calc_cosine_angle(			// calc cosine angle, [-1,1]
	int   dim,							// dimension
	const float *p1,					// 1st point
	const float *p2);					// 2nd point

// -----------------------------------------------------------------------------
float calc_angle(					// calc angle between two points, [0, PI]
	int   dim,							// dimension
	const float *p1,					// 1st point
	const float *p2);					// 2nd point

// -----------------------------------------------------------------------------
float calc_ratio(					// calc overall ratio
	int   k,							// top-k value
	const Result *R,					// ground truth results 
	MinK_List *list);					// results returned by algorithms
	
// -----------------------------------------------------------------------------
float calc_recall(					// calc recall (percentage)
	int   k,							// top-k value
	const Result *R,					// ground truth results 
	MinK_List *list);					// results returned by algorithms

// -----------------------------------------------------------------------------
void calc_pre_recall(				// calc precision and recall (percentage)
	int   top_k,						// top-k value
	int   check_k,						// number of checked objects
	const Result *R,					// ground truth results 
	MinK_List *list,					// results returned by algorithms
	float &recall,						// recall value (return)
	float &precision);					// precision value (return)

// -----------------------------------------------------------------------------
void get_normalized_query(			// get normalized query
	int   d,							// dimension
	const float *query,					// input query
	float *norm_q);						// normalized query (return)

// -----------------------------------------------------------------------------
void norm_distribution(				// histogram of l2-norm of data
	int   n, 							// number of data objects
	int   d, 							// dimensionality
	const char *data_name,				// name of dataset
	const char *output_folder, 			// output folder
	const float *data);					// data objects

// -----------------------------------------------------------------------------
void angle_distribution(			// histogram of angle between data and query
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	const char *data_name,				// name of dataset
	const char *output_folder,			// output folder
	const float *data,					// data  objects
	const float *query);				// query objects

// -----------------------------------------------------------------------------
void heatmap(						// heatmap between l2-norm and |cos \theta|
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	const char *data_name,				// name of dataset
	const char *output_folder,			// output folder
	const float *data,					// data  objects
	const float *query);				// query objects

// -----------------------------------------------------------------------------
int ground_truth(					// find ground truth results
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	const char *data_name,				// name of dataset
	const char *truth_set,				// address of truth set
	const char *output_folder,			// output folder
	const float *data,					// data  objects
	const float *query);				// query objects

} // end namespace p2h
