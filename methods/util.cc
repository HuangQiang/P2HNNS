#include "util.h"

namespace p2h {
	
timeval g_start_time;				// global parameter: start time
timeval g_end_time;					// global parameter: end time

float g_memory    = -1.0f;			// global parameter: memory usage (MB)
float g_indextime = -1.0f;			// global parameter: indexing time (seconds)

float g_runtime   = -1.0f;			// global parameter: running time (ms)
float g_ratio     = -1.0f;			// global parameter: overall ratio
float g_recall    = -1.0f;			// global parameter: recall (%)
float g_precision = -1.0f;			// global parameter: precision (%)
float g_fraction  = -1.0f;			// global parameter: fraction (%)

// -----------------------------------------------------------------------------
void create_dir(					// create dir if the path not exists
	char *path)							// input path
{
	int len = (int) strlen(path);
	for (int i = 0; i < len; ++i) {
		if (path[i] != '/') continue; 
		
		char ch = path[i + 1];
		path[i + 1] = '\0';
		if (access(path, F_OK) != 0) { // create directory if not exists
			if (mkdir(path, 0755) != 0) {
				printf("Could not create directory %s\n", path);
				exit(1);
			}
		}
		path[i + 1] = ch;
	}
}

// -----------------------------------------------------------------------------
int read_bin_data(					// read data set (binary) from disk
	int   n,							// number of data points
	int   d,							// dimensionality
	const char *fname,					// address of data
	float *data)						// data (return)
{
	gettimeofday(&g_start_time, NULL);
	FILE *fp = fopen(fname, "rb");
	if (!fp) { printf("Could not open %s\n", fname); return 1; }
	
	for (int i = 0; i < n * d; i += d) {
		fread(&data[i], SIZEFLOAT, d-1, fp);
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
int read_bin_query(					// read query set (binary) from disk
	int   qn,							// number of queries
	int   d,							// dimensionality
	const char *fname,					// address of query set
	float *query)						// query (return)
{
	gettimeofday(&g_start_time, NULL);
	FILE *fp = fopen(fname, "rb");
	if (!fp) { printf("Could not open %s\n", fname); return 1; }

	fread(query, SIZEFLOAT, qn * d, fp);
	fclose(fp);
	gettimeofday(&g_end_time, NULL);

	float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
		(g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;	
	printf("Read Query: %f Seconds\n", running_time);

	return 0;
}

// -----------------------------------------------------------------------------
int read_ground_truth(				// read ground truth results from disk
	int qn,								// number of query objects
	const char *fname,					// address of truth set
	Result *R)							// ground truth results (return)
{
	gettimeofday(&g_start_time, NULL);
	FILE *fp = fopen(fname, "r");
	if (!fp) { printf("Could not open %s\n", fname); return 1; }

	int tmp1 = -1;
	int tmp2 = -1;
	fscanf(fp, "%d,%d\n", &tmp1, &tmp2);
	assert(tmp1 == qn && tmp2 == MAXK);

	for (int i = 0; i < qn; ++i) {
		fscanf(fp, "%d", &tmp1);
		for (int j = 0; j < MAXK; ++j) {
			fscanf(fp, ",%d,%f", &R[i*MAXK+j].id_, &R[i*MAXK+j].key_);
		}
		fscanf(fp, "\n");
	}
	fclose(fp);
	gettimeofday(&g_end_time, NULL);

	float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
		(g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
	printf("Read Truth: %f Seconds\n\n", running_time);

	return 0;
}

// -----------------------------------------------------------------------------
void get_csv_from_line(				// get an array with csv format from a line
	std::string str_data,				// a string line
	std::vector<int> &csv_data)			// csv data (return)
{
	csv_data.clear();

	std::istringstream ss(str_data);
	while (ss) {
		std::string s;
	    if (!getline(ss, s, ',')) break;
    	csv_data.push_back(stoi(s));
    }
}

// -----------------------------------------------------------------------------
int get_conf(						// get cand list from configuration file
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	std::vector<int> &cand)				// candidates list (return)
{
	std::ifstream infile(conf_name);
	if (!infile) { printf("Could not open %s\n", conf_name); return 1; }

	std::string dname, mname, tmp;
	bool stop = false;
	while (infile) {
		getline(infile, dname);
		
		// skip the first two methods
		getline(infile, mname);
		getline(infile, tmp);
		getline(infile, tmp);

		getline(infile, mname);
		getline(infile, tmp);
		getline(infile, tmp);

		// check the remaining methods
		while (true) {
			getline(infile, mname);
			if (mname.length() == 0) break;

			if ((dname.compare(data_name)==0) && (mname.compare(method_name)==0)) {
				getline(infile, tmp); get_csv_from_line(tmp, cand);
				stop = true; break;
			}
			else {
				getline(infile, tmp);
			}
		}
		if (stop) break;
	}
	infile.close();
	return 0;
}

// -----------------------------------------------------------------------------
int get_conf(						// get nc and cand list from config file
	const char *conf_name,				// name of configuration
	const char *data_name,				// name of dataset
	const char *method_name,			// name of method
	std::vector<int> &l,				// a list of separation threshold (return)
	std::vector<int> &cand)				// a list of #candidates (return)
{
	std::ifstream infile(conf_name);
	if (!infile) { printf("Could not open %s\n", conf_name); return 1; }

	std::string dname, mname, tmp;
	while (infile) {
		getline(infile, dname);
		
		// check the 1st method
		getline(infile, mname);
		if ((dname.compare(data_name)==0) && (mname.compare(method_name)==0)) {
			getline(infile, tmp); get_csv_from_line(tmp, l);
			getline(infile, tmp); get_csv_from_line(tmp, cand);
			break;
		}
		else {
			getline(infile, tmp);
			getline(infile, tmp);
		}
		
		// check the 2nd method
		getline(infile, mname);
		if ((dname.compare(data_name)==0) && (mname.compare(method_name)==0)) {
			getline(infile, tmp); get_csv_from_line(tmp, l);
			getline(infile, tmp); get_csv_from_line(tmp, cand);
			break;
		}
		else {
			getline(infile, tmp);
			getline(infile, tmp);
		}

		// skip the remaining methods
		while (true) {
			getline(infile, mname);
			if (mname.length() == 0) break;
			else getline(infile, tmp); // skip the setting of this method
		}
		if (dname.length() == 0 && mname.length() == 0) break;
	}
	infile.close();
	return 0;
}

// -----------------------------------------------------------------------------
float uniform(						// r.v. from Uniform(min, max)
	float min,							// min value
	float max)							// max value
{
	int   num  = rand();
	float base = (float) RAND_MAX - 1.0F;
	float frac = ((float) num) / base;

	return (max - min) * frac + min;
}

// -----------------------------------------------------------------------------
//  Given a mean and a standard deviation, gaussian generates a normally 
//  distributed random number.
//
//  Algorithm:  Polar Method, p.104, Knuth, vol. 2
// -----------------------------------------------------------------------------
float gaussian(						// r.v. from Gaussian(mean, sigma)
	float mean,							// mean value
	float sigma)						// std value
{
	float v1 = -1.0f, v2 = -1.0f, s = -1.0f, x = -1.0f;
	do {
		v1 = 2.0F * uniform(0.0F, 1.0F) - 1.0F;
		v2 = 2.0F * uniform(0.0F, 1.0F) - 1.0F;
		s = v1 * v1 + v2 * v2;
	} while (s >= 1.0F);
	x = v1 * sqrt (-2.0F * log (s) / s);

	return x * sigma + mean; 		// x is distributed from N(0, 1)
}

// -----------------------------------------------------------------------------
float calc_dot_plane(				// calc distance from a point to a hyperplane
	int   dim,							// dimension
	const float *point,					// input point
	const float *plane) 				// input hyperplane
{
	float ip = 0.0f, norm = 0.0f;
	for (int i = 0; i < dim; ++i) {
		ip   += point[i] * plane[i];
		norm += plane[i] * plane[i];
	}
	ip += plane[dim];

	return fabs(ip) / sqrt(norm);
}

// -----------------------------------------------------------------------------
float calc_inner_product(			// calc inner product
	int   dim,							// dimension
	const float *p1,					// 1st point
	const float *p2)					// 2nd point
{
	float ret = 0.0f;
	for (int i = 0; i < dim; ++i) ret += p1[i] * p2[i];
	return ret;
}

// -----------------------------------------------------------------------------
float calc_l2_sqr(					// calc L2 square distance
	int   dim,							// dimension
	const float *p1,					// 1st point
	const float *p2)					// 2nd point
{
	float ret = 0.0f;
	for (int i = 0; i < dim; ++i) ret += SQR(p1[i] - p2[i]);
	return ret;
}

// -----------------------------------------------------------------------------
float calc_l2_dist(					// calc L2 distance
	int   dim,							// dimension
	const float *p1,					// 1st point
	const float *p2)					// 2nd point
{
	return sqrt(calc_l2_sqr(dim, p1, p2));
}

// -----------------------------------------------------------------------------
float calc_cosine_angle(			// calc cosine angle, [-1,1]
	int   dim,							// dimension
	const float *p1,					// 1st point
	const float *p2)					// 2nd point
{
	float ip    = calc_inner_product(dim, p1, p2);
	float norm1 = sqrt(calc_inner_product(dim, p1, p1));
	float norm2 = sqrt(calc_inner_product(dim, p2, p2));

	return ip / (norm1 * norm2);
}

// -----------------------------------------------------------------------------
float calc_angle(					// calc angle between two points, [0, PI]
	int   dim,							// dimension
	const float *p1,					// 1st point
	const float *p2)					// 2nd point
{
	return acos(calc_cosine_angle(dim, p1, p2));
}

// -----------------------------------------------------------------------------
float calc_ratio(					// calc overall ratio
	int   k,							// top-k value
	const Result *R,					// ground truth results 
	MinK_List *list)					// results returned by algorithms
{
	// add penalty if list->size() < k
	if (list->size() < k) return sqrt(MAXREAL);

	// consider geometric mean instead of arithmetic mean for overall ratio
	float sum = 0.0f, r = -1.0f;
	for (int j = 0; j < k; ++j) {
		if (fabs(list->ith_key(j) - R[j].key_) < CHECK_ERROR) r = 1.0f;
		else r = sqrt((list->ith_key(j) + 1e-9) / (R[j].key_ + 1e-9));
		sum += log(r);
	}
	return pow(E, sum / k);
}

// -----------------------------------------------------------------------------
float calc_recall(					// calc recall (percentage)
	int   k,							// top-k value
	const Result *R,					// ground truth results 
	MinK_List *list)					// results returned by algorithms
{
	int i = list->size() - 1;
	int last = k - 1;
	while (i >= 0 && list->ith_key(i) - R[last].key_ > FLOATZERO) {
		i--;
	}
	return (i + 1) * 100.0f / k;
}

// -----------------------------------------------------------------------------
void calc_pre_recall(				// calc precision and recall (percentage)
	int   top_k,						// top-k value
	int   check_k,						// number of checked objects
	const Result *R,					// ground truth results 
	MinK_List *list,					// results returned by algorithms
	float &recall,						// recall value (return)
	float &precision)					// precision value (return)
{
	int i = list->size() - 1;
	int last = top_k - 1;
	while (i >= 0 && list->ith_key(i) - R[last].key_ > FLOATZERO) {
		i--;
	}
	recall    = (i + 1) * 100.0f / top_k;
	precision = (i + 1) * 100.0f / check_k;
}

// -----------------------------------------------------------------------------
void get_normalized_query(			// get normalized query
	int   d,							// dimension
	const float *query,					// input query
	float *norm_q)						// normalized query (return)
{
	float norm = sqrt(calc_inner_product(d-1, query, query));
	for (int i = 0; i < d; ++i) norm_q[i] = query[i] / norm;
}

// -----------------------------------------------------------------------------
void norm_distribution(				// analysis of the norm distribution of data
	int   n, 							// number of data objects
	int   d, 							// dimensionality
	const char *data_name,				// name of dataset
	const char *output_folder, 			// output folder
	const float *data) 					// data objects
{
	gettimeofday(&g_start_time, NULL);

	// find l2-norm of data objects, and calc min, max, mean, and std
	std::vector<float> norm(n, 0.0f);
	float max_norm  = MINREAL;
	float min_norm  = MAXREAL;
	float mean_norm = 0.0f;
	for (int i = 0; i < n; ++i) {
		norm[i] = sqrt(calc_inner_product(d-1, &data[i*d], &data[i*d]));
		
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
	char output_set[200];
	sprintf(output_set, "%s%s_%d_L2_Norm_Distr.out", output_folder, data_name, m);

	FILE *fp = fopen(output_set, "w");
	if (!fp) { printf("Could not create %s\n", output_set); exit(1); }

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
void angle_distribution(			// histogram of angle between data and query
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	const char *data_name,				// name of dataset
	const char *output_folder,			// output folder
	const float *data,					// data  objects
	const float *query)					// query objects
{
	gettimeofday(&g_start_time, NULL);

	// calc the angle between data and query, and min, max, mean, std
	uint64_t N = n * qn;
	float *norm_q = new float[d];
	std::vector<std::vector<double> > angle(qn, std::vector<double>(n, 0.0));
	
	double max_angle  = MINREAL;
	double min_angle  = MAXREAL;
	double mean_angle = 0.0;
	for (int i = 0; i < qn; ++i) {
		get_normalized_query(d, &query[i*d], norm_q);
		for (int j = 0; j < n; ++j) {
			double this_angle = fabs(calc_cosine_angle(d, norm_q, &data[j*d]));
			assert(this_angle >= 0 && this_angle <= 1);
			// double this_angle = calc_angle(d, norm_q, &data[j*d]);
			// assert(this_angle >= 0 && this_angle <= PI);

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
	int m = M; 					// split m partitions in [0, 1]
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
	char output_set[200];
	sprintf(output_set, "%s%s_%d_Cos_Angle_Distr.out", output_folder, data_name, m);

	FILE *fp = fopen(output_set, "w");
	if (!fp) { printf("Could not create %s\n", output_set); exit(1); }

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
void heatmap(						// heatmap between l2-norm and |cos(angle)|
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	const char *data_name,				// name of dataset
	const char *output_folder,			// output folder
	const float *data,					// data  objects
	const float *query)					// query objects
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
		norm[i] = sqrt(calc_inner_product(d-1, &data[i*d], &data[i*d]));
		
		mean_norm += norm[i];
		if (norm[i] > max_norm) max_norm = norm[i];
		else if (norm[i] < min_norm) min_norm = norm[i];

		for (int j = 0; j < qn; ++j) {
			// calc mean, min, max of |cos(angle)|
			get_normalized_query(d, &query[j*d], norm_q);

			angle[i][j] = fabs(calc_cosine_angle(d, norm_q, &data[i*d]));
			assert(angle[i][j] >= 0 && angle[i][j] <= 1);

			mean_angle  += angle[i][j];
			if (angle[i][j] > max_angle) max_angle = angle[i][j];
			else if (angle[i][j] < min_angle) min_angle = angle[i][j];
		}
	}
	mean_norm  /= n;
	mean_angle /= N;

	// calc std
	double std_norm  = 0.0;
	double std_angle = 0.0;
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
	int    m = M; 					// split m partitions in [0, 1]
	double interval_norm  = (max_norm  - min_norm)  / m;
	double interval_angle = (max_angle - min_angle) / m;
	printf("n = %d, qn = %d, m = %d, interval_norm = %lf, interval_angle = %lf\n", 
		n, qn, m, interval_norm, interval_angle);

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
	char output_set[200];
	sprintf(output_set, "%s%s_%d_Heatmap.out", output_folder, data_name, m);

	FILE *fp = fopen(output_set, "w");
	if (!fp) { printf("Could not create %s\n", output_set); exit(1); }

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
int ground_truth(					// find ground truth results
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimension of space
	const char *data_name,				// name of dataset
	const char *truth_set,				// address of truth set
	const char *output_folder,			// output folder
	const float *data,					// data  objects
	const float *query)					// query objects
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
		get_normalized_query(d, &query[i*d], norm_q);
		for (int j = 0; j < n; ++j) {
			float dp = fabs(calc_inner_product(d, &data[j*d], norm_q));
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
	norm_distribution(n, d, data_name, output_folder, data);
	angle_distribution(n, qn, d, data_name, output_folder, data, query);
	heatmap(n, qn, d, data_name, output_folder, data, query);

	return 0;
}

} // end namespace p2hnns
