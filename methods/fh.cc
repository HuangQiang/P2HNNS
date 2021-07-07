#include "fh.h"

namespace p2h {

// -----------------------------------------------------------------------------
Furthest_Hash::Furthest_Hash(       // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // number of hash tables
    int   s,                            // scale factor of dimension
    float b,                            // interval ratio
    const float *data)                  // input data
    : n_pts_(n), dim_(d), scale_(s), b_(b), data_(data)
{
    sample_dim_ = d * s;
    fh_dim_     = d * (d + 1) / 2 + 1;
    M_          = MINREAL;

    // -------------------------------------------------------------------------
    //  calc centroid, l2-norm, and max l2-norm
    // -------------------------------------------------------------------------
    int    fh_dim_1  = fh_dim_ - 1;
    float  *norm     = new float[n]; // l2-norm
    float  *centroid = new float[fh_dim_];
    int    *ctrd_cnt = new int[fh_dim_1];

    float  *prob     = new float[d];
    bool   *checked  = new bool[fh_dim_];
    int    *sample_d = new int[n];
    Result *sample_data = new Result[n * sample_dim_];

    memset(centroid, 0.0f, fh_dim_*sizeof(float));
    memset(ctrd_cnt, 0,    fh_dim_1*sizeof(int));

    for (int i = 0; i < n; ++i) {
        transform_data(&data[i*d], prob, checked, norm[i], sample_d[i], 
            &sample_data[i*sample_dim_], centroid, ctrd_cnt);
        if (M_ < norm[i]) M_ = norm[i];
    }
    for (int i = 0; i < fh_dim_1; ++i) centroid[i] /= ctrd_cnt[i];

    for (int i = 0; i < n; ++i) {
        norm[i] = sqrt(M_ - norm[i]);
        centroid[fh_dim_1] += norm[i];
    }
    centroid[fh_dim_1] /= n;

    // -------------------------------------------------------------------------
    //  determine shift_id after shifting data objects to centroid
    // -------------------------------------------------------------------------
    Result *arr = new Result[n];
    for (int i = 0; i < n; ++i) {
        float dist = calc_transform_dist(sample_d[i], 
            &sample_data[i*sample_dim_], centroid);

        arr[i].id_  = i;
        arr[i].key_ = sqrt(dist + SQR(norm[i] - centroid[fh_dim_1]));
    }
    qsort(arr, n, sizeof(Result), ResultCompDesc);

    shift_id_ = new int[n];
    for (int i = 0; i < n; ++i) shift_id_[i] = arr[i].id_;

    // -------------------------------------------------------------------------
    //  divide datasets into blocks and build hash tables for each block
    // -------------------------------------------------------------------------
    int start = 0;
    while (start < n) {
        // partition block
        float min_radius  = b_ * arr[start].key_;
        int   block_index = start, cnt = 0;
        while (block_index < n && arr[block_index].key_ > min_radius) {
            ++block_index;
            if (++cnt >= MAX_BLOCK_NUM) break;
        }

        // add block
        const int *index = (const int*) shift_id_ + start;
        RQALSH *hash = new RQALSH(cnt, fh_dim_, m, index);

        for (int i = 0; i < cnt; ++i) {
            // calc hash value
            int idx = index[i];
            for (int j = 0; j < m; ++j) {
                float val = hash->calc_hash_value(sample_d[idx], j, 
                    &sample_data[idx * sample_dim_]);
                val += hash->a_[j*fh_dim_+fh_dim_1] * norm[idx];

                hash->tables_[j*cnt+i].id_  = i;
                hash->tables_[j*cnt+i].key_ = val;
            }
        }
        // sort hash tables in ascending order by hash values
        for (int i = 0; i < m; ++i) {
            qsort(&hash->tables_[i*cnt], cnt, sizeof(Result), ResultComp);
        }
        hash_.push_back(hash);
        start += cnt;
    }
    assert(start == n);

    // -------------------------------------------------------------------------
    //  release space
    // -------------------------------------------------------------------------
    delete[] arr;
    delete[] sample_data;
    delete[] sample_d;
    delete[] checked;
    delete[] prob;
    delete[] ctrd_cnt;
    delete[] centroid;
    delete[] norm;
}

// -----------------------------------------------------------------------------
void Furthest_Hash::transform_data( // data transformation
    const  float *data,                 // input data
    float  *prob,                       // probability vector
    bool   *checked,                    // is checked?
    float  &norm,                       // norm of fh_data (return)
    int    &sample_d,                   // sample dimension (return)
    Result *sample_data,                // sample data (return)
    float  *centroid,                   // centroid (return)
    int    *ctrd_cnt)                   // centroid coordinate conuter (return)
{
    // calc probability vector and the l2-norm-square of data
    float norm2 = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        norm2 += data[i] * data[i];
        prob[i] = norm2;
    }
    for (int i = 0; i < dim_; ++i) prob[i] /= norm2;

    // randomly sample coordinate of data as the coordinate of sample_data
    int   tmp_idx, tmp_idy, idx, idy;
    float tmp_key;

    norm = 0.0f; sample_d = 0;
    memset(checked, false, fh_dim_*sizeof(bool));
    
    // first select the largest coordinate
    tmp_idx = dim_-1;
    tmp_key = data[tmp_idx] * data[tmp_idx];

    checked[tmp_idx] = true;
    sample_data[sample_d].id_  = tmp_idx;
    sample_data[sample_d].key_ = tmp_key;
    centroid[tmp_idx] += tmp_key; ++ctrd_cnt[tmp_idx];
    norm += tmp_key * tmp_key;
    ++sample_d;

    // consider the combination of the left coordinates
    for (int i = 1; i < sample_dim_; ++i) {
        tmp_idx = sampling(dim_-1, prob);
        tmp_idy = sampling(dim_, prob);
        idx = std::min(tmp_idx, tmp_idy); idy = std::max(tmp_idx, tmp_idy);

        if (idx == idy) {
            tmp_idx = idx;
            if (!checked[tmp_idx]) {
                tmp_key = data[idx] * data[idx];
                
                checked[tmp_idx] = true;
                sample_data[sample_d].id_  = tmp_idx;
                sample_data[sample_d].key_ = tmp_key;
                centroid[tmp_idx] += tmp_key; ++ctrd_cnt[tmp_idx];
                norm += tmp_key * tmp_key; 
                ++sample_d; 
            }
        }
        else {
            tmp_idx = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
            if (!checked[tmp_idx]) {
                tmp_key = data[idx] * data[idy];

                checked[tmp_idx] = true;
                sample_data[sample_d].id_  = tmp_idx;
                sample_data[sample_d].key_ = tmp_key;
                centroid[tmp_idx] += tmp_key; ++ctrd_cnt[tmp_idx];
                norm += tmp_key * tmp_key; 
                ++sample_d;
            }
        }
    }
}

// -----------------------------------------------------------------------------
int Furthest_Hash::sampling(        // sampling coordinate based on prob
    int   d,                            // dimension
    const float *prob)                  // input probability
{
    float end = prob[d-1];
    float rnd = uniform(0.0f, end);
    return std::lower_bound(prob, prob + d, rnd) - prob;
}

// -----------------------------------------------------------------------------
float Furthest_Hash::calc_transform_dist(// calc l2-dist-sqr after transform
    int   sample_d,                     // dimension of sample data
    const Result *sample_data,          // sample data
    const float *centroid)              // centroid after data transformation
{
    float dist = 0.0f;
    for (int i = 0; i < sample_d; ++i) {
        int idx = sample_data[i].id_;
        dist += SQR(sample_data[i].key_ - centroid[idx]);
    }
    return dist;
}

// -----------------------------------------------------------------------------
Furthest_Hash::~Furthest_Hash()     // destructor
{
    if (!hash_.empty()) {
        for (auto hash : hash_) { delete hash; hash = NULL; }
        hash_.clear(); hash_.shrink_to_fit();
    }
    delete[] shift_id_; shift_id_ = NULL;
}

// -----------------------------------------------------------------------------
void Furthest_Hash::display()       // display parameters
{
    printf("Parameters of FH:\n");
    printf("    n            = %d\n",   n_pts_);
    printf("    dim          = %d\n",   dim_);
    printf("    scale factor = %d\n",   scale_);
    printf("    sample_dim   = %d\n",   sample_dim_);
    printf("    fh_dim       = %d\n",   fh_dim_);
    printf("    b            = %.2f\n", b_);
    printf("    M            = %f\n",   sqrt(M_));
    printf("    #blocks      = %d\n\n", (int) hash_.size());
}

// -----------------------------------------------------------------------------
int Furthest_Hash::nns(             // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   l,                            // separation threshold
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    // -------------------------------------------------------------------------
    //  query transformation
    // -------------------------------------------------------------------------
    float  norm_sample_q = 0.0f;
    int    sample_d = -1;
    Result *sample_query = new Result[sample_dim_];
    transform_query(query, norm_sample_q, sample_d, sample_query);
    
    // -------------------------------------------------------------------------
    //  point-to-hyperplane NNS
    // -------------------------------------------------------------------------
    int   verif_cnt = 0;
    int   n_cand    = cand + top_k - 1;
    float fix_val   = norm_sample_q + M_;
    std::vector<int> cand_list;
    
    for (auto hash : hash_) {
        // ---------------------------------------------------------------------
        //  check candidates returned by rqalsh
        // ---------------------------------------------------------------------
        float kfn_dist = -1.0f;
        if (list->isFull()) {
            float kdist = list->max_key();
            kfn_dist = sqrt(fix_val - 2 * kdist * kdist);
        }
        // std::min(block_cand_, n_cand)
        int size = hash->fns(l, n_cand, kfn_dist, sample_d, 
            (const Result*) sample_query, cand_list);

        for (int j = 0; j < size; ++j) {
            int   idx  = cand_list[j];
            float dist = fabs(calc_inner_product(dim_, &data_[idx*dim_], query));
            list->insert(dist, idx + 1);
        }

        // ---------------------------------------------------------------------
        //  update info
        // ---------------------------------------------------------------------
        verif_cnt += size; n_cand -= size; 
        if (n_cand <= 0) break;
    }
    delete[] sample_query;

    return verif_cnt;
}

// -----------------------------------------------------------------------------
void Furthest_Hash::transform_query(// query transformation
    const  float *query,                // input query
    float  &norm_sample_q,              // l2-norm sqr of q after transform (return)
    int    &sample_d,                   // dimension of sample query (return)
    Result *sample_query)               // sample query after transform (return)
{
    // calc probability vector
    float norm2 = 0.0f;
    float *prob = new float[dim_];
    for (int i = 0; i < dim_; ++i) {
        norm2 += query[i] * query[i];
        prob[i] = norm2;
    }
    for (int i = 0; i < dim_; ++i) prob[i] /= norm2;

    // randomly sample coordinate of query as the coordinate of sample_query
    int   tmp_idx, tmp_idy, idx, idy;
    float tmp_key;
    bool  *checked = new bool[fh_dim_];
    memset(checked, false, fh_dim_*sizeof(bool));

    norm_sample_q = 0.0f;
    sample_d = 0;
    for (int i = 0; i < sample_dim_; ++i) {
        tmp_idx = sampling(dim_, prob);
        tmp_idy = sampling(dim_, prob);
        idx = std::min(tmp_idx, tmp_idy); idy = std::max(tmp_idx, tmp_idy);

        if (idx == idy) {
            tmp_idx = idx;
            if (!checked[tmp_idx]) {
                tmp_key = query[idx] * query[idx];

                checked[tmp_idx] = true;
                sample_query[sample_d].id_  = tmp_idx;
                sample_query[sample_d].key_ = tmp_key;
                norm_sample_q += tmp_key * tmp_key;
                ++sample_d;
            }
        }
        else {
            tmp_idx = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
            if (!checked[tmp_idx]) {
                tmp_key = 2 * query[idx] * query[idy];

                checked[tmp_idx] = true;
                sample_query[sample_d].id_  = tmp_idx;
                sample_query[sample_d].key_ = tmp_key;
                norm_sample_q += tmp_key * tmp_key;
                ++sample_d;
            }
        }
    }
    // multiply lambda
    float lambda = sqrt(M_ / norm_sample_q);
    norm_sample_q = norm_sample_q / (lambda * lambda);
    for (int i = 0; i < sample_d; ++i) sample_query[i].key_ *= lambda;

    delete[] prob;
    delete[] checked;
}

// -----------------------------------------------------------------------------
FH_wo_S::FH_wo_S(                   // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // number of hash tables
    float b,                            // interval ratio
    const float *data)                  // input data
    : n_pts_(n), dim_(d), b_(b), data_(data)
{
    fh_dim_ = d * (d + 1) / 2 + 1;
    M_      = MINREAL;

    // -------------------------------------------------------------------------
    //  calc centroid, l2-norm, and max l2-norm
    // -------------------------------------------------------------------------
    int   fh_dim_1  = fh_dim_ - 1;
    float *fh_data  = new float[fh_dim_1];
    float *norm     = new float[n];    // l2-norm
    float *centroid = new float[fh_dim_];

    memset(centroid, 0.0f, fh_dim_*sizeof(float));
    for (int i = 0; i < n; ++i) {
        calc_transform_centroid(&data[i*d], norm[i], centroid);
        if (M_ < norm[i]) M_ = norm[i];
    }
    for (int i = 0; i < fh_dim_1; ++i) centroid[i] /= n;

    for (int i = 0; i < n; ++i) {
        norm[i] = sqrt(M_ - norm[i]);
        centroid[fh_dim_1] += norm[i];
    }
    centroid[fh_dim_1] /= n;

    // -------------------------------------------------------------------------
    //  determine shift_id after shifting data objects to centroid
    // -------------------------------------------------------------------------
    Result *arr = new Result[n];
    for (int i = 0; i < n; ++i) {
        float dist = calc_transform_dist(&data[i*d], centroid);

        arr[i].id_  = i;
        arr[i].key_ = sqrt(dist + SQR(norm[i] - centroid[fh_dim_1]));
    }
    qsort(arr, n, sizeof(Result), ResultCompDesc);

    shift_id_ = new int[n];
    for (int i = 0; i < n; ++i) shift_id_[i] = arr[i].id_;

    // -------------------------------------------------------------------------
    //  divide datasets into blocks and build hash tables for each block
    // -------------------------------------------------------------------------
    int start = 0;
    while (start < n) {
        // partition block
        float min_radius  = b_ * arr[start].key_;
        int   block_index = start, cnt = 0;
        while (block_index < n && arr[block_index].key_ > min_radius) {
            ++block_index;
            if (++cnt >= MAX_BLOCK_NUM) break;
        }

        // add block
        const int *index = (const int*) shift_id_ + start;
        RQALSH *hash = new RQALSH(cnt, fh_dim_, m, index);

        for (int i = 0; i < cnt; ++i) {
            // calc hash value
            int idx = index[i];
            transform_data(&data[idx*d], fh_data);
            for (int j = 0; j < m; ++j) {
                float val = hash->calc_hash_value(fh_dim_1, j, fh_data);
                val += hash->a_[j*fh_dim_+fh_dim_1] * norm[idx];

                hash->tables_[j*cnt+i].id_  = i;
                hash->tables_[j*cnt+i].key_ = val;
            }
        }
        // sort hash tables in ascending order by hash values
        for (int i = 0; i < m; ++i) {
            qsort(&hash->tables_[i*cnt], cnt, sizeof(Result), ResultComp);
        }
        hash_.push_back(hash);
        start += cnt;
    }
    assert(start == n);

    // -------------------------------------------------------------------------
    //  release space
    // -------------------------------------------------------------------------
    delete[] fh_data;
    delete[] norm;
    delete[] centroid;
    delete[] arr;
}

// -----------------------------------------------------------------------------
void FH_wo_S::calc_transform_centroid(// data transformation with output centroid
    const float *data,                  // input data
    float &norm,                        // norm of fh_data (return)
    float *centroid)                    // fh data (return)
{
    norm = 0.0f;

    // calc square
    int cnt = 0;
    for (int i = 0; i < dim_; ++i) {
        float tmp = data[i] * data[i];
        centroid[cnt++] += tmp; norm += tmp * tmp; 
    }
    // calc differential
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            float tmp = data[i] * data[j];
            centroid[cnt++] += tmp; norm += tmp * tmp; 
        }
    }
}

// -----------------------------------------------------------------------------
float FH_wo_S::calc_transform_dist( // calc l2-dist after data transformation 
    const float *data,                  // input data
    const float *centroid)              // centroid after data transformation
{
    // calc square
    int   cnt  = 0;
    float dist = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        dist += SQR(data[i] * data[i] - centroid[cnt]);
        ++cnt;
    }
    // calc differential
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            dist += SQR(data[i] * data[j] - centroid[cnt]);
            ++cnt;
        }
    }
    return dist;
}

// -----------------------------------------------------------------------------
void FH_wo_S::transform_data(       // data transformation
    const float *data,                  // input data
    float *fh_data)                     // fh data (return)
{
    // calc square
    int cnt = 0;
    for (int i = 0; i < dim_; ++i) {
        fh_data[cnt++] = data[i] * data[i];
    }
    // calc differential
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            fh_data[cnt++] = data[i] * data[j];
        }
    }
}

// -----------------------------------------------------------------------------
FH_wo_S::~FH_wo_S()                 // destructor
{
    if (!hash_.empty()) {
        for (int i = 0; i < (int) hash_.size(); ++i) {
            delete hash_[i]; hash_[i] = NULL;
        }
        hash_.clear(); hash_.shrink_to_fit();
    }
    delete[] shift_id_; shift_id_ = NULL;
}

// -----------------------------------------------------------------------------
void FH_wo_S::display()             // display parameters
{
    printf("Parameters of FH_wo_S:\n");
    printf("    n       = %d\n",   n_pts_);
    printf("    dim     = %d\n",   dim_);
    printf("    fh_dim  = %d\n",   fh_dim_);
    printf("    b       = %.2f\n", b_);
    printf("    M       = %f\n",   sqrt(M_));
    printf("    #blocks = %d\n\n", (int) hash_.size());
}

// -----------------------------------------------------------------------------
int FH_wo_S::nns(                   // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   l,                            // separation threshold
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    // -------------------------------------------------------------------------
    //  query transformation
    // -------------------------------------------------------------------------
    float *fh_query = new float[fh_dim_];
    float norm_q    = 0.0f;
    transform_query(query, norm_q, fh_query);
    
    // -------------------------------------------------------------------------
    //  point-to-hyperplane NNS
    // -------------------------------------------------------------------------
    int   n_cand    = cand + top_k - 1;
    int   verif_cnt = 0;
    float fix_val   = norm_q + M_;
    std::vector<int> cand_list;
    
    for (auto hash : hash_) {
        // ---------------------------------------------------------------------
        //  check candidates returned by rqalsh
        // ---------------------------------------------------------------------
        float kfn_dist = -1.0f;
        if (list->isFull()) {
            float kdist = list->max_key();
            kfn_dist = sqrt(fix_val - 2 * kdist * kdist);
        }
        int size = hash->fns(l, n_cand, kfn_dist, (const float*) fh_query, 
            cand_list); // std::min(block_cand_, n_cand)

        for (int j = 0; j < size; ++j) {
            int   idx  = cand_list[j];
            float dist = fabs(calc_inner_product(dim_, &data_[idx*dim_], query));
            list->insert(dist, idx + 1);
        }

        // ---------------------------------------------------------------------
        //  update info
        // ---------------------------------------------------------------------
        verif_cnt += size; n_cand -= size; 
        if (n_cand <= 0) break;
    }
    delete[] fh_query;

    return verif_cnt;
}

// -----------------------------------------------------------------------------
void FH_wo_S::transform_query(      // query transformation
    const float *query,                 // input query
    float &norm_q,                      // l2-norm sqr of q after transform (return)
    float *fh_query)                    // fh_query after transform (return)
{
    norm_q = 0.0f;

    // calc square
    int cnt = 0;
    for (int i = 0; i < dim_; ++i) {
        float tmp = query[i] * query[i];
        fh_query[cnt++] = tmp; norm_q += tmp * tmp; 
    }
    // calc differential
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            float tmp = 2 * query[i] * query[j];
            fh_query[cnt++] = tmp; norm_q += tmp * tmp; 
        }
    }
    fh_query[cnt] = 0.0f;

    // multiply lambda
    float lambda = sqrt(M_ / norm_q);
    norm_q = norm_q / (lambda * lambda);
    for (int i = 0; i < cnt; ++i) fh_query[i] *= lambda;
}

// -----------------------------------------------------------------------------
FH_Minus::FH_Minus(                 // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // #hash tables
    int   s,                            // scale factor of dimension
    const float *data)                  // input data
    : n_pts_(n), dim_(d), scale_(s), data_(data)
{
    sample_dim_ = d * s;
    fh_dim_     = d * (d + 1) / 2 + 1;
    M_          = MINREAL;
    lsh_        = new RQALSH(n, fh_dim_, m, NULL);
    
    // -------------------------------------------------------------------------
    //  build hash tables for rqalsh
    // -------------------------------------------------------------------------
    int    fh_dim_1 = fh_dim_ - 1; assert(sample_dim_ <= fh_dim_1);
    float  *norm    = new float[n];
    
    int    sample_d = -1; // actual sample dimension
    float  *prob    = new float[d];
    bool   *checked = new bool[fh_dim_];
    Result *sample_data = new Result[sample_dim_];

    for (int i = 0; i < n; ++i) {
        // data transformation
        transform_data(&data[i*d], prob, checked, norm[i], sample_d, sample_data);
        if (M_ < norm[i]) M_ = norm[i];

        // calc partial hash value
        for (int j = 0; j < m; ++j) {
            float val = lsh_->calc_hash_value(sample_d, j, sample_data);
            lsh_->tables_[j*n+i].id_  = i;
            lsh_->tables_[j*n+i].key_ = val;
        }
    }
    // calc the final hash value
    for (int i = 0; i < n; ++i) {
        float tmp = sqrt(M_ - norm[i]);
        for (int j = 0; j < m; ++j) {
            lsh_->tables_[j*n+i].key_ += lsh_->a_[j*fh_dim_+fh_dim_1] * tmp;
        }
    }
    // sort hash tables in ascending order by hash values
    for (int i = 0; i < m; ++i) {
        qsort(&lsh_->tables_[i*n], n, sizeof(Result), ResultComp);
    }

    // -------------------------------------------------------------------------
    //  release space
    // -------------------------------------------------------------------------
    delete[] sample_data;
    delete[] prob;
    delete[] checked;
    delete[] norm;
}

// -----------------------------------------------------------------------------
void FH_Minus::transform_data(      // data transformation
    const  float *data,                 // input data
    float  *prob,                       // probability vector
    bool   *checked,                    // is checked?
    float  &norm,                       // norm of fh_data (return)
    int    &sample_d,                   // sample dimension (return)
    Result *sample_data)                // sample data (return)
{
    // calc probability vector and the l2-norm-square of data
    float norm2 = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        norm2 += data[i] * data[i];
        prob[i] = norm2;
    }
    for (int i = 0; i < dim_; ++i) prob[i] /= norm2;

    // randomly sample coordinate of data as the coordinate of sample_data
    int   tmp_idx, tmp_idy, idx, idy;
    float tmp_key;

    norm = 0.0f;
    sample_d = 0;
    memset(checked, false, fh_dim_*sizeof(bool));

    // first consider the largest coordinate
    tmp_idx = dim_-1;
    tmp_key = data[tmp_idx] * data[tmp_idx];

    checked[tmp_idx] = true;
    sample_data[sample_d].id_  = tmp_idx;
    sample_data[sample_d].key_ = tmp_key;
    norm += tmp_key * tmp_key;
    ++sample_d;
    
    // consider the combination of the left coordinates
    for (int i = 1; i < sample_dim_; ++i) {
        tmp_idx = sampling(dim_-1, prob);
        tmp_idy = sampling(dim_, prob);
        idx = std::min(tmp_idx, tmp_idy); idy = std::max(tmp_idx, tmp_idy);

        if (idx == idy) {
            tmp_idx = idx;
            if (!checked[tmp_idx]) {
                tmp_key = data[idx] * data[idx];
                
                checked[tmp_idx] = true;
                sample_data[sample_d].id_  = tmp_idx;
                sample_data[sample_d].key_ = tmp_key;
                norm += tmp_key * tmp_key; 
                ++sample_d; 
            }
        }
        else {
            tmp_idx = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
            if (!checked[tmp_idx]) {
                tmp_key = data[idx] * data[idy];

                checked[tmp_idx] = true;
                sample_data[sample_d].id_  = tmp_idx;
                sample_data[sample_d].key_ = tmp_key;
                norm += tmp_key * tmp_key; 
                ++sample_d;
            }
        }
    }
}

// -----------------------------------------------------------------------------
int FH_Minus::sampling(             // sampling coordinate based on prob
    int   d,                            // dimension
    const float *prob)                  // input probability
{
    float end = prob[d-1];
    float rnd = uniform(0.0f, end);
    return std::lower_bound(prob, prob + d, rnd) - prob;
}

// -----------------------------------------------------------------------------
FH_Minus::~FH_Minus()               // destructor
{
    if (lsh_ != NULL) { delete lsh_; lsh_ = NULL; }
}

// -----------------------------------------------------------------------------
void FH_Minus::display()            // display parameters
{
    printf("Parameters of FH_Minus:\n");
    printf("    n            = %d\n",   n_pts_);
    printf("    dim          = %d\n",   dim_);
    printf("    scale factor = %d\n",   scale_);
    printf("    fh_dim       = %d\n",   fh_dim_);
    printf("    sample_dim   = %d\n",   sample_dim_);
    printf("    m            = %d\n",   lsh_->m_);
    printf("    M            = %f\n\n", sqrt(M_));
}

// -----------------------------------------------------------------------------
int FH_Minus::nns(                  // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   l,                            // separation threshold
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    // query transformation
    int    sample_d = -1;
    Result *sample_query = new Result[sample_dim_];
    transform_query(query, sample_d, sample_query);

    // conduct furthest neighbor search by rqalsh
    std::vector<int> cand_list;
    int verif_cnt = lsh_->fns(l, cand + top_k - 1, MINREAL, sample_d,
        (const Result*) sample_query, cand_list);

    // calc true distance for candidates returned by qalsh
    for (int i = 0; i < verif_cnt; ++i) {
        int   idx  = cand_list[i];
        float dist = fabs(calc_inner_product(dim_, &data_[idx*dim_], query));
        list->insert(dist, idx + 1);
    }
    delete[] sample_query;

    return verif_cnt;
}

// -----------------------------------------------------------------------------
void FH_Minus::transform_query(     // query transformation
    const  float *query,                // input query
    int    &sample_d,                   // sample dimension (return)
    Result *sample_query)               // sample query (return)
{
    // calc probability vector
    float norm2 = 0.0f;
    float *prob = new float[dim_];
    for (int i = 0; i < dim_; ++i) {
        norm2 += query[i] * query[i];
        prob[i] = norm2;
    }
    for (int i = 0; i < dim_; ++i) prob[i] /= norm2;

    // randomly sample coordinate of query as the coordinate of sample_query
    int   tmp_idx, tmp_idy, idx, idy;
    float tmp_key;
    bool  *checked = new bool[fh_dim_];
    memset(checked, false, fh_dim_*sizeof(bool));

    float norm_sample_q = 0.0f;
    sample_d = 0;
    for (int i = 0; i < sample_dim_; ++i) {
        tmp_idx = sampling(dim_, prob);
        tmp_idy = sampling(dim_, prob);
        idx = std::min(tmp_idx, tmp_idy); idy = std::max(tmp_idx, tmp_idy);

        if (idx == idy) {
            tmp_idx = idx;
            if (!checked[tmp_idx]) {
                tmp_key = query[idx] * query[idx];

                checked[tmp_idx] = true;
                sample_query[sample_d].id_  = tmp_idx;
                sample_query[sample_d].key_ = tmp_key;
                norm_sample_q += tmp_key * tmp_key;
                ++sample_d;
            }
        }
        else {
            tmp_idx = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
            if (!checked[tmp_idx]) {
                tmp_key = 2 * query[idx] * query[idy];

                checked[tmp_idx] = true;
                sample_query[sample_d].id_  = tmp_idx;
                sample_query[sample_d].key_ = tmp_key;
                norm_sample_q += tmp_key * tmp_key;
                ++sample_d;
            }
        }
    }
    // multiply lambda
    float lambda = sqrt(M_ / norm_sample_q);
    for (int i = 0; i < sample_d; ++i) sample_query[i].key_ *= lambda;
    
    delete[] prob;
    delete[] checked;
}

// -----------------------------------------------------------------------------
FH_Minus_wo_S::FH_Minus_wo_S(       // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // #hash tables
    const float *data)                  // input data
    : n_pts_(n), dim_(d), data_(data)
{
    fh_dim_ = d * (d + 1) / 2 + 1;
    M_      = MINREAL;
    lsh_    = new RQALSH(n, fh_dim_, m, NULL);
    
    // -------------------------------------------------------------------------
    //  build hash tables for rqalsh
    // -------------------------------------------------------------------------
    int   fh_dim_1 = fh_dim_ - 1;
    float *fh_data = new float[fh_dim_1];
    float *norm    = new float[n];    

    for (int i = 0; i < n; ++i) {
        // data transformation
        transform_data(&data[i*d], norm[i], fh_data);
        if (M_ < norm[i]) M_ = norm[i];

        // calc partial hash value
        for (int j = 0; j < m; ++j) {
            float val = lsh_->calc_hash_value(fh_dim_1, j, fh_data);
            lsh_->tables_[j*n+i].id_  = i;
            lsh_->tables_[j*n+i].key_ = val;
        }
    }
    // calc the final hash value
    for (int i = 0; i < n; ++i) {
        float tmp = sqrt(M_ - norm[i]);
        for (int j = 0; j < m; ++j) {
            lsh_->tables_[j*n+i].key_ += lsh_->a_[j*fh_dim_+fh_dim_1] * tmp;
        }
    }
    // sort hash tables in ascending order by hash values
    for (int i = 0; i < m; ++i) {
        qsort(&lsh_->tables_[i*n], n, sizeof(Result), ResultComp);
    }

    // -------------------------------------------------------------------------
    //  release space
    // -------------------------------------------------------------------------
    delete[] fh_data;
    delete[] norm;
}

// -----------------------------------------------------------------------------
void FH_Minus_wo_S::transform_data( // data transformation
    const float *data,                  // input data
    float &norm,                        // norm of fh_data (return)
    float *fh_data)                     // fh data (return)
{
    // calc square
    int cnt = 0;
    float norm2 = 0.0f, norm4 = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        float tmp = data[i] * data[i];
        fh_data[cnt++] = tmp; norm2 += tmp; norm4 += tmp * tmp;
    }
    norm = (norm2 * norm2 + norm4) / 2.0f;

    // calc differential
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            fh_data[cnt++] = data[i] * data[j];
        }
    }
}

// -----------------------------------------------------------------------------
FH_Minus_wo_S::~FH_Minus_wo_S()     // destructor
{
    if (lsh_ != NULL) { delete lsh_; lsh_ = NULL; }
}

// -----------------------------------------------------------------------------
void FH_Minus_wo_S::display()       // display parameters
{
    printf("Parameters of FH_Hash:\n");
    printf("    n      = %d\n",   n_pts_);
    printf("    dim    = %d\n",   dim_);
    printf("    fh_dim = %d\n",   fh_dim_);
    printf("    m      = %d\n",   lsh_->m_);
    printf("    M      = %f\n\n", sqrt(M_));
}

// -----------------------------------------------------------------------------
int FH_Minus_wo_S::nns(             // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   l,                            // separation threshold
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    // query transformation
    float *fh_query = new float[fh_dim_];
    transform_query(query, fh_query);

    // conduct furthest neighbor search by rqalsh
    std::vector<int> cand_list;
    int verif_cnt = lsh_->fns(l, cand + top_k - 1, MINREAL, 
        (const float *) fh_query, cand_list);

    // calc actual distance for candidates returned by qalsh
    for (int i = 0; i < verif_cnt; ++i) {
        int   idx  = cand_list[i];
        float dist = fabs(calc_inner_product(dim_, &data_[idx*dim_], query));
        list->insert(dist, idx + 1);
    }
    delete[] fh_query;

    return verif_cnt;
}

// -----------------------------------------------------------------------------
void FH_Minus_wo_S::transform_query(// query transformation
    const float *query,                 // input query
    float *fh_query)                    // fh_query after transform (return)
{
    // calc square
    int cnt = 0;
    float norm_q = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        float tmp = query[i] * query[i];
        fh_query[cnt++] = tmp; norm_q += tmp * tmp;
    }
    // calc differential
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            float tmp = 2 * query[i] * query[j];
            fh_query[cnt++] = tmp; norm_q += tmp * tmp;
        }
    }
    fh_query[cnt] = 0.0f;

    // multiply lambda
    float lambda = sqrt(M_ / norm_q);
    for (int i = 0; i < cnt; ++i) fh_query[i] *= lambda;
}

} // end namespace p2h
