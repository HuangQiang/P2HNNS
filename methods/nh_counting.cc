#include "nh_counting.h"

namespace p2h {

// -----------------------------------------------------------------------------
NH_Counting::NH_Counting(           // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // #hash tables
    int   s,                            // scale factor of dimension
    const float *data)                  // input data
    : n_pts_(n), dim_(d), scale_(s), data_(data)
{
    sample_dim_ = d * s;
    nh_dim_     = d * (d + 1) / 2 + 1;
    M_          = MINREAL;
    lsh_        = new QALSH(n, nh_dim_, m);
    
    // -------------------------------------------------------------------------
    //  build hash tables for rqalsh
    // -------------------------------------------------------------------------
    int    nh_dim_1 = nh_dim_ - 1; assert(sample_dim_ <= nh_dim_1);
    float  *norm    = new float[n];
    
    int    sample_d = -1;           // actual sample dimension
    float  *prob    = new float[d];
    bool   *checked = new bool[nh_dim_];
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
            lsh_->tables_[j*n+i].key_ += lsh_->a_[j*nh_dim_+nh_dim_1] * tmp;
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
void NH_Counting::transform_data(   // data transformation
    const  float *data,                 // input data
    float  *prob,                       // probability vector
    bool   *checked,                    // is checked?
    float  &norm,                       // norm of nh_data (return)
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

    norm     = 0.0f;
    sample_d = 0;
    memset(checked, false, nh_dim_*sizeof(bool));

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
int NH_Counting::sampling(          // sampling coordinate based on prob
    int   d,                            // dimension
    const float *prob)                  // input probability
{
    float end = prob[d-1];
    float rnd = uniform(0.0f, end);
    return std::lower_bound(prob, prob + d, rnd) - prob;
}

// -----------------------------------------------------------------------------
NH_Counting::~NH_Counting()         // destructor
{
    if (lsh_ != NULL) { delete lsh_; lsh_ = NULL; }
}

// -----------------------------------------------------------------------------
void NH_Counting::display()         // display parameters
{
    printf("Parameters of NH_Counting:\n");
    printf("    n            = %d\n",   n_pts_);
    printf("    dim          = %d\n",   dim_);
    printf("    scale factor = %d\n",   scale_);
    printf("    nh_dim       = %d\n",   nh_dim_);
    printf("    sample_dim   = %d\n",   sample_dim_);
    printf("    m            = %d\n",   lsh_->m_);
    printf("    M            = %f\n\n", sqrt(M_));
}

// -----------------------------------------------------------------------------
int NH_Counting::nns(               // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   l,                            // collision threshold
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
    int verif_cnt = lsh_->nns(l, cand + top_k - 1, sample_d,
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
void NH_Counting::transform_query(  // query transformation
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
    bool  *checked = new bool[nh_dim_];
    memset(checked, false, nh_dim_*sizeof(bool));

    float norm_sample_q = 0.0f;
    sample_d = 0;
    for (int i = 0; i < sample_dim_; ++i) {
        tmp_idx = sampling(dim_, prob);
        tmp_idy = sampling(dim_, prob);
        idx = std::min(tmp_idx, tmp_idy); idy = std::max(tmp_idx, tmp_idy);

        if (idx == idy) {
            tmp_idx = idx;
            if (!checked[tmp_idx]) {
                tmp_key = -query[idx] * query[idx];
                
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
                tmp_key = -2 * query[idx] * query[idy];

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
NH_Counting_wo_S::NH_Counting_wo_S( // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // #hash tables
    const float *data)                  // input data
    : n_pts_(n), dim_(d), data_(data)
{
    nh_dim_ = d * (d + 1) / 2 + 1;
    M_      = MINREAL;
    lsh_    = new QALSH(n, nh_dim_, m);
    
    // -------------------------------------------------------------------------
    //  build hash tables for qalsh
    // -------------------------------------------------------------------------
    int   nh_dim_1 = nh_dim_ - 1;
    float *norm    = new float[n];
    float *nh_data = new float[nh_dim_1];

    for (int i = 0; i < n; ++i) {
        // data transformation
        transform_data(&data[i*d], norm[i], nh_data);
        if (M_ < norm[i]) M_ = norm[i];

        // calc partial hash value
        for (int j = 0; j < m; ++j) {
            float val = lsh_->calc_hash_value(nh_dim_1, j, nh_data);
            lsh_->tables_[j*n+i].id_  = i;
            lsh_->tables_[j*n+i].key_ = val;
        }
    }
    // calc the final hash value
    for (int i = 0; i < n; ++i) {
        float tmp = sqrt(M_ - norm[i]);
        for (int j = 0; j < m; ++j) {
            lsh_->tables_[j*n+i].key_ += lsh_->a_[j*nh_dim_+nh_dim_1] * tmp;
        }
    }
    // sort hash tables in ascending order by hash values
    for (int i = 0; i < m; ++i) {
        qsort(&lsh_->tables_[i*n], n, sizeof(Result), ResultComp);
    }

    // -------------------------------------------------------------------------
    //  release space
    // -------------------------------------------------------------------------
    delete[] nh_data;
    delete[] norm;
}

// -----------------------------------------------------------------------------
void NH_Counting_wo_S::transform_data(// data transformation
    const float *data,                  // input data
    float &norm,                        // l2-norm-sqr of nh_data (return)
    float *nh_data)                     // nh_data (return)
{
    // calc square
    int cnt = 0;
    float norm2 = 0.0f, norm4 = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        float tmp = data[i] * data[i];
        nh_data[cnt++] = tmp; norm2 += tmp; norm4 += tmp * tmp;
    }
    norm = (norm2 * norm2 + norm4) / 2.0f;

    // calc differential
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            nh_data[cnt++] = data[i] * data[j];
        }
    }
}

// -----------------------------------------------------------------------------
NH_Counting_wo_S::~NH_Counting_wo_S()// destructor
{
    if (lsh_ != NULL) { delete lsh_; lsh_ = NULL; }
}

// -----------------------------------------------------------------------------
void NH_Counting_wo_S::display()    // display parameters
{
    printf("Parameters of NH_Counting_wo_S:\n");
    printf("    n      = %d\n",   n_pts_);
    printf("    dim    = %d\n",   dim_);
    printf("    nh_dim = %d\n",   nh_dim_);
    printf("    m      = %d\n",   lsh_->m_);
    printf("    M      = %f\n\n", sqrt(M_));
}

// -----------------------------------------------------------------------------
int NH_Counting_wo_S::nns(          // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   l,                            // separation threshold
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    // query transformation
    float *nh_query = new float[nh_dim_];
    transform_query(query, nh_query);

    // conduct nearest neighbor search by qalsh
    std::vector<int> cand_list;
    int verif_cnt = lsh_->nns(l, cand + top_k - 1, nh_query, cand_list);

    // calc actual distance for candidates returned by qalsh
    for (int i = 0; i < verif_cnt; ++i) {
        int   idx  = cand_list[i];
        float dist = fabs(calc_inner_product(dim_, &data_[idx*dim_], query));
        list->insert(dist, idx + 1);
    }
    delete[] nh_query;

    return verif_cnt;
}

// -----------------------------------------------------------------------------
void NH_Counting_wo_S::transform_query(// query transformation
    const float *query,                 // input query
    float *nh_query)                    // nh_query after transform (return)
{
    // calc square
    int cnt = 0;
    float norm_q = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        float tmp = -query[i] * query[i];
        nh_query[cnt++] = tmp; norm_q += tmp * tmp;
    }
    // calc differential
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            float tmp = -2 * query[i] * query[j];
            nh_query[cnt++] = tmp; norm_q += tmp * tmp;
        }
    }
    nh_query[cnt] = 0.0f;

    // multiply lambda
    float lambda = sqrt(M_ / norm_q);
    for (int i = 0; i < cnt; ++i) nh_query[i] *= lambda;
}

} // end namespace p2h
