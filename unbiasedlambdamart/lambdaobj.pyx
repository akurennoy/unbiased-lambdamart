cimport cython
cimport openmp
from cython.parallel import prange

from libc.math cimport fabs, pow
from libc.stdlib cimport malloc, free
from libc.string cimport memset

from argsort cimport argsort


cdef double get_value(
    double score, 
    double* table, 
    int n_bins,
    double min_arg,
    double max_arg,
    double idx_factor
) nogil:
    if score <= min_arg:
        return table[0]
    elif score >= max_arg:
        return table[n_bins - 1]
    else:
        return table[<int>((score - min_arg) * idx_factor)]


cdef void get_gradients_for_one_query_fixed_t(
    double* gains, 
    double* preds,
    double* t_plus,
    double* t_minus,
    long* ranks0, 
    long start, 
    long end,
    double* grad, 
    double* hess, 
    double* discounts,
    double inverse_max_dcg,
    double* sigmoid_table,
    long n_sigmoid_bins,
    double min_sigmoid_arg,
    double max_sigmoid_arg,
    double sigmoid_idx_factor
) nogil:
    cdef int cnt
    cnt = <int>(end - start)
    cdef int* sorted_idx = <int*> malloc(cnt * sizeof(int))

    argsort((&preds[0]) + start, sorted_idx, cnt)

    cdef double best_score, worst_score
    best_score = preds[start + sorted_idx[0]]
    worst_score = preds[start + sorted_idx[cnt - 1]]
    cdef int should_adjust 
    if best_score != worst_score:
        should_adjust = 1
    else:
        should_adjust = 0
    
    cdef double p_labmda, p_hess
    cdef double gain_high, gain_low
    cdef double score_high, score_low, delta_score
    cdef double abs_delta_ndcg, paired_discount, gain_diff
    cdef int high, low

    cdef int i
    cdef int j
    for i in range(cnt):
        high = sorted_idx[i]
        gain_high = gains[start + high]
        score_high = preds[start + high]
        for j in range(cnt):
            low = sorted_idx[j]
            gain_low = gains[start + low]
            score_low = preds[start + low]
            rank_high = ranks0[start + high]
            rank_low = ranks0[start + low]
            if gain_high > gain_low:
                delta_score = score_high - score_low
                p_lambda = get_value(
                    delta_score,
                    sigmoid_table,
                    n_sigmoid_bins,
                    min_sigmoid_arg,
                    max_sigmoid_arg,
                    sigmoid_idx_factor
                )
                p_hess = p_lambda * (2.0 - p_lambda)

                gain_diff = gain_high - gain_low
                paired_discount = fabs(discounts[1 + j] - discounts[1 + i])
                abs_delta_ndcg = gain_diff * paired_discount * inverse_max_dcg
                if should_adjust == 1:
                    abs_delta_ndcg /= (0.01 + fabs(delta_score))
                    
                p_lambda *= -abs_delta_ndcg
                p_hess *= 2 * abs_delta_ndcg

                p_lambda /= t_plus[rank_high] * t_minus[rank_low]
                p_hess /= t_plus[rank_high] * t_minus[rank_low]

                grad[start + high] += p_lambda
                hess[start + high] += p_hess
                grad[start + low] -= p_lambda
                hess[start + low] += p_hess
    free(sorted_idx)


cdef void _get_gradients_fixed_t(
    double* gains, 
    double* preds,
    double* t_plus, 
    double* t_minus,
    long* ranks0,
    long n_preds,
    long* groups,
    long* query_boundaries,
    long n_queries,
    double* discounts,
    double* inverse_max_dcgs,
    double* sigmoid_table,
    long n_sigmoid_bins,
    double min_sigmoid_arg,
    double max_sigmoid_arg,
    double sigmoid_idx_factor,
    double* grad,
    double* hess
) nogil:
    memset(grad, 0, n_preds * sizeof(double))
    memset(hess, 0, n_preds * sizeof(double))

    cdef double inverse_max_dcg
    cdef long start, end, i

    for i in prange(n_queries, nogil=True):
        start = query_boundaries[i]
        end = query_boundaries[i + 1]
        inverse_max_dcg = inverse_max_dcgs[i]
        get_gradients_for_one_query_fixed_t(
            gains, 
            preds,
            t_plus,
            t_minus,
            ranks0, 
            start, 
            end,
            grad, 
            hess, 
            discounts,
            inverse_max_dcg,
            sigmoid_table,
            n_sigmoid_bins,
            min_sigmoid_arg,
            max_sigmoid_arg,
            sigmoid_idx_factor
        )


def get_unbiased_gradients_fixed_t(
  double[::1] gains, 
  double[::1] preds,
  double[::1] t_plus, 
  double[::1] t_minus,
  long[::1] ranks0,
  long n_preds,
  long[::1] groups,
  long[::1] query_boundaries,
  long n_queries,
  double[::1] discounts,
  double[::1] inverse_max_dcgs,
  double[::1] sigmoid_table,
  long n_sigmoid_bins,
  double min_sigmoid_arg,
  double max_sigmoid_arg,
  double sigmoid_idx_factor,
  double[::1] grad,
  double[::1] hess
):
    _get_gradients_fixed_t(
        (&gains[0]),
        (&preds[0]),
        (&t_plus[0]),
        (&t_minus[0]),
        (&ranks0[0]),
        n_preds,
        (&groups[0]),
        (&query_boundaries[0]),
        n_queries,
        (&discounts[0]),
        (&inverse_max_dcgs[0]),
        (&sigmoid_table[0]),
        n_sigmoid_bins,
        min_sigmoid_arg,
        max_sigmoid_arg,
        sigmoid_idx_factor,
        (&grad[0]),
        (&hess[0])
    )


cdef void get_gradients_for_one_query(
    double* gains, 
    double* preds,
    double* cur_t_plus,
    double* cur_t_minus,
    double* t_plus_buffer,
    double* t_minus_buffer,
    long n_positions,
    long* ranks0, 
    long start, 
    long end,
    double* grad, 
    double* hess, 
    double* discounts,
    double inverse_max_dcg,
    double* sigmoid_table,
    long n_sigmoid_bins,
    double min_sigmoid_arg,
    double max_sigmoid_arg,
    double sigmoid_idx_factor,
    double* log_table,
    long n_log_bins,
    double min_log_arg,
    double max_log_arg,
    double log_idx_factor
) nogil:
    cdef int tid = openmp.omp_get_thread_num()
    cdef int cnt
    cnt = <int>(end - start)
    cdef int* sorted_idx = <int*> malloc(cnt * sizeof(int))

    argsort((&preds[0]) + start, sorted_idx, cnt)

    cdef double best_score, worst_score
    best_score = preds[start + sorted_idx[0]]
    worst_score = preds[start + sorted_idx[cnt - 1]]
    cdef int should_adjust 
    if best_score != worst_score:
        should_adjust = 1
    else:
        should_adjust = 0
    
    cdef double p_labmda, p_hess
    cdef double gain_high, gain_low
    cdef double score_high, score_low, delta_score
    cdef double abs_delta_ndcg, paired_discount, gain_diff, loss
    cdef int high, low, rank_high, rank_low

    cdef int i
    cdef int j
    for i in range(cnt):
        high = sorted_idx[i]
        gain_high = gains[start + high]
        score_high = preds[start + high]
        for j in range(cnt):
            low = sorted_idx[j]
            gain_low = gains[start + low]
            score_low = preds[start + low]
            rank_high = ranks0[start + high]
            rank_low = ranks0[start + low]
            if gain_high > gain_low:
                delta_score = score_high - score_low
                p_lambda = get_value(
                    delta_score,
                    sigmoid_table,
                    n_sigmoid_bins,
                    min_sigmoid_arg,
                    max_sigmoid_arg,
                    sigmoid_idx_factor
                )
                p_hess = p_lambda * (2.0 - p_lambda)

                gain_diff = gain_high - gain_low
                paired_discount = fabs(discounts[1 + j] - discounts[1 + i])
                abs_delta_ndcg = gain_diff * paired_discount * inverse_max_dcg
                if should_adjust == 1:
                    abs_delta_ndcg /= (0.01 + fabs(delta_score))
                    
                p_lambda *= -abs_delta_ndcg
                p_hess *= 2 * abs_delta_ndcg

                p_lambda /= cur_t_plus[rank_high] * cur_t_minus[rank_low]
                p_hess /= cur_t_plus[rank_high] * cur_t_minus[rank_low]

                grad[start + high] += p_lambda
                hess[start + high] += p_hess
                grad[start + low] -= p_lambda
                hess[start + low] += p_hess

                loss = abs_delta_ndcg * get_value(
                    delta_score,
                    log_table,
                    n_log_bins,
                    min_log_arg,
                    max_log_arg,
                    log_idx_factor
                )
                t_plus_buffer[tid * n_positions + rank_high] += loss / cur_t_minus[rank_low]
                t_minus_buffer[tid * n_positions + rank_low] += loss / cur_t_plus[rank_high] 
    
    free(sorted_idx)


cdef void _get_gradients(
    double* gains, 
    double* preds,
    long* ranks0,
    long n_preds,
    long* groups,
    long* query_boundaries,
    long n_queries,
    double* discounts,
    double* inverse_max_dcgs,
    double* sigmoid_table,
    long n_sigmoid_bins,
    double min_sigmoid_arg,
    double max_sigmoid_arg,
    double sigmoid_idx_factor,
    double* log_table,
    long n_log_bins,
    double min_log_arg,
    double max_log_arg,
    double log_idx_factor,
    double p,
    long n_positions,
    double* grad,
    double* hess,
    double* t_plus, # must contain the previous or initial values
    double* t_minus # must contain the previous or initial values
) nogil:
    memset(grad, 0, n_preds * sizeof(double))
    memset(hess, 0, n_preds * sizeof(double))

    cdef int n_threads = openmp.omp_get_max_threads()

    cdef double* t_plus_buffer = <double*> malloc(n_threads * n_positions * sizeof(double))
    cdef double* t_minus_buffer = <double*> malloc(n_threads * n_positions * sizeof(double))
    memset(t_plus_buffer,  0, n_threads * n_positions * sizeof(double))
    memset(t_minus_buffer, 0, n_threads * n_positions * sizeof(double))

    cdef double inverse_max_dcg
    cdef long start, end, i

    for i in prange(n_queries, nogil=True):
        start = query_boundaries[i]
        end = query_boundaries[i + 1]
        inverse_max_dcg = inverse_max_dcgs[i]
        get_gradients_for_one_query(gains, 
                                    preds,
                                    t_plus,
                                    t_minus,
                                    t_plus_buffer,
                                    t_minus_buffer,
                                    n_positions,
                                    ranks0, 
                                    start, 
                                    end,
                                    grad, 
                                    hess, 
                                    discounts,
                                    inverse_max_dcg,
                                    sigmoid_table,
                                    n_sigmoid_bins,
                                    min_sigmoid_arg,
                                    max_sigmoid_arg,
                                    sigmoid_idx_factor,
                                    log_table,
                                    n_log_bins,
                                    min_log_arg,
                                    max_log_arg,
                                    log_idx_factor)

    memset(t_plus, 0, n_positions * sizeof(double))
    memset(t_minus, 0, n_positions * sizeof(double))
    cdef int tid, pos
    for tid in range(n_threads):
        for pos in range(n_positions):
            t_plus[pos] += t_plus_buffer[tid * n_positions + pos]
            t_minus[pos] += t_minus_buffer[tid * n_positions + pos]
    for pos in range(1, n_positions):
        t_plus[pos] = pow(t_plus[pos] / t_plus[0], 1 / (1 + p))
        t_minus[pos] = pow(t_minus[pos] / t_minus[0], 1 / (1 + p))
    t_plus[0] = 1.0
    t_minus[0] = 1.0
    
    free(t_plus_buffer)
    free(t_minus_buffer)


def get_unbiased_gradients(
  double[::1] gains, 
  double[::1] preds,
  long[::1] ranks0,
  long n_preds,
  long[::1] groups,
  long[::1] query_boundaries,
  long n_queries,
  double[::1] discounts,
  double[::1] inverse_max_dcgs,
  double[::1] sigmoid_table,
  long n_sigmoid_bins,
  double min_sigmoid_arg,
  double max_sigmoid_arg,
  double sigmoid_idx_factor,
  double[::1] log_table,
  long n_log_bins,
  double min_log_arg,
  double max_log_arg,
  double log_idx_factor,
  double p,
  long n_positions,
  double[::1] grad,
  double[::1] hess,
  double[::1] t_plus, 
  double[::1] t_minus
):
    _get_gradients(
      (&gains[0]),
      (&preds[0]),
      (&ranks0[0]),
      n_preds,
      (&groups[0]),
      (&query_boundaries[0]),
      n_queries,
      (&discounts[0]),
      (&inverse_max_dcgs[0]),
      (&sigmoid_table[0]),
      n_sigmoid_bins,
      min_sigmoid_arg,
      max_sigmoid_arg,
      sigmoid_idx_factor,
      (&log_table[0]),
      n_log_bins,
      min_log_arg,
      max_log_arg,
      log_idx_factor,
      p,
      n_positions,
      (&grad[0]),
      (&hess[0]),
      (&t_plus[0]),
      (&t_minus[0])
    )