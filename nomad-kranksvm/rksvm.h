#ifndef _RANKSVM_H
#define _RANKSVM_H
#include <tbb/scalable_allocator.h>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>
#include <tbb/atomic.h>
#include <tbb/tbb.h>
#include <tbb/compat/thread>
#include <mpi.h>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

struct rksvm_node
{
	int index;
	double value;
};

struct rksvm_problem
{
	int l;//local number of training instances
	int global_l;//total number of training instances
	int *query;
	double *y;
	struct rksvm_node **x;
	int *length_of_each_rksvm_node;/*for scheduler*/
};

enum {  L2R_RANK };	/* rksvm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

struct data_node
{
	bool first_time;/*whether used at the first time*/
	int length;/*the length of a data_node*/
	int initial_rank;
	int current_rank;
	int global_index;/*the global index of a rksvm_node*/
	struct rksvm_node *x;
};

struct rksvm_parameter
{
	int rksvm_type;
	int kernel_type;
	int degree;	/* for poly */
	int thread_count; /*the number of threads*/
	int nr_ranks;/*the number of machinces or processes*/
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

	/* these are for training only */
	double eps;	/* stopping criteria */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
};

//
// rksvm_model
// 
struct rksvm_model
{
	struct rksvm_parameter param;	/* parameter */
	int nr_class;		/* number of classes, = 2 in regression/one class svm */
	int l;			/* total #SV */
	struct rksvm_node **SV;		/* SVs (SV[l]) */
	double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
};

struct rksvm_model *rksvm_train(const struct rksvm_problem *prob, const struct rksvm_parameter *param);

int rksvm_save_model(const char *model_file_name, const struct rksvm_model *model);
//struct rksvm_model *rksvm_load_model(const char *model_file_name);

double rksvm_predict_values(const struct rksvm_model *model, const struct rksvm_node *x, double* dec_values);
double rksvm_predict(const struct rksvm_model *model, const struct rksvm_node *x);

void rksvm_free_model_content(struct rksvm_model *model_ptr);
void rksvm_free_and_destroy_model(struct rksvm_model **model_ptr_ptr);
void rksvm_destroy_param(struct rksvm_parameter *param);

const char *rksvm_check_parameter(const struct rksvm_problem *prob, const struct rksvm_parameter *param);

void rksvm_set_print_string_function(void (*print_func)(const char *));
void group_queries(const rksvm_problem *prob, int *nr_subset_ret, int **start_ret, int **count_ret, int *perm);
void eval_list(double *label, double *target, int *query, int l, double *result_ret);

#ifdef __cplusplus
}
#endif

int mpi_get_rank();

int mpi_get_size();

void mpi_exit(const int status);

template<typename T>
void mpi_allreduce(T *buf, const int count, MPI_Datatype type, MPI_Op op)
{
  std::vector<T> buf_reduced(count);
	MPI_Allreduce(buf, buf_reduced.data(), count, type, op, MPI_COMM_WORLD);
	for(int i=0;i<count;i++)
		buf[i] = buf_reduced[i];
}

#endif /* _RANKSVM_H */
