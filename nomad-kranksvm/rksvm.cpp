#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
#include <functional> 
#include <unistd.h> 
#include <sched.h>
#include <mutex>
#include <vector>
#include <thread>
#include <condition_variable>
#include <mpi.h>
#include <omp.h>
#include <tbb/tbb.h>
#include <tbb/compat/thread>
#include <tbb/scalable_allocator.h>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>
#include <tbb/atomic.h>
#include "rksvm.h"
#include "tron.h"
#include "selectiontree.h"
#include "scheduler.h"

typedef tbb::concurrent_queue<data_node*, callocator<data_node*>> con_queue;
template <typename T>
using sallocator = tbb::scalable_allocator<T>;

/*#ifdef __cplusplus
extern "C" {
#endif
extern int dspmv_(char *, int *, double *, double *, double *, int *, double *,double *, int *);
#ifdef __cplusplus
}
#endif*/

/* LEVEL 2 BLAS */

//typedef float Qfloat;
//typedef signed char schar;
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct id_and_value
{
	int id;
	double value;
};
static int compare_id_and_value(const void *a, const void *b)
{
	struct id_and_value *ia = (struct id_and_value *)a;
	struct id_and_value *ib = (struct id_and_value *)b;
	if(ia->value > ib->value)
		return -1;
	if(ia->value < ib->value)
		return 1;
	return 0;
}

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void (*rksvm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
	if(mpi_get_rank()!=0)
		return;
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*rksvm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

int save_Q(const char *file_name, const rksvm_problem *prob, double *Q)
{
	int i;
	int local_l = prob->l;
	int global_l = prob->global_l;

	FILE *fp = fopen(file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	fprintf(fp, "l_count: %d\n", local_l);
	fprintf(fp, "g_count: %d\n", global_l);

	for(i=0; i<local_l; i++)
	{
		int j;
		for(j=0; j<global_l; j++)
			fprintf(fp, "%.16g(%d,%d) ", Q[i*global_l+j],i,j);
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}


double dot(const rksvm_node *px, const rksvm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}

double k_function(const rksvm_node *x, const rksvm_node *y,
			  const rksvm_parameter& param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
			return dot(x,y);
		case POLY:
			return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
		case RBF:
		{
			double sum = 0;
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{	
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}
			
			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;
		default:
			return 0;  // Unreachable 
	}
}

int cyclic_loading_rank(int current_rank, int nr_ranks){///{{{
	 int next_rank;
	 next_rank = (current_rank+1)%nr_ranks;
	 return next_rank;
 }////}}}

void nomad_fun(const rksvm_problem *prob, const rksvm_parameter *param, Scheduler *scheduler, double *Q)
{
	int l = prob->l;
	int global_l = prob->global_l;
	int thread_count = param->thread_count;
	int nr_ranks = param->nr_ranks;
	int current_rank = mpi_get_rank();
	int *nr_send = scheduler->nr_send;
	int *nr_recv = scheduler->nr_recv;
	//atomic variables
	atomic<int> count_setup_threads;
	count_setup_threads = 0;
	atomic<int> computed_data_nodes;//record the number of data_nodes that have been utilized.
	computed_data_nodes = 0;
	atomic<int> sended_count;//record the number of data_nodes that have been sended.
	sended_count = 0;
	atomic<int> recvd_count;////record the number of data_nodes that have been received.
	recvd_count = 0;
	// two auxiliary atomic flags for both sending and receiving
	atomic<bool> flag_send_ready;
	flag_send_ready = false;
	atomic<bool> flag_receive_ready;
	flag_receive_ready = false;

	//build several job queues and one sending queue
	con_queue *job_queues = callocator<con_queue>().allocate(thread_count);
	for(int i=0;i<thread_count;i++)
			callocator<con_queue>().construct(job_queues + i);
	con_queue send_queue;
	//initilize job queues
	int interval = (int)ceil((double)prob->l/thread_count);
	int thread_id = 0;
	for(int i=0;i<l;i++)
	{
		data_node *copy_x = nullptr;
		copy_x = scheduler->pop();
		if((i!=0)&&(i%interval==0))
			thread_id++;
		job_queues[thread_id].push(copy_x);
	}

	//the first function
	auto QMatrix = [&](struct data_node *copy_x)->void{//{{{
		int i = 0;
		int global_index = copy_x->global_index;
		for(i=0;i<l;i++)
		{
			rksvm_node *s = prob->x[i];
			rksvm_node *t = copy_x->x;
			Q[global_index + i*global_l] = k_function(s,t,*param);
		}
		return;
	};//}}}	

	//the second function
	auto computer_fun = [&](int thread_id)->void{///{{{
		count_setup_threads++;
		while(count_setup_threads < thread_count)
		{
			std::this_thread::yield();
		}
		while(true)
		{
			if(computed_data_nodes == global_l)
				break;
			data_node *copy_x = nullptr;
			bool success = job_queues[thread_id].try_pop(copy_x);
			if(success)
			{
				if(copy_x->first_time)
				{
					QMatrix(copy_x);
					computed_data_nodes++;
					if(nr_ranks==1)
					{
						int lth = copy_x->length;
						callocator<rksvm_node>().deallocate(copy_x->x, lth);
						callocator<data_node>().destroy(copy_x);
						callocator<data_node>().deallocate(copy_x,1);
					}
					else
					{
						copy_x->first_time = false;
						send_queue.push(copy_x);
						flag_send_ready = true;
					}
				}
				else
				{
					QMatrix(copy_x);
					computed_data_nodes++;
					copy_x->current_rank = current_rank;
					int next_rank = cyclic_loading_rank(copy_x->current_rank, nr_ranks);
					if(next_rank==copy_x->initial_rank)
					{
						int lth = copy_x->length; 
						callocator<rksvm_node>().deallocate(copy_x->x, lth);
						callocator<data_node>().destroy(copy_x);
						callocator<data_node>().deallocate(copy_x,1);
					}
					else
					{
						send_queue.push(copy_x);
					}
				}
			}
		}
	return;
	};///}}}

	//the third function
	auto sender_fun = [&]()->void{///{{{
		while(flag_send_ready == false)
		{
			std::this_thread::yield();
		}
		int lth;
		int msg_bytenum;
		while(true)
		{
			if(sended_count == nr_send[current_rank])
				break;
			data_node *copy_x = nullptr;
			bool success = send_queue.try_pop(copy_x);
			
			if(success)
			{
				int next_rank = cyclic_loading_rank(copy_x->current_rank, nr_ranks);
				if(next_rank == copy_x->initial_rank)
				{
					lth = copy_x->length; 
					callocator<rksvm_node>().deallocate(copy_x->x, lth);
					callocator<data_node>().destroy(copy_x);
					callocator<data_node>().deallocate(copy_x,1);
				}
				else
				{
					lth = copy_x->length; 
					msg_bytenum = sizeof(bool)+4*sizeof(int)+lth*sizeof(rksvm_node);
					char *send_message = sallocator<char>().allocate(msg_bytenum);
					*(reinterpret_cast<bool *>(send_message)) = copy_x->first_time;
					*(reinterpret_cast<int *>(send_message + sizeof(bool))) = copy_x->length;
					*(reinterpret_cast<int *>(send_message + sizeof(bool) + sizeof(int))) = copy_x->initial_rank;
					*(reinterpret_cast<int *>(send_message + sizeof(bool) + 2*sizeof(int))) = copy_x->current_rank;
					*(reinterpret_cast<int *>(send_message + sizeof(bool) + 3*sizeof(int))) = copy_x->global_index;
					rksvm_node *dest = reinterpret_cast<rksvm_node *>(send_message + sizeof(bool) + 4*sizeof(int));
					std::copy(copy_x->x, copy_x->x + lth, dest);
					flag_receive_ready = true;
					MPI_Ssend(send_message, msg_bytenum, MPI_CHAR, next_rank, 1, MPI_COMM_WORLD);
					//destroying
					callocator<rksvm_node>().deallocate(copy_x->x, lth);
					callocator<data_node>().destroy(copy_x);
					callocator<data_node>().deallocate(copy_x,1);
					//record the sended count
					sended_count++;
					sallocator<char>().deallocate(send_message, msg_bytenum);
				}
			}
		}
		return;
	};///}}}

	//the fourth function
	auto receiver_fun = [&]()->void{///{{{
		
		while(flag_receive_ready == false)
		{
			std::this_thread::yield();
		}
		int flag = 0;
		int src_rank;
		int lth;
		MPI_Status status;
		while(true)
		{
			if(recvd_count == nr_recv[mpi_get_rank()])
				break;
			MPI_Iprobe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag, &status);
			if(flag == 0)
			{
				std::this_thread::yield();
			}
			else
			{
				src_rank = status.MPI_SOURCE;
				int msg_size = 0; 
				MPI_Get_count(&status, MPI_CHAR, &msg_size);
				char *recv_message = sallocator<char>().allocate(msg_size);
				MPI_Recv(recv_message, msg_size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &status);
				//recovering
				data_node *copy_x = callocator<data_node>().allocate(1);
				copy_x->first_time = *(reinterpret_cast<bool *>(recv_message));
				copy_x->length = *(reinterpret_cast<int *>(recv_message + sizeof(bool)));
				copy_x->initial_rank = *(reinterpret_cast<int *>(recv_message + sizeof(bool) + sizeof(int)));
				copy_x->current_rank = *(reinterpret_cast<int *>(recv_message + sizeof(bool) + 2*sizeof(int)));
				copy_x->global_index = *(reinterpret_cast<int *>(recv_message + sizeof(bool) + 3*sizeof(int)));
				rksvm_node *dest = reinterpret_cast<rksvm_node *>(recv_message + sizeof(bool) + 4*sizeof(int));
				//please notice that the approach to recover cp_x->x
				lth = copy_x->length;
				copy_x->x = callocator<rksvm_node>().allocate(lth);
				memcpy(copy_x->x, dest, (size_t)sizeof(rksvm_node)*lth);
				sallocator<char>().deallocate(recv_message, msg_size); 
				//push an item to the job_queue who has the smallest number of items.
				//In doing so, the dynamic loading balancing can be achieved.	
				int smallest_items_thread_id = 0;	
				auto smallest_items = job_queues[0].unsafe_size();	
				for(int i=1;i<thread_count;i++)	
				{
					auto tmp = job_queues[i].unsafe_size();		
					if(tmp < smallest_items)		
					{			
						smallest_items_thread_id = i;			
						smallest_items = tmp;		
					}	
				}
				job_queues[smallest_items_thread_id].push(copy_x);
				recvd_count++;
			}
		}
		return;
	};///}}}
	//notice that tht above functions are important to our program

	//create some functional threads
	std::vector<std::thread> computers;
	std::thread *sender = nullptr;
	std::thread *receiver = nullptr;
	for (int i=0; i < thread_count; i++){
		computers.push_back(std::thread(computer_fun, i));
    }
	if(nr_ranks>1)
	{
		sender = new std::thread(sender_fun);
		receiver = new std::thread(receiver_fun);
	}
	//wait until data loading and initialization
	//the main thread is used to test the results
	while(count_setup_threads < thread_count){
		std::this_thread::yield();
	}
	if(current_rank==0)
	{
		printf("Start to compute kernel matrix!\n");
		fflush(stdout);
	}
	//test the time used to compute Q
	tbb::tick_count start_time = tbb::tick_count::now();
	while(true)
	{
		if(nr_ranks==1)
		{
			if(computed_data_nodes == global_l)
				break;
		}
		else
		{
			if((computed_data_nodes==global_l)&&
				(sended_count==nr_send[current_rank])&&
				(recvd_count==nr_recv[current_rank]))
				break;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);//sychronization
	double elapsed_seconds = (tbb::tick_count::now() - start_time).seconds();
	if(current_rank==0)
	{
		printf("Computing Q has done!, the elapsed time is %f secs\n", elapsed_seconds);
		fflush(stdout);
	}

	callocator<con_queue>().deallocate(job_queues, thread_count); 
	for(auto &th: computers)
		th.join();
	if(nr_ranks > 1)
	{
		sender->join();
		receiver->join();
		delete sender;
		delete receiver;
	}
	return;
}

// construct and solve various formulations
//

struct SolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		//double r;	
};

class l2r_rank_fun: public function
{
	public:
		l2r_rank_fun(const rksvm_problem *prob, const rksvm_parameter *param, 
		Scheduler *scheduler, struct SolutionInfo *si);
		~l2r_rank_fun();

		double fun(double *w);
		void grad(double *w, double *g);
		void Hv(double *s, double *Hs);
		int get_nr_variable(void);
	private:
		double *Q;
		void Qv(double *s, double *Qs);

		double C;
		double *z;
		double *ATAQb;
		double *ATe;
		int *l_plus;
		int *l_minus;
		double *gamma_plus;
		double *gamma_minus;
		const rksvm_problem *prob;
		const rksvm_parameter *param;
		int nr_subset;
		int *perm;
		int *start;
		int *count;
		id_and_value **pi;
		int *nr_class;
		int *int_y;
		struct SolutionInfo *si;

		//the variables we added to this program
		Scheduler *scheduler;//
		int thread_count;//
		int current_rank;
		int global_l;//
		int *local_l;//
		//int *nr_send;//
		//int *nr_recv;//
		int *start_ptr;//
		double *gz;//
		//double *gATAQb;//
		//double *gATe;//
};

l2r_rank_fun::l2r_rank_fun(const rksvm_problem *prob, const rksvm_parameter *param, 
		Scheduler *scheduler, struct SolutionInfo *si)
{
	this->si = si;
	this->param = param;
	si->rho = 0;
	si->upper_bound_p = INF;
	si->upper_bound_n = INF;
	int l=prob->l;
	this->prob = prob;
	this->C = param->C;
	this->thread_count = param->thread_count;//
	this->current_rank = mpi_get_rank();//
	this->global_l = prob->global_l;//
	z = new double[l];

	int i,j,k;
	perm = new int[l];
	group_queries(prob, &nr_subset ,&start, &count, perm);
	pi = new id_and_value* [nr_subset];

#pragma omp parallel for default(shared) if(nr_subset > 50)
	for (int i=0;i<nr_subset;i++)
	{
		pi[i] = new id_and_value[count[i]];
	}

	double *y=prob->y;
	int_y = new int[prob->l];
	nr_class = new int[nr_subset];
	l_plus = new int[l];
	l_minus = new int[l];
	gamma_plus = new double[l];
	gamma_minus = new double[l];
	ATAQb = new double[l];
	ATe = new double[l];

	// the variable we have changed;
	this->scheduler = scheduler;
	this->local_l = scheduler->local_l;
	this->start_ptr = scheduler->start_ptr;
	//this->nr_recv = scheduler->nr_recv;
	//this->nr_send = scheduler->nr_send;
	gz = new double[global_l];
	//gATAQb = new double[global_];
	//gATe = new double[global_l];
	Q = new double[l*global_l];

	//here, it shows how to compute Q through TBB library.
	nomad_fun(prob, param, scheduler, Q);

//testing Q 
//char *file = "/home/jing/model/Q.txt";
//save_Q(file, prob, Q);	
//mpi_exit(1);


#pragma omp parallel for default(shared) private(i,j,k)	
	for (i=0;i<nr_subset;i++)
	{
		k=1;
		for (j=0;j<count[i];j++)
		{
			pi[i][j].id=perm[j+start[i]];
			pi[i][j].value=y[perm[j+start[i]]];
		}
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);
		int_y[pi[i][count[i]-1].id]=1;
		for(j=count[i]-2;j>=0;j--)
		{
			if (pi[i][j].value>pi[i][j+1].value)
				k++;
			int_y[pi[i][j].id]=k;
		}
		nr_class[i]=k;
	}
}

l2r_rank_fun::~l2r_rank_fun()
{
	int i;
	delete[] Q;
	delete[] local_l;
	delete[] gz;
	//delete[] gATe;
	//delete[] gATAQb;
	delete[] l_plus;
	delete[] l_minus;
	delete[] gamma_plus;
	delete[] gamma_minus;
	delete[] z;

#pragma omp parallel for default(shared) if(nr_subset > 50)
	for (int i=0;i<nr_subset;i++)
	{
		delete[] pi[i];
	}

	delete[] pi;
	delete[] int_y;
	delete[] nr_class;
	delete[] ATe;
	delete[] ATAQb;
}

double l2r_rank_fun::fun(double *w)// w is with the size of global_l 
{
	int i,j,k;
	double f = 0.0;
	double reg = 0.0;
	int l=prob->l;
	selectiontree *T;
	Qv(w,z);
	//generate gz via MPI_Allgatherv
	MPI_Allgatherv((void*)z, l, MPI_DOUBLE, (void*)gz, local_l, start_ptr, MPI_DOUBLE, MPI_COMM_WORLD);

#pragma omp parallel for default(shared) private(i,j,k,T)
	for (i=0;i<nr_subset;i++)
	{
		for (j=0;j<count[i];j++)
		{
			pi[i][j].id= perm[j+start[i]];
			pi[i][j].value = z[perm[j+start[i]]];
		}
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);

		T=new selectiontree(nr_class[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(int_y[pi[i][k].id],pi[i][k].value);

				k++;
			}
			T->count_smaller(int_y[pi[i][j].id],&l_minus[pi[i][j].id], &gamma_minus[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;

		T = new selectiontree(nr_class[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(int_y[pi[i][k].id],pi[i][k].value);
				k--;
			}
			T->count_larger(int_y[pi[i][j].id],&l_plus[pi[i][j].id], &gamma_plus[pi[i][j].id]);
		}
		delete T;
	}

//#pragma omp parallel for default(shared) private(i) reduction(+:f) schedule(dynamic)
	for(i=0;i<global_l;i++)
	{
		f += w[i]*gz[i];
	}

#pragma omp parallel for default(shared) private(i)
	for (i=0;i<l;i++)
	{
		ATe[i] = l_minus[i] - l_plus[i];
		ATAQb[i] = (l_plus[i]+l_minus[i])*gz[i+start_ptr[current_rank]]-gamma_plus[i]-gamma_minus[i];
	}

//#pragma omp parallel for default(shared) //private(i) //reduction(+:reg) schedule(runtime)
	for (int i=0;i<l;i++)
	{
		//#pragma omp atomic
		reg += C*(gz[i+start_ptr[current_rank]]*(ATAQb[i] - 2 * ATe[i]) + l_minus[i]);
	}

	mpi_allreduce(&reg, 1, MPI_DOUBLE, MPI_SUM);	
	f /= 2.0;
	f += reg;
	si->obj=f;
	return(f);
}

void l2r_rank_fun::grad(double *w, double *g)
{
	int i;
	int l=prob->l;
	double *lg = new double[l];
	double *tmp_vector = new double[l];
	double *gtmp_vector = new double[global_l];

#pragma omp parallel for default(shared) private(i)
	for (i=0;i<l;i++)
	{
		tmp_vector[i] = ATAQb[i] - ATe[i];
	}

	MPI_Allgatherv((void*)tmp_vector, l, MPI_DOUBLE, (void*)gtmp_vector, local_l, start_ptr, MPI_DOUBLE, MPI_COMM_WORLD);
	Qv(gtmp_vector, lg);
	MPI_Allgatherv((void*)lg, l, MPI_DOUBLE, (void*)g, local_l, start_ptr, MPI_DOUBLE, MPI_COMM_WORLD);

#pragma omp parallel for default(shared) private(i)
	for(i=0;i<global_l;i++)
	{
		g[i] = gz[i] + 2*C*g[i];
	}

	delete[] tmp_vector;
	delete[] gtmp_vector;
	delete[] lg;
}
int l2r_rank_fun::get_nr_variable(void)
{
	return prob->global_l;//
}

void l2r_rank_fun::Hv(double *s, double *Hs)
{
	int i,j,k;
	int l=prob->l;
	double *wa = new double[global_l];
	double *lHs = new double[l];//
	double *lwa = new double[l];//
	selectiontree *T;
	double *tmp_vector = new double[l];
	double *gtmp_vector = new double[global_l];//
	int tmp_value;
	double gamma_tmp;
	Qv(s, lwa);
	MPI_Allgatherv((void*)lwa, l, MPI_DOUBLE, (void*)wa, local_l, start_ptr, MPI_DOUBLE, MPI_COMM_WORLD);//

#pragma omp parallel for private(i,j,k,T,gamma_tmp)
	for (i=0;i<nr_subset;i++)
	{
		T=new selectiontree(nr_class[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(int_y[pi[i][k].id],lwa[pi[i][k].id]);
				k++;
			}
			T->count_smaller(int_y[pi[i][j].id],&tmp_value, &tmp_vector[pi[i][j].id]);
		}
		delete T;

		k=count[i]-1;
		T = new selectiontree(nr_class[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(int_y[pi[i][k].id],lwa[pi[i][k].id]);
				k--;
			}
			T->count_larger(int_y[pi[i][j].id],&tmp_value, &gamma_tmp);
			tmp_vector[pi[i][j].id] += gamma_tmp;
		}
		delete T;
	}

#pragma omp parallel for default(shared) private(i)
	for (i=0;i<l;i++)
	{
		tmp_vector[i]=wa[i+start_ptr[current_rank]]*((double)l_plus[i]+(double)l_minus[i])-tmp_vector[i];//
	}
	
	MPI_Allgatherv((void*)tmp_vector, l, MPI_DOUBLE, (void*)gtmp_vector, local_l, start_ptr, MPI_DOUBLE, MPI_COMM_WORLD);//
	Qv(gtmp_vector, lHs);//
	MPI_Allgatherv((void*)lHs, l, MPI_DOUBLE, (void*)Hs, local_l, start_ptr, MPI_DOUBLE, MPI_COMM_WORLD);//

#pragma omp parallel for default(shared) private(i)
	for(i=0;i<global_l;i++)
	{
		Hs[i] = wa[i] + 2*C*Hs[i];//
	}
	delete[] wa;
	delete[] lwa;//
	delete[] tmp_vector;
	delete[] gtmp_vector;
	delete[] lHs;
}

void l2r_rank_fun::Qv(double *s, double *Qs)
{
	int i, j;
	int l = prob->l;

#pragma omp parallel for default(shared) private(i,j)
	for(i=0;i<l;i++)
	{ 
		Qs[i] = 0.0;
		for(j=0;j<global_l;j++)
		{
			Qs[i] += s[j]*Q[i*global_l+j];
		}
	}
}

/*void l2r_rank_fun::Qv(double *s, double *Qs)
{
	double one = 1.0;
	int l = prob->l;
	int inc = 1;
	double zero = 0;
	dspmv_("U",&l,&one,Q,s,&inc,&zero,Qs,&inc);
}*/
//
// decision_function
//
struct decision_function
{
	double *alpha;
	double rho;	
};

static decision_function rksvm_train_one(const rksvm_problem *prob, 
	const rksvm_parameter *param, Scheduler *scheduler)
{
	int current_rank = mpi_get_rank();
	double *alpha = Malloc(double, prob->global_l);//
	struct SolutionInfo si;
	int *start_ptr = scheduler->start_ptr;//
	double begin, end;

	begin = MPI_Wtime();
	function *fun_obj=NULL;
	fun_obj = new l2r_rank_fun(prob, param, scheduler, &si);
	TRON tron_obj(fun_obj, param->eps);
	tron_obj.set_print_string(rksvm_print_string);
	tron_obj.tron(alpha);
	delete fun_obj;
	end = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	info("Training time = %g\n", end-begin);
	info("obj = %f, rho = %f\n",si.obj,si.rho);
	// output SVs
	int nSV = 0;
	int nBSV = 0;
	for(int i=0;i<prob->l;i++)
	{
		if(fabs(alpha[i+start_ptr[current_rank]]) > 0)
		{
			++nSV;
			if(prob->y[i] > 0)
			{
				if(fabs(alpha[i+start_ptr[current_rank]]) >= si.upper_bound_p)
					++nBSV;
			}
			else
			{
				if(fabs(alpha[i+start_ptr[current_rank]]) >= si.upper_bound_n)
					++nBSV;
			}
		}
	}
	mpi_allreduce(&nSV, 1, MPI_INT, MPI_SUM);
	mpi_allreduce(&nBSV, 1, MPI_INT, MPI_SUM);
	info("nSV = %d, nBSV = %d\n",nSV,nBSV);

	decision_function f;
	f.alpha = alpha;
	f.rho = si.rho;
	return f;
}


//
// Interface functions
//
rksvm_model *rksvm_train(const rksvm_problem *prob, const rksvm_parameter *param)
{
	rksvm_model *model = Malloc(rksvm_model,1);
	model->param = *param;
	int current_rank = mpi_get_rank();//
	int nr_ranks = param->nr_ranks;//
	Scheduler *scheduler = nullptr;
	scheduler = new Scheduler(prob, param);
	int *start_ptr = scheduler->start_ptr;

	if(param->rksvm_type == L2R_RANK)
	{
		model->nr_class = 2;
		model->sv_coef = Malloc(double *,1);//model-sv_coef[0][~]

		decision_function f = rksvm_train_one(prob,param,scheduler);// cut two parameters, Cp and Cn, care about this change
		model->rho = Malloc(double,1);//model->rho[0]
		model->rho[0] = f.rho;

		int nSV = 0;
		int gnSV = 0;
		int i;
		//int *local_nSV = new int[nr_ranks];//
		//int *start_nSV_ptr = new int[nr_ranks];//

		for(i=0;i<prob->l;i++)
		{
			if(fabs(f.alpha[i+start_ptr[current_rank]]) > 0) 
				++nSV;
		}
		gnSV = nSV;
		//MPI_Allgather(&nSV, 1, MPI_INT, local_nSV, 1, MPI_INT, MPI_COMM_WORLD);
		//for(i=1;i<nr_ranks;i++)
			//start_nSV_ptr[i] = start_nSV_ptr[i-1]+ local_nSV[i-1];

		mpi_allreduce(&gnSV, 1, MPI_INT, MPI_SUM);
		info("total number of SVs:%d\n",gnSV);

		model->l = nSV;
		model->SV = Malloc(rksvm_node *,nSV);
		model->sv_coef[0] = Malloc(double,nSV);
		//model->gSV = Malloc(rksvm_node *,gnSV);//
		//model->gsv_coef[0] = Malloc(double,gnSV);//
		int j = 0;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i+start_ptr[current_rank]]) > 0)
			{
				model->SV[j] = prob->x[i];
				model->sv_coef[0][j] = f.alpha[i+start_ptr[current_rank]];
				++j;
			}

		free(f.alpha);
	}
	else
	{
		info("A wrong rksvm_type! Please check it up carefully!");
		mpi_exit(1);
	}
	//delete scheduler;?????
	return model;
}


double rksvm_predict_values(const rksvm_model *model, const rksvm_node *x, double* dec_values)
{
	int i;
	if(model->param.rksvm_type == L2R_RANK)
	{
		double *sv_coef = model->sv_coef[0];
		double sum = 0;
		for(i=0;i<model->l;i++)
			sum += sv_coef[i] * k_function(x,model->SV[i],model->param);
		sum -= model->rho[0];
		*dec_values = sum;
		
		return sum;
	}
	else
	{
		info("Wrong svm type, please check it up.");
		mpi_exit(1);
	}
}

double rksvm_predict(const rksvm_model *model, const rksvm_node *x)
{
	double *dec_values;
	dec_values = Malloc(double, 1);
	double pred_result = rksvm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}


static const char *rksvm_type_table[] =
{
	"l2r_rank",NULL
};

static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

int rksvm_save_model(const char *model_file_name, const rksvm_model *model)
{
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;
//mpi_exit(1);
	char *old_locale = strdup(setlocale(LC_ALL, NULL));
//mpi_exit(1);
	setlocale(LC_ALL, "C");

	const rksvm_parameter& param = model->param;

	fprintf(fp,"rksvm_type %s\n", rksvm_type_table[param.rksvm_type]);
	fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

	if(param.kernel_type == POLY)
		fprintf(fp,"degree %d\n", param.degree);

	if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
		fprintf(fp,"gamma %g\n", param.gamma);

	if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
		fprintf(fp,"coef0 %g\n", param.coef0);

	int nr_class = model->nr_class;
	int l = model->l;
	fprintf(fp, "nr_class %d\n", nr_class);
	fprintf(fp, "total_sv %d\n",l);
	
	{
		fprintf(fp, "rho");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->rho[i]);
		fprintf(fp, "\n");
	}	

	fprintf(fp, "SV\n");
	const double * const *sv_coef = model->sv_coef;
	const rksvm_node * const *SV = model->SV;

	for(int i=0;i<l;i++)
	{
		for(int j=0;j<nr_class-1;j++)
			fprintf(fp, "%.16g ",sv_coef[j][i]);

		const rksvm_node *p = SV[i];

		if(param.kernel_type == PRECOMPUTED)
			fprintf(fp,"0:%d ",(int)(p->value));
		else
			while(p->index != -1)
			{
				fprintf(fp,"%d:%.8g ",p->index,p->value);
				p++;
			}
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

/*rksvm_model *rksvm_load_model(const char *model_file_name)// we will check it in the svm-predict.c file.
{
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL) return NULL;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	// read parameters

	rksvm_model *model = Malloc(rksvm_model,1);
	rksvm_parameter& param = model->param;
	model->rho = NULL;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);

		if(strcmp(cmd,"rksvm_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;rksvm_type_table[i];i++)
			{
				if(strcmp(rksvm_type_table[i],cmd)==0)
				{
					param.rksvm_type=i;
					break;
				}
			}
			if(rksvm_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown svm type.\n");
				
				setlocale(LC_ALL, old_locale);
				free(old_locale);
				free(model->rho);
				free(model);
				return NULL;
			}
		}
		else if(strcmp(cmd,"kernel_type")==0)
		{		
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i],cmd)==0)
				{
					param.kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");
				
				setlocale(LC_ALL, old_locale);
				free(old_locale);
				free(model->rho);
				free(model);
				return NULL;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			fscanf(fp,"%d",&param.degree);
		else if(strcmp(cmd,"gamma")==0)
			fscanf(fp,"%lf",&param.gamma);
		else if(strcmp(cmd,"coef0")==0)
			fscanf(fp,"%lf",&param.coef0);
		else if(strcmp(cmd,"nr_class")==0)
			fscanf(fp,"%d",&model->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			fscanf(fp,"%d",&model->l);
		else if(strcmp(cmd,"rho")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->rho = Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->rho[i]);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c = getc(fp);
				if(c==EOF || c=='\n') break;	
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			
			setlocale(LC_ALL, old_locale);
			free(old_locale);
			free(model->rho);
			free(model);
			return NULL;
		}
	}

	// read sv_coef and SV

	int elements = 0;
	long pos = ftell(fp);

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	char *p,*endptr,*idx,*val;

	while(readline(fp)!=NULL)
	{
		p = strtok(line,":");
		while(1)
		{
			p = strtok(NULL,":");
			if(p == NULL)
				break;
			++elements;
		}
	}
	elements += model->l;

	fseek(fp,pos,SEEK_SET);

	int m = model->nr_class - 1;
	int l = model->l;
	model->sv_coef = Malloc(double *,m);
	int i;
	for(i=0;i<m;i++)
		model->sv_coef[i] = Malloc(double,l);
	model->SV = Malloc(rksvm_node*,l);
	rksvm_node *x_space = NULL;
	if(l>0) x_space = Malloc(rksvm_node,elements);

	int j=0;
	for(i=0;i<l;i++)
	{
		readline(fp);
		model->SV[i] = &x_space[j];

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p,&endptr);
		for(int k=1;k<m;k++)
		{
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p,&endptr);
		}

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			x_space[j].value = strtod(val,&endptr);

			++j;
		}
		x_space[j++].index = -1;
	}
	free(line);

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	return model;
}*/

void rksvm_free_model_content(rksvm_model* model_ptr)
{

	if(model_ptr->sv_coef)
	{
		for(int i=0;i<model_ptr->nr_class-1;i++)
			free(model_ptr->sv_coef[i]);
	}

	free(model_ptr->SV);
	model_ptr->SV = NULL;

	free(model_ptr->sv_coef);
	model_ptr->sv_coef = NULL;

	free(model_ptr->rho);
	model_ptr->rho = NULL;
}

void rksvm_free_and_destroy_model(rksvm_model** model_ptr_ptr)
{
	if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
	{
		rksvm_free_model_content(*model_ptr_ptr);
		free(*model_ptr_ptr);
		*model_ptr_ptr = NULL;
	}
}

const char *rksvm_check_parameter(const rksvm_problem *prob, const rksvm_parameter *param)
{
	// rksvm_type

	int rksvm_type = param->rksvm_type;
	if(rksvm_type != L2R_RANK)
		return "unknown svm type";
	
	// kernel_type, degree
	
	int kernel_type = param->kernel_type;
	if(kernel_type != LINEAR &&
	   kernel_type != POLY &&
	   kernel_type != RBF &&
	   kernel_type != SIGMOID &&
	   kernel_type != PRECOMPUTED)
		return "unknown kernel type";

	if(param->gamma < 0)
		return "gamma < 0";

	if(param->degree < 0)
		return "degree of polynomial kernel < 0";

	//eps,C,nu,p,shrinking
	if(param->eps <= 0)
		return "eps <= 0";

	return NULL;
}

void rksvm_set_print_string_function(void (*print_func)(const char *))
{
	if(print_func == NULL)
		rksvm_print_string = &print_string_stdout;
	else
		rksvm_print_string = print_func;
}

void group_queries(const rksvm_problem *prob, int *nr_subset_ret, int **start_ret, int **count_ret, int *perm)
{
	int i,j;
	int l = prob->l;
	int max_nr_subset = 16;
	int nr_subset = 0;
	int *query = Malloc(int,max_nr_subset);
	int *count = Malloc(int,max_nr_subset);
	int *data_query = Malloc(int,l);

	for(i=0;i<l;i++)
	{
		int this_query = (int)prob->query[i];
		for(j=0;j<nr_subset;j++)
		{
			if(this_query == query[j])
			{
				++count[j];
				break;
			}
		}
		data_query[i] = j;
		if(j == nr_subset)
		{
			if(nr_subset == max_nr_subset)
			{
				max_nr_subset *= 2;
				query = (int *)realloc(query,max_nr_subset*sizeof(int));
				count = (int *)realloc(count,max_nr_subset*sizeof(int));
			}
			query[nr_subset] = this_query;
			count[nr_subset] = 1;
			++nr_subset;
		}
	}

	int *start = Malloc(int,nr_subset);
	start[0] = 0;
	for(i=1;i<nr_subset;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_query[i]]] = i;
		++start[data_query[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_subset;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_subset_ret = nr_subset;
	*start_ret = start;
	*count_ret = count;
	free(query);
	free(data_query);
}
static void group_queries(const int *query_id, int l, int *nr_query_ret, int **start_ret, int **count_ret, int *perm)
{
	int max_nr_query = 16;
	int nr_query = 0;
	int *query = Malloc(int,max_nr_query);
	int *count = Malloc(int,max_nr_query);
	int *data_query = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_query = (int)query_id[i];
		int j;
		for(j=0;j<nr_query;j++)
		{
			if(this_query == query[j])
			{
				++count[j];
				break;
			}
		}
		data_query[i] = j;
		if(j == nr_query)
		{
			if(nr_query == max_nr_query)
			{
				max_nr_query *= 2;
				query = (int *)realloc(query,max_nr_query * sizeof(int));
				count = (int *)realloc(count,max_nr_query * sizeof(int));
			}
			query[nr_query] = this_query;
			count[nr_query] = 1;
			++nr_query;
		}
	}

	int *start = Malloc(int,nr_query);
	start[0] = 0;
	for(i=1;i<nr_query;i++)
		start[i] = start[i-1] + count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_query[i]]] = i;
		++start[data_query[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_query;i++)
		start[i] = start[i-1] + count[i-1];

	*nr_query_ret = nr_query;
	*start_ret = start;
	*count_ret = count;
	free(query);
	free(data_query);
}

void eval_list(double *label, double *target, int *query, int l, double *result_ret)
{
	int q,i,j,k;
	int nr_query;
	int *start = NULL;
	int *count = NULL;
	int *perm = Malloc(int, l);
	id_and_value *order_perm;
	int true_query;
	int ndcg_size;
	long long totalnc = 0, totalnd = 0;
	long long nc = 0;
	long long nd = 0;
	double tmp;
	double accuracy = 0;
	int *l_plus;
	int *int_y;
	int same_y = 0;
	double *ideal_dcg;
	double *dcg;
	double meanndcg = 0;
	double ndcg;
	double dcg_yahoo,idcg_yahoo,ndcg_yahoo;
	selectiontree *T;
	group_queries(query, l, &nr_query, &start, &count, perm);
	true_query = nr_query;
	for (q=0;q<nr_query;q++)
	{
		//We use selection trees to compute pairwise accuracy
		nc = 0;
		nd = 0;
		l_plus = new int[count[q]];
		int_y = new int[count[q]];
		order_perm = new id_and_value[count[q]];
		int *perm_q = &perm[start[q]];
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = i;
			order_perm[i].value = label[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		int_y[order_perm[count[q]-1].id] = 1;
		same_y = 0;
		k = 1;
		for(i=count[q]-2;i>=0;i--)
		{
			if (order_perm[i].value != order_perm[i+1].value)
			{
				same_y = 0;
				k++;
			}
			else
				same_y++;
			int_y[order_perm[i].id] = k;
			nc += (count[q]-1 - i - same_y);
		}
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = i;
			order_perm[i].value = target[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		//total pairs
		T = new selectiontree(k);
		j = count[q] - 1;
		for (i=count[q] - 1;i>=0;i--)
		{
			while (j>=0 && ( order_perm[j].value < order_perm[i].value))
			{
				T->insert_node(int_y[order_perm[j].id], tmp);
				j--;
			}
			T->count_larger(int_y[order_perm[i].id], &l_plus[order_perm[i].id], &tmp);
		}
		delete T;

		for (i=0;i<count[q];i++)
			nd += l_plus[i];
		nc -= nd;
		if (nc != 0 || nd != 0)
			accuracy += double(nc)/double(nc+nd);
		else
			true_query--;
		totalnc += nc;
		totalnd += nd;
		delete[] l_plus;
		delete[] int_y;
		delete[] order_perm;
	}
	result_ret[0] = (double)totalnc/(double)(totalnc+totalnd);
	for (q=0;q<nr_query;q++)
	{
		ndcg_size = min(10,count[q]);
		ideal_dcg = new double[count[q]];
		dcg = new double[count[q]];
		ndcg = 0;
		order_perm = new id_and_value[count[q]];
		int *perm_q = &perm[start[q]];
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = perm_q[i];
			order_perm[i].value = label[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		ideal_dcg[0] = pow(2.0,order_perm[0].value) - 1;
		idcg_yahoo = pow(2.0, order_perm[0].value) - 1;
		for (i=1;i<count[q];i++)
			ideal_dcg[i] = ideal_dcg[i-1] + (pow(2.0,order_perm[i].value) - 1) * log(2.0) / log(i+1.0);
		for (i=1;i<ndcg_size;i++)
			idcg_yahoo += (pow(2.0,order_perm[i].value) - 1) * log(2.0) / log(i+2.0);
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = perm_q[i];
			order_perm[i].value = target[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		dcg[0] = pow(2.0, label[order_perm[0].id]) - 1;
		dcg_yahoo = pow(2.0, label[order_perm[0].id]) - 1;
		for (i=1;i<count[q];i++)
			dcg[i] = dcg[i-1] + (pow(2.0, label[order_perm[i].id]) - 1) * log(2.0) / log(i + 1.0);
		for (i=1;i<ndcg_size;i++)
			dcg_yahoo += (pow(2.0, label[order_perm[i].id]) - 1) * log(2.0) / log(i + 2.0);
		if (ideal_dcg[0]>0)
			for (i=0;i<count[q];i++)
				ndcg += dcg[i]/ideal_dcg[i];
		else
			ndcg = 0;
		meanndcg += ndcg/count[q];
		delete[] order_perm;
		delete[] ideal_dcg;
		delete[] dcg;
		if (idcg_yahoo > 0)
			ndcg_yahoo += dcg_yahoo/idcg_yahoo;
		else
			ndcg_yahoo += 1;
	}
	meanndcg /= nr_query;
	ndcg_yahoo /= nr_query;
	result_ret[1] = meanndcg;
	result_ret[2] = ndcg_yahoo;
	free(start);
	free(count);
	free(perm);
}

int mpi_get_rank()
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return rank;	
}

int mpi_get_size()
{
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	return size;	
}

void mpi_exit(const int status)
{
	MPI_Finalize();
	exit(status);
}
