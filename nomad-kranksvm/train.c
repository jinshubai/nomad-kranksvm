#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <sched.h>
#include <mpi.h>
#include <omp.h>
#include "rksvm.h"
#include "scheduler.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-k kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_set_file)\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-t thread count: set the number of threads (default 4)\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"//(It needs to be care.)
	"-q : quiet mode (no outputs)\n"
	);
	mpi_exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"[rank %d] Wrong input format at line %d\n", mpi_get_rank(), line_num);
	mpi_exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);

struct rksvm_parameter param;		// set by parse_command_line
struct rksvm_problem prob;		// set by read_problem
struct rksvm_model *model;
struct rksvm_node *x_space;

static char *line = NULL;
static int max_line_len;


int main(int argc, char **argv)
{
	//set the mpi settings
	int threadprovided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadprovided);
	if(threadprovided != MPI_THREAD_MULTIPLE)
	{
		printf("MPI multiple thread isn't provided!\n");
		fflush(stdout);
		mpi_exit(1);
	}
	int current_rank = mpi_get_rank();
	int nr_ranks = mpi_get_size();
	param.nr_ranks = nr_ranks;
	
	char hostname[1024];
	int hostname_len;
	MPI_Get_processor_name(hostname, &hostname_len);
    printf("processor name: %s, number of processed: %d, rank: %d\n", hostname, nr_ranks, current_rank);
	fflush(stdout);
	//
	int global_l;
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;
	parse_command_line(argc, argv, input_file_name, model_file_name);
	
	//set the number of threads for the shared-memory system
	int nr_threads = param.thread_count;
	int max_thread_count = omp_get_max_threads();

	if(nr_threads > max_thread_count)
	{
		printf("[rank %d], please enter the correct number of threads: 1~%d\n", current_rank, max_thread_count);
		mpi_exit(1);
	}
	omp_set_num_threads(nr_threads);

	//set the cpu affnity
	/*int ithread, err, cpu;
	cpu_set_t cpu_mask;
#pragma omp parallel private(ithread, cpu_mask, err, cpu)
	{
		ithread = omp_get_thread_num();
		CPU_ZERO(&cpu_mask);//set mask to zero
		CPU_SET(ithread, &cpu_mask);//set mask with ithread
		err = sched_setaffinity((pid_t)0, sizeof(cpu_mask), &cpu_mask);
		cpu = sched_getcpu();
		printf("thread_id %d on CPU %d\n", ithread, cpu);
	}*/
	//now, read the problem from the input file
	read_problem(input_file_name);
	error_msg = rksvm_check_parameter(&prob,&param);

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		mpi_exit(1);
	}

	//distributed code
	global_l = prob.l;
	mpi_allreduce(&global_l, 1, MPI_INT, MPI_SUM);//MPI_INT :int;MPI_SUM:sum
	prob.global_l = global_l;
	
	printf("#local instances = %d, #global instances = %d\n", prob.l, prob.global_l);
	fflush(stdout);

	if(current_rank==0){
	puts("Start to train!");
	}
	model = rksvm_train(&prob,&param);
	if(rksvm_save_model(model_file_name,model))
	{
		fprintf(stderr,"[rank %d] can't save model to file %s\n",mpi_get_rank(), model_file_name);
		mpi_exit(1);
	}
	rksvm_free_and_destroy_model(&model);
	free(prob.y);
	free(prob.x);
	free(prob.query);
	free(x_space);
	free(prob.length_of_each_rksvm_node);
	free(line);

	MPI_Finalize();
	return 0;
}


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.rksvm_type = L2R_RANK;
	param.kernel_type = RBF;
	param.degree = 3;
	param.thread_count = 4;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	//param.cache_size = 100;
	param.C = 1.0;
	param.eps = 1e-3;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'k':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 't':
				param.thread_count = atof(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			//case 'm':
				//param.cache_size = atof(argv[i]);// the default value is 100 MB
				//break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	rksvm_set_print_string_function(print_func);

	// determine filenames

	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"/home/jing/model/%s.model",p);
	}
}

// read in a problem (in svmlight format)
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

void read_problem(const char *filename)
{
	long int elements, max_index, inst_max_index, i, j, k;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		mpi_exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct rksvm_node *,prob.l);
	prob.query = Malloc(int, prob.l);
	x_space = Malloc(struct rksvm_node,elements);
	prob.length_of_each_rksvm_node = Malloc(int, prob.l);

	max_index = 0;
	j=0;
	k=0;
	for(i=0;i<prob.l;i++)
	{
		prob.query[i] = 0;
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			if (!strcmp(idx,"qid"))
			{
				errno = 0;
				prob.query[i] = (int) strtol(val, &endptr,10);
				if(endptr == val || errno !=0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(i+1);
			}
			else
			{

				errno = 0;
				x_space[j].index = (int) strtol(idx,&endptr,10);
				if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
					exit_input_error(i+1);
				else
					inst_max_index = x_space[j].index;

				errno = 0;
				x_space[j].value = strtod(val,&endptr);
				if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(i+1);

				++j;
			}
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
		prob.length_of_each_rksvm_node[i] = (int)(j-k);
		 k = j;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"[rank %d] Wrong input format: first column must be 0:sample_serial_number\n", mpi_get_rank());
				mpi_exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"[rank %d] Wrong input format: sample_serial_number out of range\n", mpi_get_rank());
				mpi_exit(1);
			}
		}

	fclose(fp);
}
