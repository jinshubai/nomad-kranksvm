#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include "rksvm.h"
#include "scheduler.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
int print_null(const char *s,...) {}

static int (*info)(const char *fmt,...) = &printf;

struct rksvm_node *x;
int max_nr_attr = 64;
struct rksvm_node *gx_space;//

struct rksvm_model *rksvm_load_model(const char *model_file_name);
struct rksvm_model* model;

static char *line = NULL;
static int max_line_len;

static const char *rksvm_type_table[] =
{
	"l2r_rank",NULL
};

static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

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

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	mpi_exit(1);
}

void predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
	double *dvec_t;
	double *ivec_t;
	int *query;
	
	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
		total++;
	rewind(input);
	dvec_t = new double[total];
	ivec_t = new double[total];
	query = new int[total];
	total = 0;
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		query[total] = 0;

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);
		ivec_t[total] = target_label;

		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct rksvm_node *) realloc(x,max_nr_attr*sizeof(struct rksvm_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			if (!strcmp(idx,"qid"))
			{
				errno = 0;
				query[total] = (int) strtol(val, &endptr,10);
				if(endptr == val || errno !=0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(i+1);
			}
			else
			{
				errno = 0;
				x[i].index = (int) strtol(idx,&endptr,10);
				if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
					exit_input_error(total+1);
				else
					inst_max_index = x[i].index;

				errno = 0;
				x[i].value = strtod(val,&endptr);
				if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(total+1);

				++i;
			}
		}
		x[i].index = -1;
		predict_label = rksvm_predict(model,x);// at the beginning of this file
		dvec_t[total] = predict_label;
		fprintf(output,"%g\n",predict_label);

		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}

	/************************/
	int g_total = total;
	int size = mpi_get_size();
	int *send_counts;
	send_counts = new int[size];
	MPI_Allgather((void*)&total, 1, MPI_INT, (void*)send_counts, 1, MPI_INT, MPI_COMM_WORLD);

	int *recv_counts;
	recv_counts = new int[size];
	int *recv_displs;
	recv_displs = new int[size];

	for(int j=0;j<size;j++)
	{
		recv_displs[j] = 0;
		for(int k=0;k<j;k++)
		{
			recv_displs[j] += send_counts[k];
		}
		recv_counts[j] = send_counts[j]; 
	}

	mpi_allreduce(&g_total, 1, MPI_INT, MPI_SUM);
	double *g_ivec_t;
	double *g_dvec_t;
	int *g_query;
	g_ivec_t = new double[g_total];
	g_dvec_t = new double[g_total];
	g_query = new int[g_total];

	MPI_Allgatherv((void*)ivec_t, total, MPI_DOUBLE, (void*)g_ivec_t, recv_counts, recv_displs, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Allgatherv((void*)dvec_t, total, MPI_DOUBLE, (void*)g_dvec_t, recv_counts, recv_displs, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Allgatherv((void*)query, total, MPI_INT, (void*)g_query, recv_counts, recv_displs, MPI_INT, MPI_COMM_WORLD);
	/*******************/
	
	double result[3];
	//eval_list(ivec_t,dvec_t,query,total,result);
	eval_list(g_ivec_t,g_dvec_t,g_query,g_total,result);
	if(mpi_get_rank()==0)
	{
		info("Pairwise Accuracy = %g%%\n",result[0]*100);
		info("MeanNDCG (LETOR) = %g\n",result[1]);
		info("NDCG (YAHOO) = %g\n",result[2]);
	}

	delete ivec_t;
	delete dvec_t;
	delete query;
	delete g_ivec_t;
	delete g_dvec_t;
	delete g_query;
	delete recv_counts;
	delete recv_displs;
	delete send_counts;
}

void exit_with_help()
{
	printf(
			"Usage: predict [options] test_file model_file output_file\n"
			"options:\n"
	"-q : quiet mode (no outputs)\n"
	);
	mpi_exit(1);
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	FILE *input, *output;
	int i;
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'q':
				info = &print_null;
				i--;
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	if(i>=argc-2)
		exit_with_help();

	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"[rank %d] can't open input file %s\n", mpi_get_rank(), argv[i]);
		mpi_exit(1);
	}

	output = fopen(argv[i+2],"w");
	if(output == NULL)
	{
		fprintf(stderr,"[rank %d] can't open output file %s\n", mpi_get_rank(),argv[i+2]);
		mpi_exit(1);
	}
	if((model=rksvm_load_model(argv[i+1]))==0)
	{
		fprintf(stderr,"[rank %d] can't open model file %s\n", mpi_get_rank(), argv[i+1]);
		mpi_exit(1);
	}

	x = (struct rksvm_node *) malloc(max_nr_attr*sizeof(struct rksvm_node));
	predict(input,output);
	rksvm_free_and_destroy_model(&model);
	free(x);
	free(line);
	free(gx_space);//what we added in here.
	fclose(input);
	fclose(output);

	MPI_Finalize();
	return 0;
}

rksvm_model *rksvm_load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL) return NULL;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	// read parameters

	rksvm_model *lmodel = Malloc(rksvm_model,1);
	rksvm_model *gmodel = Malloc(rksvm_model,1);
	//rksvm_parameter& param = lmodel->param;
	rksvm_parameter& param = gmodel->param;

	lmodel->rho = NULL;
	gmodel->rho = NULL;
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
				free(lmodel->rho);
				free(lmodel);
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
				free(lmodel->rho);
				free(lmodel);
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
			fscanf(fp,"%d",&lmodel->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			fscanf(fp,"%d",&lmodel->l);
		else if(strcmp(cmd,"rho")==0)
		{
			int n = lmodel->nr_class * (lmodel->nr_class-1)/2;
			lmodel->rho = Malloc(double,n);
			gmodel->rho = Malloc(double,n);//
			gmodel->nr_class = lmodel->nr_class;//
			for(int i=0;i<n;i++)
			{
				fscanf(fp,"%lf",&lmodel->rho[i]);
				gmodel->rho[i] = lmodel->rho[i];//
			}
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
			free(lmodel->rho);
			free(lmodel);
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
	elements += lmodel->l;

	fseek(fp,pos,SEEK_SET);

	int m = lmodel->nr_class - 1;
	int l = lmodel->l;
	lmodel->sv_coef = Malloc(double *,m);
	int i;
	for(i=0;i<m;i++)
		lmodel->sv_coef[i] = Malloc(double,l);//it need to be freed, we will do it later
	lmodel->SV = Malloc(rksvm_node*,l);
	rksvm_node *x_space = NULL;
	if(l>0) x_space = Malloc(rksvm_node,elements);

	//here, please take care
	int *sv_length = new int[l];//one
	int nr_ranks = mpi_get_size();//
	int current_rank = mpi_get_rank();//
	int s=0;//
	int j=0;
	for(i=0;i<l;i++)
	{
		readline(fp);
		lmodel->SV[i] = &x_space[j];

		p = strtok(line, " \t");
		lmodel->sv_coef[0][i] = strtod(p,&endptr);
		for(int k=1;k<m;k++)
		{
			p = strtok(NULL, " \t");
			lmodel->sv_coef[k][i] = strtod(p,&endptr);
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
		sv_length[i] = j-s;//
		s = j;//
	}

	free(line);
	//let's build the global model
	//first
	int *local_l = new int[nr_ranks];//two
	MPI_Allgather((void*)&l, 1, MPI_INT, (void*)local_l, 1, MPI_INT, MPI_COMM_WORLD); 
	//second
	int *l_start_pos = new int[nr_ranks];//three
	l_start_pos[0] = 0;
	for(int i=1;i<nr_ranks;i++)
		l_start_pos[i] = l_start_pos[i-1]+local_l[i-1];
	//thrid
	int g_l = l;
	mpi_allreduce(&g_l,1, MPI_INT, MPI_SUM);
	gmodel->l=g_l;
	int *global_sv_length = new int[g_l];//four
	MPI_Allgatherv((void*)sv_length, l, MPI_INT, (void*)global_sv_length, local_l, l_start_pos, MPI_INT, MPI_COMM_WORLD);	
	//fourth
	int g_elements = elements;
	int *each_elements = new int[nr_ranks];//five
	mpi_allreduce(&g_elements,1, MPI_INT, MPI_SUM); 
	MPI_Allgather((void*)&elements, 1, MPI_INT, (void*)each_elements, 1, MPI_INT, MPI_COMM_WORLD);

	//build the bcast messages 
	//all coef + all SVs
	const int msg_bytenum = l*m*sizeof(double)+elements*sizeof(rksvm_node);
	int *size_of_each_msg = new int[nr_ranks];//seven
	MPI_Allgather((void*)&msg_bytenum, 1, MPI_INT, (void*)size_of_each_msg, 1, MPI_INT, MPI_COMM_WORLD);
	char **recvd_msg =  sallocator<char*>().allocate(nr_ranks);//eight
	for(i=0;i<nr_ranks;i++)
	{
		recvd_msg[i] = sallocator<char>().allocate(size_of_each_msg[i]);//nine
	}
	
	char *cur_pos = recvd_msg[current_rank];
	for(i=0;i<m;i++)
	{
		double *dest_coef = reinterpret_cast<double *>(cur_pos);
		std::copy(lmodel->sv_coef[i], lmodel->sv_coef[i] + l, dest_coef);
		cur_pos += l*sizeof(double);
	}
	rksvm_node *dest_sv = reinterpret_cast<rksvm_node *>(cur_pos);
	std::copy(x_space, x_space + elements, dest_sv);

	for(i=0;i<nr_ranks;i++)
	{
		int success = MPI_Bcast(recvd_msg[i], size_of_each_msg[i], MPI_CHAR, i, MPI_COMM_WORLD);
		if (success != MPI_SUCCESS)
		{
			printf("The Bcast errors happen to rank %d\n",i);
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
	}

	//we should rebuild the received messages. At the same time, the 
	//gmodel should be built via these messages.
	gmodel->SV = Malloc(rksvm_node*,g_l);//ten
	gx_space = Malloc(rksvm_node, g_elements);//eleven
	gmodel->sv_coef = Malloc(double*, m);//twelve
	for(i=0;i<m;i++)
	{
		gmodel->sv_coef[i] = Malloc(double,g_l);//thirteen
	}

	rksvm_node *agx_space = gx_space;

	for(int rankid=0;rankid<nr_ranks;rankid++)
	{
		char *recvd_pos = recvd_msg[rankid];
		for(i=0;i<m;i++)
		{
			double *dest_coef_recv = reinterpret_cast<double *>(recvd_pos);
			memcpy(gmodel->sv_coef[i]+l_start_pos[rankid], dest_coef_recv, (size_t)sizeof(double)*local_l[rankid]);
			recvd_pos += local_l[rankid]*sizeof(double);
		}
		rksvm_node *dest_sv_recv = reinterpret_cast<rksvm_node *>(recvd_pos);
		memcpy(agx_space, dest_sv_recv, (size_t)sizeof(rksvm_node)*each_elements[rankid]);

		//int j=0;
		int interval = 0;
		for(i=0;i<local_l[rankid];i++)
		{
			gmodel->SV[i+l_start_pos[rankid]] = &agx_space[interval];
			interval += global_sv_length[i+l_start_pos[rankid]];
			//gmodel->SV[i+l_start_pos[rankid]] = &agx_space[j];
			//while(true)
			//{
			//	if(agx_space[j].index==-1)
			//	{
			//		j++;
			//		break;
			//	}
			//	else
			//	{
			//		j++;
			//	}
			//}
		}
		agx_space += each_elements[rankid];
	}

/*for(int rankid=0;rankid<nr_ranks;rankid++)
	{
for(i=0;i<local_l[rankid];i++)
{
if(current_rank==0)
{
printf("(%d)coef=%f\n",i+l_start_pos[rankid],gmodel->sv_coef[0][i+l_start_pos[rankid]]);
}
}
}*/
//MPI_Barrier(MPI_COMM_WORLD);
//mpi_exit(1);

/*for(int rankid=1;rankid<2;rankid++)
	{
for(i=0;i<local_l[rankid];i++)
{
	rksvm_node *s = gmodel->SV[i+l_start_pos[rankid]];
	while(true)
	{
		if(s->index==-1)
		{
if(current_rank==0)
{
			printf("\n");
}
			break;
		}
		else
		{
if(current_rank==0)
{
			printf("()%d:%f ",s->index,s->value);
}
			s++;
		}
	}
}
}*/
//mpi_exit(1);
/*int z=0;
for(i=0;i<l;i++)
{
while(z<elements)
	{
		if(gx_space[z].index==-1)
		{
			printf("\n");
		z++;
			break;
		}
		else
		{
			printf("%d:%f ",gx_space[z].index,gx_space[z].value);
		z++;
		}

	}
}*/

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	//free the memory we have assigned
	delete[] sv_length;
	delete[] local_l;
	delete[] l_start_pos;
	delete[] global_sv_length;
	delete[] each_elements;
	for(i=0;i<nr_ranks;i++)
		sallocator<char>().deallocate(recvd_msg[i],size_of_each_msg[i]);
	sallocator<char*>().deallocate(recvd_msg,nr_ranks);
	delete[] size_of_each_msg;
	//free the sub-model
	free(x_space);
	free(lmodel->rho);
	for(i=0;i<m;i++)
		free(lmodel->sv_coef[i]);
	free(lmodel->sv_coef);
	free(lmodel->SV);
	free(lmodel);
	return gmodel;
}