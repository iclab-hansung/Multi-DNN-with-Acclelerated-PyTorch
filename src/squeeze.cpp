// squeeze.net에서는 activation layer의 relu연산을 thread pool로 보내지 않고 처리합니다.

#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>
#include "cuda_runtime.h"
#include <time.h>
#include "squeeze.h"
#include <math.h>


namespace F = torch::nn::functional;
using namespace std;

bool is_ReLu(int idx){
	if(idx == 1 || idx == 3 || idx == 5)  // ReLu index
		return true;
	return false;
}


void get_submodule_squeeze(torch::jit::script::Module module, Net &net){
	Dummy concat;
	Layer t_layer;
	for(auto ch : module.named_children()){
		if(ch.name == "features"){
			for(auto ch2 : ch.value.named_children()){
				if(ch2.name == "0"){
					t_layer.layer = ch2.value;
					t_layer.exe_success = false;
					t_layer.input_idx = 0;
					t_layer.name = "conv";
					//std::cout << t_layer.name << "\n";
					net.layers.push_back(t_layer);
				}	
				else if(ch2.name == "1"){
					t_layer.layer = ch2.value;
					t_layer.exe_success = false;
					t_layer.input_idx = 0;
					t_layer.name = "relu";
					//std::cout << t_layer.name << "\n";
					net.layers.push_back(t_layer);
				}	
				else if(ch2.name == "2" || ch2.name == "6" || ch2.name == "11"){
					t_layer.layer = ch2.value;
					t_layer.exe_success = false;
					t_layer.input_idx = 0;
					t_layer.name = "maxpool";
					//std::cout << t_layer.name << "\n";
					net.layers.push_back(t_layer);
				}	
				else{	// Fire
					int index = 0;
					for(auto ch3 : ch2.value.named_children()){
						t_layer.layer = ch3.value;
						if(is_ReLu(index)){  // squeeze,1x1,3x3 activation
							index++;
							t_layer.name = "activation";
							continue;
						}
						else if(ch3.name == "expand3x3"){
							t_layer.input_idx = -2;
						}
						else{	// squeeze, expand1x1, expand3x3
							t_layer.input_idx = 0;
						}

						t_layer.name = ch3.name; // expand1x1 , expand3x3 , squeeze
						t_layer.layer = ch3.value;
						t_layer.exe_success = false;
						//std::cout << t_layer.name << "\n";
						net.layers.push_back(t_layer);

						if(ch3.name == "expand3x3"){	// 3x3밑에 concat 만드는 부분
							t_layer.input_idx = 0;
							t_layer.layer = concat;
							t_layer.from_idx = {-2, -1};
							t_layer.name = "concat";
							//std::cout << t_layer.name << "\n";
							net.layers.push_back(t_layer);
						}
						index++;
					}
				}
			}

		}
		else if(ch.name == "classifier"){
			for(auto ch2 : ch.value.named_children()){
				t_layer.layer = ch2.value;
				t_layer.exe_success = false;
				t_layer.input_idx = 0;
				if(ch2.name == "0")	t_layer.name = "dropout";
				else if(ch2.name == "1")	t_layer.name = "conv";
				else if(ch2.name == "2")	t_layer.name = "relu";
				else if(ch2.name == "3")	t_layer.name = "avgpool";
				//std::cout << t_layer.name << "\n";
				net.layers.push_back(t_layer);
			}
		}
	}
}

void *predict_squeeze(Net *squeeze){
	int i=0;
	float time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

	std::vector<int> stream_id = {(squeeze->index_s)%n_streamPerPool, abs(squeeze->index_b)%n_streamPerPool};
	for(i=0;i<squeeze->layers.size();i++){

		pthread_mutex_lock(&mutex_t[squeeze->index_n]);
		cond_i[squeeze->index_n] = 1;
		squeeze->layers[i].exe_success = false;

		netlayer nl;
		nl.net = squeeze;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;

		thpool_add_work(thpool,(void(*)(void *))forward_squeeze,&th);

		while (cond_i[squeeze->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[squeeze->index_n], &mutex_t[squeeze->index_n]);
    	}
		i = nl.net->index;
		squeeze->input.clear();
		squeeze->input.push_back(squeeze->layers[i].output);
		pthread_mutex_unlock(&mutex_t[squeeze->index_n]);
		cudaStreamSynchronize(streams[squeeze->H_L][stream_id[0]]);	
	}
	
	cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
	std::cout << "\n*****"<<squeeze->name<<" result*****" << "     Squeezenet exe time >>> " << time/1000 << "'s" <<std::endl;
	std::cout << "index num = "<< squeeze->index_n << "	priority num = "<< squeeze->priority<< std::endl;
	std::cout << "Stream [" << squeeze->H_L << "][" << stream_id[0] << "]" << "Branch Stream ["<<stream_id[1]<<"]" << std::endl;
	//std::cout<<(squeeze->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) <<"\n";
	printf("\n");
}

void forward_squeeze(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::vector<torch::jit::IValue> inputs;
	int k = nl->net->index;
	int n_all = nl->net->n_all;
	int success_check_idx; // expand1x1 = k+1, expand3x3 = k-1
	std::vector<int> stream_id = {(nl->net->index_s)%n_streamPerPool, abs(nl->net->index_b)%n_streamPerPool};

	if(nl->net->layers[k].name == "expand3x3"){ 
		int input_idx = k + nl->net->layers[k].input_idx;
		inputs.push_back(nl->net->layers[input_idx].output);
	}
	else{
		inputs = nl->net->input;
	}

	if(nl->net->layers[k].name == "expand1x1"){
		cond_i[nl->net->index_n]=0;
		pthread_cond_signal(&cond_t[nl->net->index_n]);
	}
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);  
	
	at::Tensor out;
	{
		at::cuda::CUDAStreamGuard guard(streams[nl->net->H_L][stream_id[0]]); // high, low
		
		if(k == nl->net->flatten){
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
			out = out.view({out.size(0), -1});
		}
		else if(nl->net->layers[k].name == "concat"){
			std::vector<at::Tensor> cat_input;
			for(int i=0;i<nl->net->layers[k].from_idx.size();i++){
				int concat_idx = k + nl->net->layers[k].from_idx[i];
				cat_input.push_back(nl->net->layers[concat_idx].output);
			}
			out = torch::cat(cat_input, 1);
		}
		else if(nl->net->layers[k].name == "squeeze" || nl->net->layers[k].name == "expand1x1" || nl->net->layers[k].name == "expand3x3"){ // expand1x1, expand3x3, squeeze_activation 을 여기서 처리해줌
			if(nl->net->layers[k].name == "expand1x1"){
				{
					at::cuda::CUDAStreamGuard guard(streams[nl->net->H_L][stream_id[0]]);
					success_check_idx=k+1;					
					out = nl->net->layers[k].layer.forward(inputs).toTensor();
					out = torch::relu(out); // expand1x1_activation
					cudaEventRecord(nl->net->record[0],streams[nl->net->H_L][stream_id[0]]);
				}
			}
			else if(nl->net->layers[k].name == "expand3x3"){
				{	
					at::cuda::CUDAStreamGuard guard(streams[nl->net->H_L][stream_id[1]]); 
					success_check_idx=k-1;
					out = nl->net->layers[k].layer.forward(inputs).toTensor();
					out = torch::relu(out);	// expand3x3_activation
					cudaEventRecord(nl->net->record[1],streams[nl->net->H_L][stream_id[1]]);
				}
			}
			else{ // squeeze
				out = nl->net->layers[k].layer.forward(inputs).toTensor();
				out = torch::relu(out);	// squeeze_activation
			}
		}
		else{
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
			
			if(k+1 < nl->net->layers.size() && nl->net->layers[k+1].name == "relu"){
				nl->net->layers[k].output = out;
				k++; 
				inputs.clear();
				inputs.push_back(out);
				out = nl->net->layers[k].layer.forward(inputs).toTensor();
				nl->net->index = k;
			}			
		}
	}

	if(nl->net->layers[k].name == "expand1x1"){
		cudaEventSynchronize(nl->net->record[0]);
		nl->net->layers[k].exe_success = true;
	}
	else if(nl->net->layers[k].name == "expand3x3"){
		cudaEventSynchronize(nl->net->record[1]);
		nl->net->layers[k].exe_success = true;
	}
	
	nl->net->layers[k].output = out;

	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	if(nl->net->layers[k].name != "expand1x1" && nl->net->layers[k].name != "expand3x3"){
		cond_i[nl->net->index_n]=0;
		pthread_cond_signal(&cond_t[nl->net->index_n]);
	}else if(nl->net->layers[success_check_idx].exe_success){
		cond_i[nl->net->index_n]=0;
		pthread_cond_signal(&cond_t[nl->net->index_n]);
	}
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
}
