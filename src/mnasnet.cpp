#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>
#include "cuda_runtime.h"
#include "mnasnet.h"
#include <time.h>

namespace F = torch::nn::functional;
using namespace std;


void get_submodule_MNASNet(torch::jit::script::Module module, Net &net){
	Layer t_layer;
	Dummy concat;
	for(auto child : module.named_children()){
		
		for(auto ch : child.value.named_children()){ // layers classifier
			
			if(child.name == "layers"){
				
				if(ch.name == "0" || ch.name == "3" || ch.name == "6" || ch.name =="14"){
					
					t_layer.layer = ch.value;
					t_layer.name = "conv";
					//std::cout << t_layer.name <<"\n";
					net.layers.push_back(t_layer);
				}
				else if(ch.name == "1" || ch.name == "4" || ch.name == "7" || ch.name == "15"){
					
					t_layer.layer = ch.value;
					t_layer.name = "bn";
					//std::cout  << t_layer.name <<"\n";
					net.layers.push_back(t_layer);
				}
				else if(ch.name == "2" || ch.name == "5" || ch.name == "16"){
					
					t_layer.layer = ch.value;
					t_layer.name = "relu";
					//std::cout << t_layer.name <<"\n";
					net.layers.push_back(t_layer);
				}
				// sequential
				else if(ch.name == "8" || ch.name == "9" || ch.name == "10" || ch.name == "11" || ch.name == "12" || ch.name == "13"){
					//t_layer.layer = ch.value;
					for(auto ch2 : ch.value.named_children()){		// invertedResidual
						for(auto ch3 : ch2.value.named_children()){ 	// layers
							if(ch3.name == "layers"){
								
								for(auto ch4 : ch3.value.named_children()){ //layers submodule(0~7)
									t_layer.name = "";
									t_layer.layer = ch4.value;

									if(ch4.name == "0" || ch4.name == "3" || ch4.name == "6" ){
										t_layer.name = "conv";
									}
									else if(ch4.name == "1" || ch4.name == "4" || ch4.name == "7"){
										t_layer.name = "bn";
									}
									else if(ch4.name == "2" || ch4.name == "5"){
										t_layer.name = "relu";
									}
									//std::cout << t_layer.name <<"\n";
									net.layers.push_back(t_layer); // ch4
								}
							}
						}
						if(ch2.name != "0"){
							t_layer.layer = concat;
							t_layer.name = "Residual";
							t_layer.from_idx = {-1, -9};
							//std::cout << t_layer.name <<"\n";
							net.layers.push_back(t_layer);
						}
						else{
							t_layer.name ="";
						}
					}
				}
			}
			
			else if(child.name == "classifier"){
				t_layer.layer = ch.value;
				if(ch.name == "0"){
					t_layer.name = "dropout";
				}
				else if(ch.name == "1"){
					t_layer.name = "linear";
				}
				//std::cout << t_layer.name <<"\n";
				net.layers.push_back(t_layer); 
			}
		}
	}
}


void *predict_MNASNet(Net *mnasnet){
	int i;
	float time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
	for(i=0;i<mnasnet->layers.size();i++){
		pthread_mutex_lock(&mutex_t[mnasnet->index_n]);
		cond_i[mnasnet->index_n] = 1;
		
		netlayer nl;
		nl.net = mnasnet;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;

		thpool_add_work(thpool,(void(*)(void *))forward_MNASNet,(void*) &th);
		//std::cout << "thpool_add_work END" << "\n";

		while (cond_i[mnasnet->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[mnasnet->index_n], &mutex_t[mnasnet->index_n]);
    	}
		mnasnet->input.clear();
		mnasnet->input.push_back(mnasnet->layers[i].output);
		pthread_mutex_unlock(&mutex_t[mnasnet->index_n]);
		//std::cout << i <<"\n";
		
	}
	cudaStreamSynchronize(streams[mnasnet->H_L][(mnasnet->index_s)%n_streamPerPool]);
	cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

	std::cout << "\n*****"<<mnasnet->name<<" result*****" << "     MNASnet exe time >>> " << time/1000 << "'s" <<std::endl;
	std::cout << "index_n num = "<< mnasnet->index_n << "	priority num = "<< mnasnet->priority << std::endl;
	std::cout << "Stream [" << mnasnet->H_L << "][" << (mnasnet->index_s)%n_streamPerPool <<"]" << std::endl;
	//std::cout << (mnasnet->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	std::cout << " " << std::endl;
	}

void forward_MNASNet(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::vector<torch::jit::IValue> inputs = nl->net->input;
	int k = nl->net->index;
	at::Tensor out;
	{
		at::cuda::CUDAStreamGuard guard(streams[nl->net->H_L][(nl->net->index_s)%n_streamPerPool]); // high, low
		if(k == nl->net->flatten){
			out = inputs[0].toTensor().mean({2,3});
		}
		else if(nl->net->layers[k].name == "Residual")
		{
			int add_index = k + nl->net->layers[k].from_idx[0];
			out = nl->net->layers[add_index].output;
			for(int i=1;i<nl->net->layers[k].from_idx.size();i++){
				int add_index = k + nl->net->layers[k].from_idx[i];
				out += nl->net->layers[add_index].output;
			}
		}
		else{
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
			if(k+1<nl->net->layers.size() && 
				nl->net->layers[k+1].name == "bn" && nl->net->layers[k+2].name == "relu"){
				for(int j=0;j<2;j++){
					nl->net->layers[k].output = out;
					k++;
					inputs.clear();
					inputs.push_back(out);
					out = nl->net->layers[k].layer.forward(inputs).toTensor();
				}
			}
		}
	}
	nl->net->layers[k].output = out;
	nl->net->index = k;
	cond_i[nl->net->index_n]=0;
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
}

