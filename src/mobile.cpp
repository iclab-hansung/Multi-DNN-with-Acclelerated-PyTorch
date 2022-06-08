#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>
#include <time.h>

#include "mobile.h"

namespace F = torch::nn::functional;


void get_submodule_mobilenet(torch::jit::script::Module module, Net &net){
	Layer t_layer;
	for(auto child : module.named_children()){
		t_layer.layer = child.value;
		for(auto ch : child.value.named_children()){
			int idx = net.layers.size();
			if(child.name == "features"){
				t_layer.layer = ch.value;
				if(ch.name == "0" || ch.name =="18"){	// ConvBNReLU
					for(auto ch2 : ch.value.named_children()){
						t_layer.layer = ch2.value;
						if(ch2.name == "0") t_layer.name = "conv";
						else if(ch2.name == "1") t_layer.name = "bn";
						else if(ch2.name =="2") t_layer.name = "relu6";
						//std::cout << "layer name : " << t_layer.name <<"\n";
						net.layers.push_back(t_layer);
					}
				}
				else if(ch.name == "1"){	//invertedResidual-1
					for(auto ch2 : ch.value.named_children()){
						if(ch2.name == "conv"){
							for(auto ch3 : ch2.value.named_children()){
								t_layer.layer = ch3.value;
								if(ch3.name =="0"){	// ConvBNReLU
									for(auto ch4 : ch3.value.named_children()){
										t_layer.layer = ch4.value;
										if(ch4.name =="0") t_layer.name = "conv";
										else if(ch4.name =="1") t_layer.name = "bn";
										else if(ch4.name =="2") t_layer.name = "relu6";
										//std::cout << "layer name : " << t_layer.name <<"\n";
										net.layers.push_back(t_layer);
									}
								}
								else if(ch3.name =="1"){
									t_layer.name = "conv";
									//std::cout << "layer name : " << t_layer.name <<"\n";
									net.layers.push_back(t_layer);
									
								}
								else if(ch3.name =="2"){
									t_layer.name = "bn";
									//std::cout << "layer name : " << t_layer.name <<"\n";
									net.layers.push_back(t_layer);
									
								}
								
							}
						}
					}
				}
				else if(ch.name == "3" || ch.name == "5" || ch.name == "6" || ch.name == "8" || ch.name == "9" || \
						ch.name == "10" || ch.name == "12" || ch.name == "13" || ch.name == "15" || ch.name == "16"){	//invertedResidual 2~17
					for(auto ch2 : ch.value.named_children()){
						t_layer.layer = ch2.value;
						if(ch2.name == "conv"){
							for(auto ch3 : ch2.value.named_children()){
								t_layer.layer = ch3.value;

								if(ch3.name =="0" || ch3.name == "1"){	// ConvBNReLU
									for(auto ch4 : ch3.value.named_children()){
										t_layer.layer = ch4.value;
										if(ch4.name =="0" || ch4.name == "1") t_layer.name = "conv";
										else if(ch4.name =="1") t_layer.name = "bn";
										else if(ch4.name =="2") t_layer.name = "relu6";
										//std::cout << "layer name : " << t_layer.name <<"\n";
										net.layers.push_back(t_layer);
									}
								}
								else if(ch3.name =="2"){
									t_layer.name = "conv";
									//std::cout << "layer name : " << t_layer.name <<"\n";
									net.layers.push_back(t_layer);
								}
								else if(ch3.name =="3"){
									t_layer.name = "bn";
									//std::cout << "layer name : " << t_layer.name <<"\n";
									net.layers.push_back(t_layer);
								}
								
							}
						}
					}
					net.layers.back().name = "last_use_res_connect";
					net.layers[idx].name = "first_use_res_connect";
					//std::cout << "layer name : " << t_layer.name <<"\n";
				}

				else {	//invertedResidual 2~17
					for(auto ch2 : ch.value.named_children()){
						t_layer.layer = ch2.value;
						if(ch2.name == "conv"){
							for(auto ch3 : ch2.value.named_children()){
								t_layer.layer = ch3.value;
								if(ch3.name =="0" || ch3.name == "1"){	// ConvBNReLU
									for(auto ch4 : ch3.value.named_children()){
										t_layer.layer = ch4.value;
										if(ch4.name =="0" || ch4.name == "1") t_layer.name = "conv";
										else if(ch4.name =="1") t_layer.name = "bn";
										else if(ch4.name =="2") t_layer.name = "relu6";
										//std::cout << "layer name : " << t_layer.name <<"\n";
										net.layers.push_back(t_layer);
									}
								}
								else if(ch3.name =="2"){
									t_layer.name = "conv";
									//std::cout << "layer name : " << t_layer.name <<"\n";
									net.layers.push_back(t_layer);
								}
								else if(ch3.name =="3"){
									t_layer.name = "bn";
									//std::cout << "layer name : " << t_layer.name <<"\n";
									net.layers.push_back(t_layer);
								}
								
							}
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
				//std::cout << "layer name : " << t_layer.name <<"\n";
				net.layers.push_back(t_layer); // child
			}
		}
	}
}


void *predict_mobilenet(Net *mobile){
	std::vector<torch::jit::IValue> inputs = mobile->input;
	int i;
	float time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
	for(i=0;i<mobile->layers.size();i++){
		pthread_mutex_lock(&mutex_t[mobile->index_n]);
		cond_i[mobile->index_n] = 1;
		
		netlayer nl;
		nl.net = mobile;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;


		thpool_add_work(thpool,(void(*)(void *))forward_mobilenet,(void*) &th);
		while (cond_i[mobile->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[mobile->index_n], &mutex_t[mobile->index_n]);
    	}
		mobile->input.clear();
		mobile->input.push_back(mobile->layers[i].output);
		pthread_mutex_unlock(&mutex_t[mobile->index_n]);
	
	}
	cudaStreamSynchronize(streams[mobile->H_L][(mobile->index_s)%n_streamPerPool]);
	cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

	std::cout << "\n*****"<<mobile->name<<" result*****" << "     Mobilenet exe time >>> " << time/1000 << "'s" <<std::endl;
	std::cout << "index num = "<< mobile->index_n << "	priority num = "<< mobile->priority << std::endl;
	std::cout << "Stream [" << mobile->H_L << "][" << (mobile->index_s)%n_streamPerPool <<"]" << std::endl;
	//std::cout << (mobile->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	std::cout << " " << std::endl;
	}

void forward_mobilenet(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::vector<torch::jit::IValue> inputs = nl->net->input;
	int k = nl->net->index;
	at::Tensor out;
	
	if(nl->net->layers[k].name == "first_use_res_connect"){
		nl->net->identity = inputs[0].toTensor();
	}
	{
		at::cuda::CUDAStreamGuard guard(streams[nl->net->H_L][(nl->net->index_s)%n_streamPerPool]); // high, low
		// ***임시로 flatten 변수 사용***
		if(k == nl->net->flatten){	// nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
			//std::cout <<"k : "<< k << "   layer name : " << nl->net->layers[k].name << "\n";
			out = torch::nn::functional::adaptive_avg_pool2d(inputs[0].toTensor(), \
			F::AdaptiveAvgPool2dFuncOptions(1)).reshape({inputs[0].toTensor().size(0), -1});
		}

		/*
		InvertedResidual forward function

		def forward(self, x: Tensor) -> Tensor:
			if self.use_res_connect:
				return x + self.conv(x)
			else:
				return self.conv(x)
		*/
		else if(nl->net->layers[k].name == "last_use_res_connect"){
			//std::cout <<"k : "<< k << "   layer name : " << nl->net->layers[k].name << "\n";
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
			out = nl->net->identity + out;
		}
		else{
			//std::cout <<"k : "<< k << "   layer name : " << nl->net->layers[k].name << "\n";
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
			if(k+1 < nl->net->layers.size() && nl->net->layers[k+1].name == "bn" && nl->net->layers[k+2].name == "relu6" )
				for(int j=0;j<2;j++){
					nl->net->layers[k].output = out;
					k++;
					inputs.clear();
					inputs.push_back(out);
					out = nl->net->layers[k].layer.forward(inputs).toTensor();
				}
		}
	}

	nl->net->layers[k].output = out;
	cond_i[nl->net->index_n]=0;
	nl->net->index = k;
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
}

