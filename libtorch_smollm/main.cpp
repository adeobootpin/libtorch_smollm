#include <torch/torch.h>
#include "c10\cuda\CUDAFunctions.h"
#include <torch/nn/modules/conv.h>
#include <fstream>
#include <windows.h>

torch::DeviceType device_type = torch::kCUDA;
//torch::DeviceType device_type = torch::kCPU;
torch::Device device(device_type, 0);

void SpinForEver(const char* pszMessage);
void* BlockRealloc(void* current_block_ptr, uint64_t current_size, uint64_t new_size);
char** GetTrainingFileNames(const char* training_set_folder, uint32_t* num_files);
void PrintETA(double nseconds_latest_iteration, uint32_t remaining_iterations);
int GenerateVersionedFilename(const char* basePath, char* versioned_file_name, int buffer_size);
int LoadPretrainedWeightsToGPU(char* filename, void* parameter_buffer, int size);

#define USE_PYTHON_TOKENIZER
#define USE_COSMOPEDIA_DATASET

#include "bootpin_tokenizer.h"
#include "python_tokenizer.h"
#ifdef USE_COSMOPEDIA_DATASET
#include "cosmopedia_v2_dataset.h"
#else
#include "webinstruct_dataset.h"
#endif
#include"learning_rate_schedule.h"
#include "llama.h"
#include "python_tokenizer.h"

int main()
{
	/*
	torch::Tensor attention_mask = torch::tensor({ 
		                               {0, 1, 1, 1},
								       {0, 0, 1, 1},
									   {0, 0, 0, 1},
									   {0, 0, 0, 0}},
									torch::dtype(torch::kInt));

	std::cout << attention_mask << "\n";
	attention_mask = attention_mask.to(torch::kBool);
	std::cout << attention_mask << "\n";
	
	torch::Tensor mask = attention_mask.to(torch::kFloat);

	mask.masked_fill_(attention_mask, -std::numeric_limits<float>::infinity());
	std::cout << mask << "\n";
	*/

	int batch_size = 8;
	int worker_threads;
	uint32_t num_epochs = 1;

	int max_seq_len = 1024;
	int num_layers;
	int embed_dim;
	int vocab_size;
	int num_heads;
	int head_dim;
	int mlp_intermediate_dim;
	int num_key_value_heads;
	float drop_out_rate;
	void* tokenizer;
	uint64_t num_training_examples;
	int gradient_accumulation_steps;
	uint64_t num_examples_seen = 0;
	uint64_t num_examples_seen_since_restart = 0; // need this so that ETA is calculated correctly
	uint64_t total_iterations = 0;
	float loss_val = 0;
	uint64_t avg_index = 0;
	char filename[256];

	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;
	double nseconds_total = 0;
	double lr;

	int32_t BOS;
	int32_t EOS;
	int32_t PAD;

	HMODULE hMod = LoadLibraryA("torch_cuda.dll");

	num_layers = 30;
	embed_dim = 576;
	num_heads = 9;
	head_dim = 64;
	num_key_value_heads = 3;
	mlp_intermediate_dim = 1536;
	drop_out_rate = 0.1f;


#ifdef USE_PYTHON_TOKENIZER
	tokenizer = InitializePythonTokenizer();
	vocab_size = GetPythonTokenizerVocabularySize(tokenizer);
	PAD = vocab_size;
	worker_threads = 0;
#else
	tokenizer = InitializeTokenizer("c:\\src\\bootpin_tokenizer\\data\\smollm_tokenizer_12_29_2024_4_57.bin");
	vocab_size = GetVocabularySize(tokenizer);
	BOS = vocab_size;
	EOS = BOS + 1;
	PAD = EOS + 1;
	vocab_size = PAD + 1;
	worker_threads = batch_size;
#endif

	/*
	unsigned int encoded_text[1000];
	unsigned int encoded_text_len = 1000;
	PythonTokenizerEncode(tokenizer, "This is the way I usually take to Abinci! Some like it hot, some like it cold, some like it in the pot, 9 days old", encoded_text, &encoded_text_len);
	PythonTokenizerDecode(tokenizer, encoded_text, encoded_text_len, nullptr, nullptr);
	*/

	printf("vocabular_size: %d\n", vocab_size);

#ifdef USE_COSMOPEDIA_DATASET
	auto dataset = CosmopediaDataset("d:\\smollm\\training\\", max_seq_len, batch_size, tokenizer);
#else
	auto dataset = WebInstructDataset("d:\\smollm\\fine_tuning\\", max_seq_len, batch_size, tokenizer);
#endif
	auto mapped_dataset = dataset.map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader(mapped_dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(worker_threads));

	//auto dataset = CustomDataset("d:\\smollm\\training\\", max_seq_len, batch_size, tokenizer).map(torch::data::transforms::Stack<>());
	//auto data_loader = torch::data::make_data_loader(dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(batch_size));

	auto net = std::make_shared<LLaMA>(num_layers, embed_dim, vocab_size, num_heads, head_dim, mlp_intermediate_dim, num_key_value_heads, drop_out_rate);

	num_training_examples = dataset.size().value();

	torch::optim::AdamWOptions opt_adamw(3e-3);
	opt_adamw.weight_decay(5 * 1e-2);
	LinearLearningRateSchedule lrs(2e-3, 3e-3, num_training_examples, num_epochs);
	//LinearLearningRateSchedule lrs(2e-4, 3e-4, num_training_examples, num_epochs);
	torch::optim::AdamW optimizer(net->parameters(), opt_adamw);
	
	
	//--------------------------------------------------------------
	// Restart
	//--------------------------------------------------------------
	total_iterations = 5750000;
	num_examples_seen = 25400000;
	//dataset.set_state(23, 122175, num_examples_seen, 0);
	dataset.set_state(57, 0, num_examples_seen, 0);
	torch::load(net, "c:\\src\\libtorch_smollm\\training\\libtorch_smollm_56_174371_5750000_25400000_tok_2643082549_1_29_2025_11_44.bin", device);
	//--------------------------------------------------------------

	net->to(device);

	/*
	int ret;
	char param_name[256];
	char* ch;

	for (const auto& pair : net->named_parameters())
	{
		auto param = pair.value();

		std::cout << "Parameter name: " << pair.key() << std::endl;
		strcpy_s(param_name, sizeof(param_name), pair.key().c_str());

		ch = param_name + strlen(param_name);
		while (true)
		{
			if (*ch == '-')
			{
				*ch = '.';
			}
			ch--;
			if (ch == param_name)
			{
				break;
			}
		}



		if ((pair.key().find("layers-29") != std::string::npos)||
			(pair.key().find("layers-28") != std::string::npos)||
			(pair.key().find("layers-27") != std::string::npos) || 
			(pair.key().find("layers-26") != std::string::npos))
		{
			std::cout << "****Exluded param: " << pair.key() << std::endl;
			param.set_requires_grad(true); // Exclude this parameter from training
			continue; // Skip parameters from "layers-29"
		}
		else
		{
			param.set_requires_grad(false); // Exclude this parameter from training
		}

		strcat_s(param_name, sizeof(param_name), ".bin");
		ret = LoadPretrainedWeightsToGPU(param_name, param.contiguous().data_ptr(), param.numel() * sizeof(float));
		if (ret)
		{
			//SpinForEver("Error loading weights\n");
			printf("%s failed to load\n", param_name);
		}

	}
	*/

	//======================================================================================
	/*
	printf("\n\n----------------------------------------\n\n");
	for (const auto& pair : net->named_parameters())
	{
		auto param = pair.value();

		if (pair.key() == "model-embed_tokens.weight" || pair.key() == "lm_head.weight")
		{
			param.set_requires_grad(true); // Include this parameter in training
			std::cout << "Emabling " << pair.key() << " for training." << std::endl;
		}
		else
		{
			param.set_requires_grad(false); // Exclude this parameter from training
			std::cout << "Excluding " << pair.key() << " from training." << std::endl;

			if (param.grad().defined())
			{
				param.mutable_grad().reset();
			}
			
		}
	}
	*/
	//======================================================================================


	//======================================================================================
	/*
	std::vector<int> end_conditions = { 128001, 128009 };
	char* output = new char[5000];
	int output_len = 5000;

	net->train(false);

	torch::Tensor test_prompt = torch::tensor({ 1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411, 407, 19712, 8182, 2, 198, 1, 4093, 198, 10576, 314, 48102, 47, 2, 198 }, torch::kInt);
	test_prompt = test_prompt.to(device);
	net->generate(test_prompt, end_conditions, output, output_len, nullptr);
	*/
	//======================================================================================


	//======================================================================================
	/*
	char* output = new char[5000];
	int output_len = 5000;

	//torch::load(net, "c:\\src\\libtorch_smollm\\data\\smollm_huggingface_original.bin");
	torch::load(net, "c:\\src\\libtorch_smollm\\training\\libtorch_smollm_0_200000_50000_200000_1_3_2025_20_1.bin");
	net->to(device);


	unsigned int encoded_text[500];
	unsigned int encoded_text_len = 500;
	const char* text = "Who is Chopin?";

#ifdef USE_PYTHON_TOKENIZER
	PythonTokenizerEncode(tokenizer, text, encoded_text, &encoded_text_len);
	std::vector<int> end_conditions = { 0, 2 };  // <|endoftext|> , <|im_end|> in smolllm, TODO: get this from the tokenizer
#else
	Encode(tokenizer, text, encoded_text, &encoded_text_len);
	std::vector<int> end_conditions = { 128001, 128009 };
#endif
	net->train(false);
	torch::Tensor test_prompt = torch::from_blob(encoded_text, { encoded_text_len }, torch::dtype(torch::kInt));
	test_prompt = test_prompt.to(device);
	net->generate(test_prompt, end_conditions, output, output_len, tokenizer);
	*/
	//======================================================================================


	net->train(true);

	for (const auto& pair : net->named_parameters())
	{
		auto param = pair.value();
		//std::cout << pair.key() << " [" << param.sizes() << "]" << std::endl;
	}

	int64_t param_count = 0;
	for (const auto& param : net->parameters())
	{
		param_count += param.numel();
	}
	printf("Number of parameters: %lld\n", param_count - (49152 * 576)); // remove size of lm_head_->weight (which is tied to embed_tokens_->weight)

	auto criterion = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(PAD));


	gradient_accumulation_steps = 1024 / batch_size;

	torch::Tensor x;
	torch::Tensor target;
	torch::Tensor self_attn_mask;
	torch::Tensor logits;
	torch::Tensor probs;
	torch::Tensor loss;
	torch::Tensor tokens_and_mask;

	int64_t warmpup_steps = 10;

	for (uint32_t epoch = 0; epoch < num_epochs; epoch++)
	{
		for (auto& batch : *data_loader)
		{
			clock_begin = std::chrono::steady_clock::now();

			tokens_and_mask = batch.data;
			x = tokens_and_mask.index({ torch::indexing::Slice(), 0, torch::indexing::Slice() }).contiguous();
			self_attn_mask = tokens_and_mask.index({ torch::indexing::Slice(), torch::indexing::Slice(1, max_seq_len + 1), torch::indexing::Slice() }).contiguous();
			self_attn_mask = self_attn_mask.to(torch::kBool);

			x = x.to(device);
			self_attn_mask = self_attn_mask.to(device).unsqueeze(1);
			target = batch.target.to(device);

			//std::cout << x.sizes() << "\n";

			x = net->embed_tokens(x);
			logits = net->forward(x, self_attn_mask);

			probs = torch::nn::functional::log_softmax(logits, 2);

			loss = criterion(probs.view({ probs.sizes()[0] * probs.sizes()[1], probs.sizes()[2] }), target.view({ target.sizes()[0] * target.sizes()[1] }));
			loss = loss * (1.0f / gradient_accumulation_steps);

			loss.backward();

			num_examples_seen += batch_size;
			num_examples_seen_since_restart += batch_size;
			total_iterations++;


			if (!(total_iterations % gradient_accumulation_steps))
			{
				uint64_t tokens_seen;
				float floss = loss.item<float>() * gradient_accumulation_steps;
				loss_val += floss;
				lr = lrs.GetLearningRate(num_examples_seen - 1);
				
				if (warmpup_steps > 0)
				{
					lr /= warmpup_steps;
					warmpup_steps--;
				}
				
				dataset.get_state(nullptr, nullptr, nullptr, &tokens_seen);
				printf("loss: %f avg loss: %f [epoch: %d iteration: %llu / %llu tokens: %llu lr: %f] <%llu>\n", floss, loss_val / (avg_index + 1), epoch, num_examples_seen, num_training_examples * num_epochs, tokens_seen, lr, total_iterations);
				avg_index++;


				for (const auto& param_group : optimizer.param_groups())
				{
					((torch::optim::AdamWOptions&)param_group.options()).lr(lr);
				}

				/*
				for (auto& param_group : optimizer.param_groups())
				{
					param_group.options().set_lr(lr);
				}
				*/
				torch::nn::utils::clip_grad_norm_(net->parameters(), 1.0f);

				optimizer.step();
				optimizer.zero_grad();

				PrintETA(nseconds_total / num_examples_seen_since_restart, num_training_examples * num_epochs - num_examples_seen);
				printf("\n");
			}

			if (!(total_iterations % 50000))
			//if (!(total_iterations % 100))
			{
				uint32_t file_index;
				uint64_t record_index;
				uint64_t tokens_seen;
				char versioned_file_name[256];

				dataset.get_state(&file_index, &record_index, nullptr, &tokens_seen);
				
				sprintf_s(filename, sizeof(filename), "c:\\src\\libtorch_smollm\\training\\libtorch_smollm_%d_%llu_%llu_%llu_tok_%llu.bin", file_index, record_index, total_iterations, num_examples_seen, tokens_seen);

				GenerateVersionedFilename(filename, versioned_file_name, sizeof(versioned_file_name));

				printf("versioned_file_name: %s\n", versioned_file_name);
				torch::save(net, versioned_file_name);
			}

			clock_end = std::chrono::steady_clock::now();
			time_span = clock_end - clock_begin;
			nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
			nseconds_total += nseconds;
		}
	}

	return 0;
}


void SpinForEver(const char* pszMessage)
{
	while (true)
	{
		printf("\r\n%s", pszMessage);
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
}


void* BlockRealloc(void* current_block_ptr, uint64_t current_size, uint64_t new_size)
{
	unsigned char* reallocated_block_ptr;

	reallocated_block_ptr = new unsigned char[new_size];

	memcpy(reallocated_block_ptr, current_block_ptr, current_size);

	delete current_block_ptr;

	return reallocated_block_ptr;
}



int ReadDataFromFile(const char* pszFileName, void** ppvData, int* pcbDataSize)
{
	int iRet;
	HANDLE hFile = INVALID_HANDLE_VALUE;
	DWORD dwBytes;
	DWORD dwBytesRead;
	DWORD dwFileSize;
	char* pchData;

	hFile = CreateFileA(pszFileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);

	if (hFile == INVALID_HANDLE_VALUE)
	{
		iRet = GetLastError();
		goto Exit;
	}

	dwFileSize = GetFileSize(hFile, 0);
	pchData = new char[dwFileSize];
	if (!pchData)
	{
		assert(0);
		iRet = -1;
		goto Exit;
	}

	*ppvData = (void*)pchData;

	dwBytesRead = 0;

	while (dwBytesRead < dwFileSize)
	{
		ReadFile(hFile, pchData, dwFileSize, &dwBytes, NULL);
		dwBytesRead += dwBytes;
		pchData += dwBytes;
	}

	*pcbDataSize = dwBytesRead;

	CloseHandle(hFile);

	iRet = 0;

Exit:
	return iRet;
}

int LoadPretrainedWeights(char* filename, void* parameter_buffer, int size)
{
	char path[256];
	int ret;
	void* data;
	int file_size;

	sprintf_s(path, sizeof(path), "d:\\smollm\\huggingface_wts\\%s", filename);
	ret = ReadDataFromFile(path, &data, &file_size);

	if (ret)
	{
		printf("ReadDataFromFile failed!\n");
		return ret;
	}

	if (file_size != size)
	{
		printf("Wrong file size\n");
		return -1;
	}

	memcpy(parameter_buffer, data, size);

	return 0;
}


int CopyDataToGPU(void* gpu, void* host, size_t size)
{
	cudaMemcpy(gpu, host, size, cudaMemcpyHostToDevice);
	return 0;
}


int LoadPretrainedWeightsToGPU(char* filename, void* parameter_buffer, int size)
{
	char path[256];
	int ret;
	void* data;
	int file_size;

	sprintf_s(path, sizeof(path), "d:\\smollm\\huggingface_wts\\%s", filename);
	ret = ReadDataFromFile(path, &data, &file_size);

	if (ret)
	{
		printf("ReadDataFromFile failed!\n");
		return ret;
	}

	if (file_size != size)
	{
		printf("Wrong file size\n");
		return -1;
	}

	CopyDataToGPU(parameter_buffer, data, size);

	return 0;
}
