#include "csvparser.h"
#include <locale>
#include <clocale>


static struct SharedState 
{
	int max_seq_len_ = 0;
	double avg_seq_len_ = 0.0;
	uint64_t current_training_example_idx_ = 0;
	uint32_t current_training_file_idx_ = 0;
	uint64_t current_record_idx_ = 0; // record idx within a training file
	uint64_t current_token_count_ = 0; // record idx within a training file
};

struct CosmopediaDataset : torch::data::datasets::Dataset<CosmopediaDataset>
{
	CosmopediaDataset(const char* training_set_folder, int max_context_len, int batch_size, void* tokenizer) : resource_mutex_(std::make_shared<std::mutex>()), shared_state_(std::make_shared<SharedState>())
	{
		bool csv_open;
		int i;

		int32_t bytes_read;
		num_training_examples_ = 0;

		training_file_names_ = GetTrainingFileNames(training_set_folder, &num_training_files_);
		printf("number of files: %d\n", num_training_files_);

		tokenizer_ = tokenizer;

		//TODO: get these from the tokenizer
#ifdef USE_PYTHON_TOKENIZER
		BOS_ = 1; // <|im_start|>
		EOS_ = 2; // <|im_end|>
		PAD_ = EOS_;
		vocabular_size_ = 49152;
#else
		BOS_ = ::GetVocabularySize(tokenizer_);
		EOS_ = BOS_ + 1;
		PAD_ = EOS_;
		vocabular_size_ = PAD_ + 1;
#endif
		csv_reader_ = new CSVReader;

		
		buffer_ = new char[scratch_buffer_size];
		scratch1_ = new char[scratch_buffer_size];
		scratch2_ = new char[scratch_buffer_size];


		num_training_examples_ = 228133637; // pre-counted!
		//-------------------------------------------------------------------------------------------------
		/*
		//
		// calculate number to training examples (takes a very long time!)
		//
		int max_len = 0;
		num_training_examples_ = 0;
		for (i = 0; i < num_training_files_; i++)
		{
			csv_open = csv_reader_->Init(training_file_names_[i], "text");
			if (csv_open)
			{
				while (true)
				{
					bytes_read = csv_reader_->ReadNextRecord(buffer_, 1024 * 1024, scratch1_, scratch_buffer_size, scratch2_, scratch_buffer_size);
					if (bytes_read <= 0)
					{
						if (bytes_read < 0)
						{
							SpinForEver("Error reading training set file");
						}
						else
						{
							printf("%d num_training_examples_: %d\n", i, num_training_examples_, max_len);
							break;
						}
					}
					num_training_examples_++;
					if (bytes_read > max_len) max_len = bytes_read;
				}
			}
			else
			{
				SpinForEver("Unable to open training set file");
			}
		}
		*/
		//-------------------------------------------------------------------------------------------------
		printf("number of training examples: %d\n", num_training_examples_);

		if (!csv_reader_->Init(training_file_names_[shared_state_->current_training_file_idx_], "text"))
		{
			SpinForEver("Error initializing training set file");
		}
		max_context_len_ = max_context_len;
	}

	//current_training_file_idx_: 3, idx: 2213541, Avg seq len = 677.509666, Max seq len = 4070
	torch::data::Example<> get(size_t index) override
	{
		uint32_t* tokens_ptr;
		uint32_t* attn_mask_ptr;
		int64_t* target_ptr;

		uint32_t len;
		uint32_t char_len;
		uint32_t pad_len;
		uint32_t i;
		uint32_t j;
		int ret;
		int32_t bytes_read;
		uint32_t PAD;

		torch::Tensor contig_tokens;
		torch::Tensor contig_target;

		torch::Tensor tokens;
		torch::Tensor target;

		wchar_t w_scratch_buffer[10000];

		static thread_local bool initialized = false; // Thread-local flag
		if (!initialized) 
		{
			if (!::setlocale(LC_ALL, "en_US.UTF-8"))
			{
				SpinForEver("Failed to set UTF-8 locale.");
			}
			initialized = true;
		}


		tokens = torch::empty({ max_context_len_ + 1,  max_context_len_ }, torch::kInt); // don't know how to return 3 tensors, so returning tokens and mask together, first row are the tokens, bottom 'square' is attention mask
		target = torch::empty({ max_context_len_ }, torch::kLong);

		contig_tokens = tokens.contiguous();
		contig_target = target.contiguous();

		tokens_ptr = (uint32_t*)contig_tokens.data_ptr();
		attn_mask_ptr = tokens_ptr + max_context_len_;
		target_ptr = (int64_t*)contig_target.data_ptr();


		while (true)
		{			
			resource_mutex_->lock();
			bytes_read = csv_reader_->ReadNextRecord(buffer_, scratch_buffer_size, scratch1_, scratch_buffer_size, scratch2_, scratch_buffer_size);
			if (bytes_read <= 0)
			{
				if (bytes_read < 0)
				{
					SpinForEver("ReadNextRecord error!");
				}
				else
				{
					shared_state_->current_training_file_idx_++;
					shared_state_->current_record_idx_ = 0;
					shared_state_->current_training_file_idx_ = shared_state_->current_training_file_idx_ % num_training_files_;
					if (!csv_reader_->Init(training_file_names_[shared_state_->current_training_file_idx_], "text"))
					{
						SpinForEver("Error initializing training set file");
					}
					resource_mutex_->unlock();
					continue;
				}
			}
			resource_mutex_->unlock();

			char_len = strlen(buffer_);
			if (char_len < 10)
			{
				continue;
			}
			
			tokens_ptr[0] = BOS_;
			len = max_context_len_ - 1;

#ifdef USE_PYTHON_TOKENIZER
			ret = PythonTokenizerEncode(tokenizer_, buffer_, &tokens_ptr[1], &len);
#else
			ret = Encode(tokenizer_, buffer_, &tokens_ptr[1], &len, w_scratch_buffer, sizeof(w_scratch_buffer) / sizeof(wchar_t));
#endif
			if (ret || (len > char_len))
			{
				continue;
			}
			else
			{
				resource_mutex_->lock();
				
				if (len > shared_state_->max_seq_len_) shared_state_->max_seq_len_ = len;
				shared_state_->avg_seq_len_ += len;
				shared_state_->current_training_example_idx_++;
				shared_state_->current_record_idx_++;
				shared_state_->current_token_count_ += len;
				//std::cout << buffer_ << "\n";
				//printf("current_training_file_idx_: %d, idx: %d, Avg seq len = %f, Max seq len = %d\n", (int)shared_state_->current_training_file_idx_, (int)shared_state_->current_training_example_idx_, shared_state_->avg_seq_len_ / shared_state_->current_training_example_idx_, shared_state_->max_seq_len_);
				
				resource_mutex_->unlock();

				break;
			}			
		}


		for (i = 0; i < len; i++)
		{
			target_ptr[i] = tokens_ptr[i + 1];
		}
		target_ptr[len] = EOS_;

		if (PAD_ == EOS_ || PAD_ == BOS_)
		{
			PAD = vocabular_size_;
		}
		else
		{
			PAD = PAD_; // use dedicated value if it exists
		}
		

		len++; // accomodate BOS_/EOS_
		pad_len = max_context_len_ - len;
		for (i = 0; i < pad_len; i++)
		{
			tokens_ptr[len + i] = PAD_; // ok to use this value even if the same as some other special token since it will be ignored during loss calculation
			target_ptr[len + i] = PAD; // set this to the same value used as ignore_index in the loss function!
		}

		//
		// generate self attention mask
		//
		for (i = 0; i < max_context_len_; i++)
		{
			for (j = 0; j < max_context_len_; j++)
			{
				if (i < j)
				{
					attn_mask_ptr[i * max_context_len_ + j] = 0; // look ahead masking
				}
				else
				{
					if (target_ptr[j] == PAD)
					{
						attn_mask_ptr[i * max_context_len_ + j] = 0;
					}
					else
					{
						attn_mask_ptr[i * max_context_len_ + j] = 1;
					}
				}
			}
		}


		return { tokens, target };
	}

	torch::optional<size_t> size() const override
	{
		return num_training_examples_;
	}

	/*
	uint32_t GetVocabularySize()
	{
		return vocabular_size_;
	}*/

	void get_state(uint32_t* file_index, uint64_t* record_index, uint64_t* current_training_example_idx, uint64_t* current_token_count)
	{
		resource_mutex_->lock();
		if(file_index) *file_index = shared_state_->current_training_file_idx_;
		if(record_index) *record_index = shared_state_->current_record_idx_;
		if (current_training_example_idx) *current_training_example_idx = shared_state_->current_training_example_idx_;
		if(current_token_count) *current_token_count = shared_state_->current_token_count_;
		resource_mutex_->unlock();
	}


	int set_state(uint32_t file_index, uint64_t record_index, uint64_t current_training_example_idx, uint64_t current_token_count)
	{
		uint64_t i;
		int32_t bytes_read;
		int ret;

		printf("Setting dataloader state\n");
		resource_mutex_->lock();
		shared_state_->current_training_file_idx_ = file_index;
		shared_state_->current_token_count_ = current_token_count;
		shared_state_->current_training_example_idx_ = current_training_example_idx;

		csv_reader_->Init(training_file_names_[shared_state_->current_training_file_idx_], "text");
		for (i = 0; i < record_index; i++)
		{
			bytes_read = csv_reader_->ReadNextRecord(buffer_, scratch_buffer_size, scratch1_, scratch_buffer_size, scratch2_, scratch_buffer_size);
			if (bytes_read < 0)
			{
				ret = -1;
				goto Exit;
			}
		}

		printf("Dataloader state set succesfully\n");
		ret = 0;
	Exit:
		resource_mutex_->unlock();
		return ret;
	}




private:
	char** training_file_names_;
	uint32_t num_training_files_;
	uint64_t num_training_examples_;
	
	CSVReader* csv_reader_;
	void* tokenizer_;
	uint32_t BOS_;
	uint32_t EOS_;
	uint32_t PAD_;
	uint32_t vocabular_size_;
	uint32_t max_context_len_;
	char* buffer_;
	char* scratch1_;
	char* scratch2_;
	const uint32_t scratch_buffer_size = 1024 * 1024;
	
	
	std::shared_ptr<std::mutex> resource_mutex_;
	std::shared_ptr<SharedState> shared_state_;
	
};


