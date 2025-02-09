class feature_cache
{
public:
	// Members
	int64_t _seen_tokens = 0;
	std::vector<torch::Tensor> key_cache;
	std::vector<torch::Tensor> value_cache;

	// Method
	std::tuple<torch::Tensor, torch::Tensor> update_cache(int64_t layer_idx, torch::Tensor key_states, torch::Tensor value_states)
	{
		// Update _seen_tokens if layer_idx is 0
		if (layer_idx == 0)
		{
			this->_seen_tokens += key_states.size(-2);
		}

		// Update the cache
		if (this->key_cache.size() <= layer_idx)
		{
			this->key_cache.push_back(key_states);
			this->value_cache.push_back(value_states);
		}
		else
		{
			this->key_cache[layer_idx] = torch::cat({ this->key_cache[layer_idx], key_states }, /*dim=*/-2);
			this->value_cache[layer_idx] = torch::cat({ this->value_cache[layer_idx], value_states }, /*dim=*/-2);
		}

		// Return the updated key and value cache tensors at the current layer index
		return std::make_tuple(this->key_cache[layer_idx], this->value_cache[layer_idx]);
	}

	void clear()
	{
		_seen_tokens = 0;
		key_cache.clear();
		value_cache.clear();
	}
};


struct LLaMABlock : public torch::nn::Cloneable<LLaMABlock>
{
	LLaMABlock(int hidden_dim, int num_heads, int head_dim, int mlp_intermediate_dim, int num_key_value_heads, float drop_out_rate, feature_cache* kv_cache)
	{
		int base = 500000;
		eps_ = 1e-05;

		rms_norm_weight_ = register_parameter("input_layernorm-weight", torch::rand({ hidden_dim }, device));
		post_attention_rms_norm_weight_ = register_parameter("post_attention_layernorm-weight", torch::rand({ hidden_dim }, device));

		//rms_norm_weight_ = register_parameter("input_layernorm-weight", torch::rand({ hidden_dim }));
		//post_attention_rms_norm_weight_ = register_parameter("post_attention_layernorm-weight", torch::rand({ hidden_dim }));


		q_proj_ = register_module("self_attn-q_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, hidden_dim).bias(false)));
		k_proj_ = register_module("self_attn-k_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, num_key_value_heads * head_dim).bias(false)));
		v_proj_ = register_module("self_attn-v_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, num_key_value_heads * head_dim).bias(false)));
		o_proj_ = register_module("self_attn-o_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, hidden_dim).bias(false)));


		gate_proj_ = register_module("mlp-gate_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, mlp_intermediate_dim).bias(false)));
		up_proj_ = register_module("mlp-up_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, mlp_intermediate_dim).bias(false)));
		down_proj_ = register_module("mlp-down_proj", torch::nn::Linear(torch::nn::LinearOptions(mlp_intermediate_dim, hidden_dim).bias(false)));


		auto arange_tensor = torch::arange(0, head_dim, 2, torch::kInt64).to(torch::kFloat);
		inv_freq_ = 1.0 / torch::pow(base, arange_tensor / head_dim);
		register_buffer("inv_freq_", inv_freq_).set_requires_grad(false);

		num_heads_ = num_heads;
		head_dim_ = head_dim;
		num_key_value_heads_ = num_key_value_heads;
		num_key_value_groups_ = num_heads_ / num_key_value_heads_; //Note: where did this come from?

		hidden_size_ = hidden_dim;
		drop_out_rate_ = drop_out_rate;
		kv_cache_ = kv_cache;
	}

	void reset() {};

	torch::Tensor RMS_Norm(torch::Tensor x)
	{
		auto variance = x.pow(2).mean(-1, /*keepdim=*/true);
		x = x * torch::rsqrt(variance + eps_);
		x = rms_norm_weight_ * x;

		return x;
	}

	torch::Tensor post_attention_RMS_Norm(torch::Tensor x)
	{
		auto variance = x.pow(2).mean(-1, /*keepdim=*/true);
		x = x * torch::rsqrt(variance + eps_);
		x = post_attention_rms_norm_weight_ * x;

		return x;
	}

	torch::Tensor mlp(torch::Tensor x)
	{
		x = torch::nn::functional::silu(gate_proj_(x)) * up_proj_(x);
		x = down_proj_(x);
		//int ret = Compare(x, "f:\\shared\\minicpm-v_wts\\debug.bin", 0.001f);
		return x;
	}



	std::tuple<torch::Tensor, torch::Tensor> rotary_emb(torch::Tensor position_ids)
	{
		auto inv_freq_expanded = inv_freq_.unsqueeze(0).unsqueeze(2).to(torch::kFloat).expand({ position_ids.size(0), -1, 1 });

		auto position_ids_expanded = position_ids.unsqueeze(1).to(torch::kFloat);

		auto freqs = torch::matmul(inv_freq_expanded.to(torch::kFloat), position_ids_expanded.to(torch::kFloat)).transpose(1, 2);
		auto emb = torch::cat({ freqs, freqs }, -1);

		auto cos = emb.cos();
		auto sin = emb.sin();

		return std::make_tuple(cos, sin);
	}


	torch::Tensor rotate_half(const torch::Tensor& x)
	{
		// Calculate the split index along the last dimension
		int64_t split_size = x.size(-1) / 2;

		// Use torch::indexing::Ellipsis and torch::indexing::Slice for slicing
		using torch::indexing::Slice;
		using torch::indexing::Ellipsis;
		using torch::indexing::None;

		// Slice x to get x1: x[..., :split_size]
		auto x1 = x.index({ Ellipsis, Slice(None, split_size) });

		// Slice x to get x2: x[..., split_size:]
		auto x2 = x.index({ Ellipsis, Slice(split_size, None) });

		// Concatenate -x2 and x1 along the last dimension
		return torch::cat({ -x2, x1 }, /*dim=*/-1);
	}

	std::tuple<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin, int64_t unsqueeze_dim = 1)
	{
		// Unsqueeze cos and sin
		cos = cos.unsqueeze(unsqueeze_dim);
		sin = sin.unsqueeze(unsqueeze_dim);

		// Apply rotary positional embedding to q and k
		auto q_embed = (q * cos) + (rotate_half(q) * sin);
		auto k_embed = (k * cos) + (rotate_half(k) * sin);

		// Return the embedded q and k tensors
		return std::make_tuple(q_embed, k_embed);
	}

	torch::Tensor repeat_kv(const torch::Tensor& hidden_states, int64_t n_rep)
	{
		// Get the dimensions of the hidden_states tensor
		auto sizes = hidden_states.sizes();
		int64_t batch = sizes[0];
		int64_t num_key_value_heads = sizes[1];
		int64_t slen = sizes[2];
		int64_t head_dim = sizes[3];

		if (n_rep == 1)
		{
			return hidden_states;
		}

		// Add a new dimension at position 2
		auto expanded_hidden_states = hidden_states.unsqueeze(2);

		// Expand the tensor to the desired shape
		expanded_hidden_states = expanded_hidden_states.expand({ batch, num_key_value_heads, n_rep, slen, head_dim });

		// Reshape the tensor to merge num_key_value_heads and n_rep
		return expanded_hidden_states.reshape({ batch, num_key_value_heads * n_rep, slen, head_dim });
	}


	torch::Tensor scaled_dot_product_attention(const torch::Tensor& query, torch::Tensor key, torch::Tensor value, torch::Tensor attention_mask, float dropout_p)
	{
		torch::Tensor mask;

		double scale_factor = 1.0 / std::sqrt(query.size(-1));


		mask = attention_mask.to(torch::kFloat);;
		mask.masked_fill_(attention_mask, -std::numeric_limits<float>::infinity());

		torch::Tensor attn_weight = torch::matmul(query, key.transpose(-2, -1)) * scale_factor;
		//std::cout << attn_weight.sizes() << "\n";
		//std::cout << mask.sizes() << "\n";

		attn_weight += mask;

		attn_weight = torch::softmax(attn_weight, -1);

		//bool is_training = torch::autograd::GradMode::is_enabled();
		if (is_training() && dropout_p > 0.0)
		{
			attn_weight = torch::dropout(attn_weight, dropout_p, /*train=*/true);
		}

		torch::Tensor output = torch::matmul(attn_weight, value);

		return output;
	}


	torch::Tensor scaled_dot_product_attention(const torch::Tensor& query, torch::Tensor key,	torch::Tensor value, const c10::optional<torch::Tensor>& attn_mask = c10::nullopt, double dropout_p = 0.0,	bool is_causal = true, c10::optional<double> scale = c10::nullopt, bool enable_gqa = false)
	{
		// Get sizes
		int64_t L = query.size(-2);
		int64_t S = key.size(-2);

		// Compute scale factor
		double scale_factor = scale.has_value() ? scale.value() : 1.0 / std::sqrt(query.size(-1));

		return query;

		// Create attn_bias tensor
		torch::Tensor attn_bias = torch::zeros({ L, S }, query.options());


		// Handle causal masking
		if (is_causal) 
		{
			TORCH_CHECK(!attn_mask.has_value(), "attn_mask should be None when is_causal is True");
			torch::Tensor temp_mask = torch::ones({ L, S }, torch::dtype(torch::kBool));
			temp_mask = temp_mask.tril(0);
			attn_bias.masked_fill_(~temp_mask, -std::numeric_limits<float>::infinity());
			attn_bias = attn_bias.to(query.dtype());
		}

		// Handle attn_mask
		if (attn_mask.has_value()) 
		{
			torch::Tensor mask = attn_mask.value();
			if (mask.dtype() == torch::kBool) 
			{
				attn_bias.masked_fill_(~mask, -std::numeric_limits<float>::infinity());
			}
			else 
			{
				attn_bias += mask;
			}
		}

		// Handle enable_gqa
		if (enable_gqa) 
		{
			int64_t query_heads = query.size(-3);
			int64_t key_heads = key.size(-3);
			int64_t repeat_times = query_heads / key_heads;
			key = key.repeat_interleave(repeat_times, /*dim=*/-3);
			value = value.repeat_interleave(repeat_times, /*dim=*/-3);
		}

		// Compute attention weights
		torch::Tensor attn_weight = torch::matmul(query, key.transpose(-2, -1)) * scale_factor;
		attn_weight += attn_bias;

		// Apply softmax
		attn_weight = torch::softmax(attn_weight, -1);

		// Apply dropout if training
		//bool is_training = torch::autograd::GradMode::is_enabled();
		if (is_training() && dropout_p > 0.0) 
		{
			attn_weight = torch::dropout(attn_weight, dropout_p, /*train=*/true);
		}

		// Compute attention output
		torch::Tensor output = torch::matmul(attn_weight, value);

		return output;
	}

	torch::Tensor attention(torch::Tensor hidden_states, torch::Tensor attention_mask, torch::Tensor position_ids, int layer_index)
	{
		torch::Tensor query_states;
		torch::Tensor key_states;
		torch::Tensor value_states;
		int bsz;
		int q_len;

		bsz = hidden_states.sizes()[0];
		q_len = hidden_states.sizes()[1];

		query_states = q_proj_(hidden_states);
		key_states = k_proj_(hidden_states);
		value_states = v_proj_(hidden_states);


		query_states = query_states.view({ bsz, q_len, num_heads_, head_dim_ }).transpose(1, 2);
		key_states = key_states.view({ bsz, q_len, num_key_value_heads_, head_dim_ }).transpose(1, 2);
		value_states = value_states.view({ bsz, q_len, num_key_value_heads_, head_dim_ }).transpose(1, 2);


		std::tuple<torch::Tensor, torch::Tensor> output = rotary_emb(position_ids);
		torch::Tensor cos = std::get<0>(output);
		torch::Tensor sin = std::get<1>(output);

		std::tuple<torch::Tensor, torch::Tensor> output2 = apply_rotary_pos_emb(query_states, key_states, cos, sin);
		query_states = std::get<0>(output2);
		key_states = std::get<1>(output2);


		//std::tuple<torch::Tensor, torch::Tensor> output3 = kv_cache_->update_cache(layer_index, key_states, value_states);
		//key_states = std::get<0>(output3);
		//value_states = std::get<1>(output3);
		//std::cout << key_states.sizes() << " : " << value_states.sizes() << "\n";

		key_states = repeat_kv(key_states, num_key_value_groups_);
		value_states = repeat_kv(value_states, num_key_value_groups_);


		//torch::Tensor attn_output = scaled_dot_product_attention(query_states, key_states, value_states);
		//torch::Tensor attn_output = scaled_dot_product_attention(query_states, key_states, value_states, attention_mask, drop_out_rate_);
		torch::Tensor attn_output = at::scaled_dot_product_attention(query_states, key_states, value_states, attention_mask, drop_out_rate_, false);


		attn_output = attn_output.transpose(1, 2).contiguous();
		attn_output = attn_output.view({ bsz, q_len, hidden_size_ });

		attn_output = o_proj_(attn_output);


		return attn_output;
	}

	torch::Tensor forward(torch::Tensor x, torch::Tensor attention_mask, int layer_index)
	{
		torch::Tensor residual;
		torch::Tensor position_ids = torch::arange(x.sizes()[1], torch::kInt64);

		/*
		try
		{
			std::cout << x.sizes() << "\n";
			std::cout << position_ids.sizes() << "\n";
			position_ids = position_ids.unsqueeze(0);
			std::cout << position_ids.sizes() << "\n";
			position_ids = position_ids.to(x.device());
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << '\n';
		}
		*/

		int xx = x.sizes()[1];
		position_ids = position_ids.unsqueeze(0).to(x.device());

		residual = x;
		x = RMS_Norm(x);

		x = attention(x, attention_mask, position_ids, layer_index);

		x = residual + x;

		residual = x;

		x = post_attention_RMS_Norm(x);

		x = mlp(x);

		x = residual + x;

		return x;
	}

	void to(const torch::Device device, bool non_blocking = false)
	{
		q_proj_->to(device);
		k_proj_->to(device);
		v_proj_->to(device);
		o_proj_->to(device);

		gate_proj_->to(device);
		up_proj_->to(device);
		down_proj_->to(device);

		rms_norm_weight_ = rms_norm_weight_.to(device);
		post_attention_rms_norm_weight_ = post_attention_rms_norm_weight_.to(device);
		inv_freq_ = inv_freq_.to(device);
	}

	void train(bool on)
	{
		q_proj_->train(on);
		k_proj_->train(on);
		v_proj_->train(on);
		o_proj_->train(on);

		gate_proj_->train(on);
		up_proj_->train(on);
		down_proj_->train(on);

		//rms_norm_weight_ = rms_norm_weight_.to(device);
		//post_attention_rms_norm_weight_ = post_attention_rms_norm_weight_.to(device);
		//inv_freq_ = inv_freq_.to(device);

		torch::nn::Module::train(on);
	}

private:
	float eps_;
	torch::Tensor rms_norm_weight_;
	torch::Tensor post_attention_rms_norm_weight_;

	torch::nn::Linear q_proj_{ nullptr };
	torch::nn::Linear k_proj_{ nullptr };
	torch::nn::Linear v_proj_{ nullptr };
	torch::nn::Linear o_proj_{ nullptr };


	torch::nn::Linear gate_proj_{ nullptr };
	torch::nn::Linear up_proj_{ nullptr };
	torch::nn::Linear down_proj_{ nullptr };

	torch::Tensor inv_freq_;

	int num_heads_;
	int head_dim_;
	int num_key_value_heads_;
	int num_key_value_groups_;
	int hidden_size_;
	float drop_out_rate_;

	feature_cache* kv_cache_;
};


struct LLaMA : public torch::nn::Cloneable<LLaMA>
{
	LLaMA(int num_levels, int embed_dim, int vocab_size, int num_heads, int head_dim, int mlp_intermediate_dim, int num_key_value_heads, float drop_out_rate)
	{
		char szName[256];
		int i;

		eps_ = 1e-05;
		num_levels_ = num_levels;
		vocab_size_ = vocab_size;
		embed_dim_ = embed_dim;

		attn_block = new std::shared_ptr<LLaMABlock>[num_levels_];
		for (i = 0; i < num_levels_; i++)
		{
			sprintf_s(szName, sizeof(szName), "model-layers-%d", i);
			attn_block[i] = register_module<LLaMABlock>(szName, std::make_shared <LLaMABlock>(embed_dim, num_heads, head_dim, mlp_intermediate_dim, num_key_value_heads, drop_out_rate, &kv_cache_));
		}

		norm_weight_ = register_parameter("model-norm-weight", torch::empty({ embed_dim }, device));
		//norm_weight_ = register_parameter("model-norm-weight", torch::empty({ embed_dim }));
		embed_tokens_ = register_module("model-embed_tokens", torch::nn::Embedding(vocab_size_, embed_dim));
		lm_head_ = register_module("lm_head", torch::nn::Linear(torch::nn::LinearOptions(embed_dim, vocab_size_).bias(false)));

		lm_head_->weight = embed_tokens_->weight;
	}

	void reset() {};

	torch::Tensor Norm(torch::Tensor x)
	{
		auto variance = x.pow(2).mean(-1, /*keepdim=*/true);
		x = x * torch::rsqrt(variance + eps_);
		x = norm_weight_ * x;

		return x;
	}


	torch::Tensor forward(torch::Tensor embeddings, torch::Tensor attn_mask)
	{
		int i;
		torch::Tensor hidden_states;


		hidden_states = embeddings;


		for (i = 0; i < num_levels_; i++)
		{
			hidden_states = attn_block[i]->forward(hidden_states, attn_mask, i);
		}

		hidden_states = Norm(hidden_states);

		hidden_states = lm_head_(hidden_states);

		return hidden_states;
	}


	int generate(torch::Tensor prompt, const std::vector<int>& end_conditions, char* output, int output_len, void* tokenizer)
	{
		torch::Tensor logits;
		int i;
		int idx;
		unsigned int encoded_text[500];


		torch::Tensor attn_mask;

		torch::Tensor prompt_embeddings;


		prompt_embeddings = embed_tokens(prompt);
		prompt_embeddings = prompt_embeddings.unsqueeze(0);


		torch::Tensor hidden_states = prompt_embeddings;
		torch::Tensor x = hidden_states;

		int num_elements = prompt.numel();

		bool shared = (lm_head_->weight.data_ptr() == embed_tokens_->weight.data_ptr());

		idx = 0;
		//std::cout << attn_mask << "\n";

		while (true)
		{
			kv_cache_.clear();

			torch::Tensor attention_mask = torch::ones({ num_elements, num_elements }, torch::kInt);
			attn_mask = attention_mask.triu(1).to(torch::kBool);
			attn_mask = attn_mask.to(hidden_states.device());

			logits = forward(hidden_states, attn_mask);

			logits = logits.to(torch::kCPU);

			float* test_data;
			test_data = (float*)logits.data_ptr();
			int seq_len = logits.numel() / vocab_size_;

			test_data += (seq_len - 1) * vocab_size_;

			float max = -9999999.0f;
			int max_index = 0;
			for (i = 0; i < vocab_size_; i++)
			{
				if (test_data[i] > max)
				{
					max = test_data[i];
					max_index = i;
				}
			}

			printf("%d, ", max_index);
			encoded_text[idx++] = max_index;

			for (int end_value : end_conditions)
			{
				if (max_index == end_value)
				{
					return 0; // Exit if max_index matches any end condition
				}
			}

			auto past_token = torch::tensor({ max_index }, torch::kLong).to(hidden_states.device());
			auto embedding_output = embed_tokens_->forward(past_token);
			embedding_output = embedding_output.reshape({ 1, 1, embed_dim_ });

			x = torch::cat({ x, embedding_output }, 1);
			hidden_states = x;
			num_elements++;

			if (idx > 100)
			{
				break;
			}
		}


		unsigned int w_buffer_len = 10000;
		wchar_t* w_buffer = new wchar_t[w_buffer_len];

#ifdef USE_PYTHON_TOKENIZER
		PythonTokenizerDecode(tokenizer, encoded_text, idx, w_buffer, &w_buffer_len);
#else
		Decode(tokenizer, encoded_text, idx, w_buffer, &w_buffer_len);
#endif

		return 0;
	}


	torch::Tensor embed_tokens(torch::Tensor x)
	{
		return embed_tokens_->forward(x);
	}


	void to(const torch::Device device, bool non_blocking = false)
	{
		int i;
		
		embed_tokens_->to(device);
		lm_head_->to(device);
		norm_weight_ = norm_weight_.to(device);

		for (i = 0; i < num_levels_; i++)
		{
			attn_block[i]->to(device);
		}

	}

	void train(bool on)
	{
		int i;

		embed_tokens_->train(on);
		lm_head_->train(on);

		for (i = 0; i < num_levels_; i++)
		{
			attn_block[i]->train(on);
		}

		torch::nn::Module::train(on);
	}

private:
	std::shared_ptr<LLaMABlock>* attn_block;
	torch::nn::Embedding embed_tokens_{ nullptr };
	float eps_;
	torch::Tensor norm_weight_;
	int num_levels_;
	int vocab_size_;
	int embed_dim_;

	torch::nn::Linear lm_head_{ nullptr };
	feature_cache kv_cache_;
};

