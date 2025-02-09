#ifndef CSV_PARSER_H
#define CSV_PARSER_H
// Note: the ___2 functions were added later to support reading multiple fields from each record (original code only read one field)

class CSVReader
{
public:
	CSVReader() {}
	~CSVReader() {}
	bool Init(const char* file_name, const char* field_name)
	{
		bool bool_ret;
		char buffer[1024];

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
		errno_t err;
		err = fopen_s(&stream_, file_name, "rb");
		if (err)
		{
			bool_ret = false;
			goto Exit;
		}
#else
		stream = fopen(file_name, "rb");
		if (!stream_)
		{
			ret = -1;
			goto Exit;
		}
#endif

		bool_ret = fgets(buffer, sizeof(buffer), stream_);
		if (bool_ret)
		{
			field_index_ = GetFieldIndex(buffer, field_name);
			if (field_index_ < 0)
			{
				return false;
			}
		}
		else
		{
			goto Exit;
		}

		count_ = 0;
		bool_ret = true;
	Exit:
		return bool_ret;
	}

	bool Init2(const char* file_name, const char** field_names, int field_name_count)
	{
		bool bool_ret;
		char buffer[1024];
		int i;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
		errno_t err;
		err = fopen_s(&stream_, file_name, "rb");
		if (err)
		{
			bool_ret = false;
			goto Exit;
		}
#else
		stream = fopen(file_name, "rb");
		if (!stream_)
		{
			ret = -1;
			goto Exit;
		}
#endif

		bool_ret = fgets(buffer, sizeof(buffer), stream_);
		if (bool_ret)
		{
			field_indies_ = new int[field_name_count];
			for (i = 0; i < field_name_count; i++)
			{
				field_indies_[i] = GetFieldIndex(buffer, field_names[i]);
				if (field_indies_[i] < 0)
				{
					return false;
				}
			}
		}
		else
		{
			goto Exit;
		}

		count_ = 0;
		field_name_count_ = field_name_count;
		bool_ret = true;
	Exit:
		return bool_ret;
	}


	uint32_t CountQuotes(const char* str) 
	{
		uint32_t count = 0;
		char ch;

		while (true) 
		{
			ch = *str;
			if (ch == '\0')
			{
				break;
			}

			if (ch == '"') 
			{
				++count;
			}
			str++;
		}
		return count;
	}

//#define CONV_CR_LF_TO_LF
	int32_t ReadNextRecord(char* buffer, uint32_t buffer_size, char* scratch, uint32_t scratch_size, char* scratch2, uint32_t scratch2_size)
	{
		uint32_t len;
		int ret;
		char* currentRecord;
		uint32_t quote_count;
		int currentRecord_len;
		currentRecord = scratch2;
		currentRecord[0] = '\0';
		quote_count = 0;

		currentRecord_len = 0;

		while (fgets(scratch, scratch_size, stream_))
		{
			len = (uint32_t)strlen(scratch);

			if (len >= scratch_size - 1)
			{
				printf("Buffer may be too small, increase buffer size to be safe\n");
				ret = -1;
				goto Exit;
			}

			if (len > 0)
			{
#ifdef CONV_CR_LF_TO_LF
				char* ch = &scratch[len - 1];
				if (*ch == '\n' || *ch == '\r')
				{
					while (*ch == '\n' || *ch == '\r')
					{
						ch--;
					}
					ch++;
					*ch = '\0';
				}
				len = ch - scratch;
#endif
				memcpy(&currentRecord[currentRecord_len], scratch, len + 1); //strcat_s(currentRecord, scratch2_size, scratch);
				currentRecord_len += len;

				quote_count += CountQuotes(scratch);
			}

			if (quote_count % 2 == 0)
			{
				len = buffer_size;
				ret = ParseCSVLine(currentRecord, buffer, &len);
				if (!ret)
				{
					count_++;
					ret = (int32_t)len;
					goto Exit;
				}
				else
				{
					assert(0);
					ret = -1;
					goto Exit;
				}
			}
			else
			{
#ifdef CONV_CR_LF_TO_LF
				currentRecord[currentRecord_len++] = '\n';  //strcat_s(currentRecord, scratch2_size, "\n");
				currentRecord[currentRecord_len] = '\0';
#endif
			}
		}

		fclose(stream_);
		stream_ = nullptr;
		ret = 0;

	Exit:
		return ret;
	}

	int ParseCSVLine(char* currentRecord, char* field, uint32_t* field_buffer_len)
	{
		size_t len;
		size_t i;
		bool insideQuotes = false;
		int field_idx;
		int idx;
		int ret;

		field_idx = 0;
		idx = 0;
		len = strlen(currentRecord);

		for (i = 0; i < len; i++)
		{
			char c = currentRecord[i];
			if (c == '"')
			{
				if (insideQuotes && i + 1 < len && currentRecord[i + 1] == '"')
				{
					if (field_idx == field_index_)
					{
						field[idx++] = '"';
					}

					++i; // Skip the second quote
				}
				else
				{
					insideQuotes = !insideQuotes;
				}
			}
			else
			{
				if (c == ',' && !insideQuotes)
				{
					field_idx++;
					if (field_idx > field_index_)
					{
						field[idx] = '\0';
						*field_buffer_len = idx;
						ret = 0;
						goto Exit;
					}
				}
				else
				{
					if (field_idx == field_index_)
					{
						field[idx++] = c;
					}
				}
			}
		}

		ret = -1;
	Exit:
		return ret;
	}

	// Note: buffer_sizes is an in-out paramter
	int32_t ReadNextRecord2(char** buffers, uint32_t* buffer_sizes, char* scratch, uint32_t scratch_size, char* scratch2, uint32_t scratch2_size)
	{
		uint32_t len;
		int ret;
		char* currentRecord;
		uint32_t quote_count;
		int currentRecord_len;
		currentRecord = scratch2;
		currentRecord[0] = '\0';
		quote_count = 0;

		currentRecord_len = 0;

		while (fgets(scratch, scratch_size, stream_))
		{
			len = (uint32_t)strlen(scratch);

			if (len >= scratch_size - 1)
			{
				printf("Buffer may be too small, increase buffer size to be safe\n");
				ret = -1;
				goto Exit;
			}

			if (len > 0)
			{
#ifdef CONV_CR_LF_TO_LF
				char* ch = &scratch[len - 1];
				if (*ch == '\n' || *ch == '\r')
				{
					while (*ch == '\n' || *ch == '\r')
					{
						ch--;
					}
					ch++;
					*ch = '\0';
				}
				len = ch - scratch;
#endif
				memcpy(&currentRecord[currentRecord_len], scratch, len + 1); //strcat_s(currentRecord, scratch2_size, scratch);
				currentRecord_len += len;

				quote_count += CountQuotes(scratch);
			}

			if (quote_count % 2 == 0)
			{
				ret = ParseCSVLine2(currentRecord, buffers, buffer_sizes);
				if (!ret)
				{
					count_++;
					len = 0;
					for (int i = 0; i < field_name_count_; i++)
					{
						len += buffer_sizes[i]; // may not make much sense but analogous to single field version
					}
					ret = (int32_t)len;
					goto Exit;
				}
				else
				{
					//assert(0);
					printf("CSV parse failed!\n");
					ret = -1;
					goto Exit;
				}
			}
			else
			{
#ifdef CONV_CR_LF_TO_LF
				currentRecord[currentRecord_len++] = '\n';  //strcat_s(currentRecord, scratch2_size, "\n");
				currentRecord[currentRecord_len] = '\0';
#endif
			}
		}

		fclose(stream_);
		stream_ = nullptr;
		ret = 0;

	Exit:
		return ret;
	}


	int ParseCSVLine2(char* currentRecord, char** fields_buffer, uint32_t* field_buffer_lens)
	{
		size_t len;
		size_t i;
		bool insideQuotes = false;
		int field_idx;
		int idx;
		int ret;
		int current_valid_field_idx;
		int fields_extracted;


		fields_extracted = 0;
		current_valid_field_idx = field_indies_[fields_extracted];
		field_idx = 0;
		idx = 0;
		len = strlen(currentRecord);

		for (i = 0; i < len; i++)
		{
			char c = currentRecord[i];
			if (c == '"')
			{
				if (insideQuotes && i + 1 < len && currentRecord[i + 1] == '"')
				{
					if (field_idx == current_valid_field_idx)
					{
						fields_buffer[fields_extracted][idx++] = '"';
					}

					++i; // Skip the second quote
				}
				else
				{
					insideQuotes = !insideQuotes;
				}
			}
			else
			{
				if (c == ',' && !insideQuotes)
				{
					if (field_idx == current_valid_field_idx) // this could be the begining or the end, use idx to differentiate
					{
						if (idx > 0)
						{
							fields_buffer[fields_extracted][idx] = '\0';
							field_buffer_lens[fields_extracted] = idx;

							fields_extracted++;
							current_valid_field_idx = field_indies_[fields_extracted];
							idx = 0;  // ready for the next field

						}
					}

					field_idx++;
					if (fields_extracted >= field_name_count_)
					{
						ret = 0;
						goto Exit;
					}

				}
				else
				{
					if (field_idx == current_valid_field_idx)
					{
						fields_buffer[fields_extracted][idx++] = c;
					}
				}
			}
		}

		ret = -1;
	Exit:
		return ret;
	}


	int GetFieldIndex(char* fields, const char* field_name) 
	{
		char* context = nullptr;
		char* ch;
		int index;
		char tokens[] = ",\r\n";
		size_t len;
		char* temp;
		int ret;

		len = strlen(fields);
		temp = new char[len + 1];
		strcpy_s(temp, len + 1, fields); // so that strtok does not modify fields

		index = 0;
		ch = strtok_s(temp, tokens, &context);

		while (ch) 
		{
			if (!strcmp(ch, field_name)) 
			{
				ret = index;
				goto Exit;
			}
			++index;
			ch = strtok_s(nullptr, tokens, &context);  // Get the next token
		}

		ret = -1;
	Exit:
		delete temp;
		return ret;
	}
	

private:
	FILE* stream_;
	int field_index_;
	int count_;

	int field_name_count_;
	int* field_indies_;
};


#endif // CSV_PARSER_H