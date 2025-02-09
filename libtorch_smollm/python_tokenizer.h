#ifndef  PYTHON_TOKENIZER_H
#define PYTHON_TOKENIZER_H


class PythonTokenizer
{
public:
	PythonTokenizer() {}
	~PythonTokenizer() {}

	bool Init();
	int Encode(const char* text, unsigned int* encoded_text, unsigned int* encoded_text_len);
	int Decode(unsigned int* encoded_text, unsigned int encoded_text_len, wchar_t* decoded_text, unsigned int* decoded_text_len);
	unsigned int GetVocabularySize();
	int PythonTokenizer::GetBOSTokenId();
	int PythonTokenizer::GetEOSTokenId();
	int PythonTokenizer::GetPADTokenId();
	int PythonTokenizer::GetUNKTokenId();
	int PythonTokenizer::GetSEPTokenId();

private:
	PyObject* pEncodeFn_;
	PyObject* pDecodeFn_;
	PyObject* pVocabSizeFn_;
	PyObject* pBOSFn_;
	PyObject* pEOSFn_;
	PyObject* pPADFn_;
	PyObject* pUNKFn_;
	PyObject* pSEPFn_;
};



void* InitializePythonTokenizer();
void FreePythonTokenizer(void* tokenizer);
int PythonTokenizerEncode(void* tokenizer, const char* text, unsigned int* encoded_text, unsigned int* encoded_text_len);
int PythonTokenizerDecode(void* tokenizer, unsigned int* encoded_text, unsigned int encoded_text_len, wchar_t* decoded_text, unsigned int* decoded_text_len);
unsigned int GetPythonTokenizerVocabularySize(void* tokenizer);
int GetPythonTokenizerBOSTokenId(void* tokenizer);
int GetPythonTokenizerEOSTokenId(void* tokenizer);
int GetPythonTokenizerPADTokenId(void* tokenizer);
int GetPythonTokenizerUNKTokenId(void* tokenizer);
int GetPythonTokenizerSEPTokenId(void* tokenizer);

#endif // ! PYTHON_TOKENIZER_H

