#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring> 
#include <cstdint>

#include "python_tokenizer.h"


static const char* python_code = "\
import numpy as np\n\
from transformers import AutoTokenizer\n\
checkpoint = \"HuggingFaceTB/SmolLM2-135M-Instruct\"\n\
tokenizer = AutoTokenizer.from_pretrained(checkpoint)\n\
def encode(text):\n\
    encoded = tokenizer.encode(text)\n\
    return np.array(encoded, dtype=np.int32)\n\
def decode(numbers):\n\
    numbers = numbers.tolist()\n\
    decoded = tokenizer.decode(numbers)\n\
    return decoded\n\
def vocabulary_size():\n\
    return tokenizer.vocab_size\n\
def bos_token_id():\n\
    return tokenizer.bos_token_id\n\
def eos_token_id():\n\
    return tokenizer.eos_token_id\n\
def pad_token_id():\n\
    return tokenizer.pad_token_id\n\
def unk_token_id():\n\
    return tokenizer.unk_token_id\n\
def sep_token_id():\n\
    return tokenizer.sep_token_id\n";


bool PythonTokenizer::Init()
{
	Py_Initialize();


	import_array();


	PyObject* pCompiledFn = Py_CompileString(python_code, "", Py_file_input);
	if (!pCompiledFn)
	{
		return false;
	}

	PyObject* pModule = PyImport_ExecCodeModule("test", pCompiledFn);
	if (!pModule)
	{
		return false;
	}


	pEncodeFn_ = PyObject_GetAttrString(pModule, "encode");
	pDecodeFn_ = PyObject_GetAttrString(pModule, "decode");
	pVocabSizeFn_ = PyObject_GetAttrString(pModule, "vocabulary_size");
	pBOSFn_ = PyObject_GetAttrString(pModule, "bos_token_id");
	pEOSFn_ = PyObject_GetAttrString(pModule, "eos_token_id");
	pPADFn_ = PyObject_GetAttrString(pModule, "pad_token_id");
	pUNKFn_ = PyObject_GetAttrString(pModule, "unk_token_id");
	pSEPFn_ = PyObject_GetAttrString(pModule, "sep_token_id");

	return true;
}

int PythonTokenizer::Encode(const char* text, unsigned int* encoded_text, unsigned int* encoded_text_len)
{
	PyObject* pArgsEncode = Py_BuildValue("(s)", text);
	PyObject* pEncodedResult = PyObject_CallObject(pEncodeFn_, pArgsEncode);
	Py_DECREF(pArgsEncode);
	int ret;

	ret = 0;
	if (pEncodedResult && PyArray_Check(pEncodedResult))
	{
		PyArrayObject* npArray = reinterpret_cast<PyArrayObject*>(pEncodedResult);

		// Get the array data and size
		int* arrayData = static_cast<int*>(PyArray_DATA(npArray));
		npy_intp arraySize = PyArray_SIZE(npArray);

		if (arraySize <= (*encoded_text_len))
		{
			for (npy_intp i = 0; i < arraySize; ++i)
			{
				encoded_text[i] = arrayData[i];
			}
			*encoded_text_len = (unsigned int)arraySize;
		}
		else
		{
			ret = -1;
		}
		Py_DECREF(pEncodedResult);
	}
	else
	{
		ret = -1;
	}

	return ret;
}

int PythonTokenizer::Decode(unsigned int* encoded_text, unsigned int encoded_text_len, wchar_t* decoded_text, unsigned int* decoded_text_len)
{
	npy_intp dims[1] = { encoded_text_len };

	PyObject* pArray = PyArray_SimpleNewFromData(1, dims, NPY_UINT32, encoded_text);
	if (!pArray)
	{
		PyErr_Print();
		std::cerr << "Failed to create NumPy array!" << std::endl;
		return -1;
	}

	PyObject* pArgsDecode = Py_BuildValue("(O)", pArray);  // Pass the array as argument
	PyObject* pDecodedResult = PyObject_CallObject(pDecodeFn_, pArgsDecode);
	Py_DECREF(pArgsDecode);
	Py_DECREF(pArray);


	if (pDecodedResult)
	{
		const char* decodedText = PyUnicode_AsUTF8(pDecodedResult);
		std::cout << "Decoded: " << decodedText << std::endl;
		Py_DECREF(pDecodedResult);
	}

	return 0;
}


unsigned int PythonTokenizer::GetVocabularySize()
{
	// Call the function with no arguments
	PyObject* pValue = PyObject_CallObject(pVocabSizeFn_, nullptr);

	if (!pValue)
	{
		PyErr_Print();
		return 0;
	}

	// Convert the result to a long
	unsigned long vocabSize = PyLong_AsUnsignedLong(pValue);
	Py_DECREF(pValue);

	return (unsigned int)vocabSize;
}

int PythonTokenizer::GetBOSTokenId()
{
	PyObject* pValue = PyObject_CallObject(pBOSFn_, nullptr);
	if (!pValue)
	{
		PyErr_Print();
		return -1;
	}

	int bosTokenId = PyLong_AsLong(pValue);
	Py_DECREF(pValue);
	return bosTokenId;
}

int PythonTokenizer::GetEOSTokenId()
{
	PyObject* pValue = PyObject_CallObject(pEOSFn_, nullptr);
	if (!pValue)
	{
		PyErr_Print();
		return -1;
	}

	int eosTokenId = PyLong_AsLong(pValue);
	Py_DECREF(pValue);
	return eosTokenId;
}

int PythonTokenizer::GetPADTokenId()
{
	PyObject* pValue = PyObject_CallObject(pPADFn_, nullptr);
	if (!pValue)
	{
		PyErr_Print();
		return -1;
	}

	int padTokenId = PyLong_AsLong(pValue);
	Py_DECREF(pValue);
	return padTokenId;
}

int PythonTokenizer::GetUNKTokenId()
{
	PyObject* pValue = PyObject_CallObject(pUNKFn_, nullptr);
	if (!pValue)
	{
		PyErr_Print();
		return -1;
	}

	int unkTokenId = PyLong_AsLong(pValue);
	Py_DECREF(pValue);
	return unkTokenId;
}

int PythonTokenizer::GetSEPTokenId()
{
	PyObject* pValue = PyObject_CallObject(pSEPFn_, nullptr);
	if (!pValue)
	{
		PyErr_Print();
		return -1;
	}

	int sepTokenId = PyLong_AsLong(pValue);
	Py_DECREF(pValue);
	return sepTokenId;
}

void* InitializePythonTokenizer()
{
	PythonTokenizer* tokenizer;

	tokenizer = new PythonTokenizer();

	if (tokenizer->Init())
	{
		return (void*)tokenizer;
	}

	delete tokenizer;
	return nullptr;
}

void FreePythonTokenizer(void* tokenizer)
{

}

int PythonTokenizerEncode(void* tokenizer, const char* text, unsigned int* encoded_text, unsigned int* encoded_text_len)
{
	PythonTokenizer* tok;

	tok = (PythonTokenizer*)tokenizer;

	return tok->Encode(text, encoded_text, encoded_text_len);

}


int PythonTokenizerDecode(void* tokenizer, unsigned int* encoded_text, unsigned int encoded_text_len, wchar_t* decoded_text, unsigned int* decoded_text_len)
{
	PythonTokenizer* tok;

	tok = (PythonTokenizer*)tokenizer;

	return tok->Decode(encoded_text, encoded_text_len, decoded_text, decoded_text_len);
}


unsigned int GetPythonTokenizerVocabularySize(void* tokenizer)
{
	PythonTokenizer* tok;

	tok = (PythonTokenizer*)tokenizer;

	return tok->GetVocabularySize();
}

int GetPythonTokenizerBOSTokenId(void* tokenizer)
{
	PythonTokenizer* tok = (PythonTokenizer*)tokenizer;
	return tok->GetBOSTokenId();
}

int GetPythonTokenizerEOSTokenId(void* tokenizer)
{
	PythonTokenizer* tok = (PythonTokenizer*)tokenizer;
	return tok->GetEOSTokenId();
}

int GetPythonTokenizerPADTokenId(void* tokenizer)
{
	PythonTokenizer* tok = (PythonTokenizer*)tokenizer;
	return tok->GetPADTokenId();
}

int GetPythonTokenizerUNKTokenId(void* tokenizer)
{
	PythonTokenizer* tok = (PythonTokenizer*)tokenizer;
	return tok->GetUNKTokenId();
}

int GetPythonTokenizerSEPTokenId(void* tokenizer)
{
	PythonTokenizer* tok = (PythonTokenizer*)tokenizer;
	return tok->GetSEPTokenId();
}