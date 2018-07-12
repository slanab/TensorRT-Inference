#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"
#include <string.h>

using namespace nvuffparser;
using namespace nvinfer1;

class Logger : public nvinfer1::ILogger			
{
    public:
	void log(nvinfer1::ILogger::Severity severity, const char* msg) override
	{
		// suppress info-level messages
        if (severity == Severity::kINFO) return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
	}
};

static Logger gLogger;

#define MAX_WORKSPACE (1 << 30)

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "sample_uff_mnist: " + std::string(message);            \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)

#define CHECK(status)									    \
{										            \
	if (status != 0)								    \
	{										    \
		std::cout << "Cuda failure: " << status;				    \
		abort();								    \
	}										    \
}

inline int64_t volume(const Dims& d)
{
	int64_t v = 1;
	for (int64_t i = 0; i < d.nbDims; i++)
		v *= d.d[i];
	return v;
}


inline unsigned int elementSize(DataType t)
{
	switch (t)
	{
	case DataType::kFLOAT: return 4;
	case DataType::kHALF: return 2;
	case DataType::kINT8: return 1;
	}
	assert(0);
	return 0;
}


static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;

static const int IMG_H = 584;
static const int IMG_W = 565;

std::string locateFile(const std::string& input)
{
	bool found{false};
	std::string dir = "../data/";
        std::string file = dir + input;
        std::cout << "Looking for " << file << std::endl;
	std::ifstream checkFile(file);
	found = checkFile.is_open();
	if (found) 
	{
		std::cout << "Found requested file\n";
	}
	return file;
}

void loadFullImage(float buffer[IMG_H][IMG_W]) 
{
	float fullImage[IMG_H * IMG_W];
	std::string filename = "../data/all_coeff_01_test.txt";
	std::cout << "Reading coefficients from " << filename << std::endl;

	string line;
	ifstream myfile (filename);
	int i = 0;
	if (myfile.is_open())
	{
		while ( myfile.good() )
		{
			getline (myfile,line);
			//cout << i << "'" << line << "'" << endl;
			if (line != "") 
			{
				fullImage[i++] = stof(line);
			}
		}
		myfile.close();
	}

        memcpy(buffer[0], fullImage, IMG_H * IMG_W * sizeof(float)) ;
	std::cout << "\nDone reading input, got " << i << " values\n";
}

void printTile(float tile[INPUT_H*INPUT_W])
{
	for (int i = 0; i < INPUT_H*INPUT_W; i++) 
	{
		if (i % INPUT_W == 0) {
			std::cout << std::endl;
		}
		std::cout << int(tile[i]) << "\t";
	}
	std::cout << std::endl;
}

void printTile2D(float tile[][INPUT_W])
{
	for (int i = 0; i < INPUT_H; i++) 
	{
		for (int j = 0; j < INPUT_W; j++) {
			std::cout << tile[i][j] << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void saveCoeffs(float* coefficients, int numElements, string filename)
{
	std::cout << "Saving coefficients to " << filename << std::endl;
	string line;
	ofstream myfile;
	myfile.open(filename);
	for (int i = 0; i < numElements; i++)
	{
		myfile << std::to_string(coefficients[i]) << std::endl;
	}
	myfile.close();
}


void saveCoeffs2D(float coefficients[][28], int numElements, string filename)
{
	std::cout << "Saving coefficients to " << filename << std::endl;
	string line;
	ofstream myfile;
	myfile.open(filename);
	for (int i = 0; i < numElements; i++)
	{
		for (int j = 0; j < numElements; j++) {
			myfile << std::to_string(coefficients[i][j]) << std::endl;
		}
	}
	myfile.close();
}

void saveImageFull(float coefficients[][IMG_W], string filename)
{
	std::cout << "Saving coefficients to " << filename << std::endl;
	string line;
	ofstream myfile;
	myfile.open(filename);
	for (int i = 0; i < IMG_H; i++)
	{
		for (int j = 0; j < IMG_W; j++) {
			myfile << std::to_string(coefficients[i][j]) << std::endl;
		}
	}
	myfile.close();
}

void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}


std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

void printOutput(int64_t eltCount, DataType dtype, void* buffer)
{
    std::cout << eltCount << " eltCount" << std::endl;
    assert(elementSize(dtype) == sizeof(float));
    std::cout << "--- OUTPUT ---" << std::endl;

    size_t memSize = eltCount * elementSize(dtype);
    float* outputs = new float[eltCount];
    CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

    int maxIdx = 0;
    for (int i = 0; i < eltCount; ++i)
        if (outputs[i] > outputs[maxIdx])
            maxIdx = i;

    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        std::cout << eltIdx << " => " << outputs[eltIdx] << "\n";
    }

    std::cout << std::endl;
    delete[] outputs;
}

void saveCoeffs(int64_t eltCount, DataType dtype, void* buffer, string name)
{
	size_t memSize = eltCount * elementSize(dtype);
	float* outputs = new float[eltCount];
	CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

	string filename = "../" + name + "_output_coeffs.txt";
	saveCoeffs(outputs, eltCount, filename);

	delete[] outputs;
}

void loadOutput(int64_t eltCount, DataType dtype, void* buffer, float output_buffer[28*28*2])
{
	size_t memSize = eltCount * elementSize(dtype);
	float* outputs = new float[eltCount];
	CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

	memcpy(output_buffer, outputs, memSize);

	delete[] outputs;
}

ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

#if 1
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
#else
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kHALF))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
    builder->setHalf2Mode(true);
#endif

    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");

    /* we can clean the network and the parser */
    network->destroy();
    builder->destroy();

    return engine;
}


void printImage(int image[][IMG_W])
{
	for (int i = 0; i < IMG_H; i++) {
		for (int j = 0; j < IMG_W; j++) {
			std::cout << image[i][j] << " ";
		}
		std::cout << std::endl;
	}
}
void execute(ICudaEngine& engine)
{
	float img2D[IMG_H][IMG_W];
	loadFullImage(img2D);

	IExecutionContext* context = engine.createExecutionContext();
	int batchSize = 1;

	int nbBindings = engine.getNbBindings();
	assert(nbBindings == 2);

	std::vector<void*> buffers(nbBindings);
	auto buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

	int bindingIdxInput = 0;
	for (int i = 0; i < nbBindings; ++i)
	{
		if (engine.bindingIsInput(i))
			bindingIdxInput = i;
		else
		{
			auto bufferSizesOutput = buffersSizes[i];
			buffers[i] = safeCudaMalloc(bufferSizesOutput.first *
					elementSize(bufferSizesOutput.second));
		}
	}

	auto bufferSizesInput = buffersSizes[bindingIdxInput];

	float tile1D[INPUT_H*INPUT_W];

	int imgHits[IMG_H][IMG_W];
	for (int i = 0; i < IMG_H; i++) {
		for (int j = 0; j < IMG_W; j++) {
			imgHits[i][j] = 0;
		}
	}
	float imgTotal[IMG_H][IMG_W];
	for (int i = 0; i < IMG_H; i++) {
		for (int j = 0; j < IMG_W; j++) {
			imgTotal[i][j] = 0.0;
		}
	}

	int x_offs_next = 0;
	int y_offs_next = 0;
	int maxRuns = 10000;
	int totalRuns = 0;
	float total = 0, ms;
	int separation = 8;
	for (int run = 0; run < maxRuns; run++)
	{
		int x_offs = x_offs_next;
		int y_offs = y_offs_next;
		//std::cout << "==========\nRun number " << run << std::endl;
		//std::cout << "X: " << x_offs << " Y: " << y_offs << std::endl;
		int index = 0;
		for (int i = 0; i < INPUT_H; i++)
		{
			for (int j = 0; j < INPUT_W; j++)
			{
				float value = img2D[x_offs + i][y_offs + j];
		                tile1D[index++] = value;
			}
		}
		//std::cout << "Data to be sent to CUDA:\n";
		//printTile(tile1D);
		
		int64_t eltCount = bufferSizesInput.first;
		DataType dtype = bufferSizesInput.second;
		size_t memSize = eltCount * elementSize(dtype);

		float fileData[INPUT_H * INPUT_W];
		memcpy(fileData, tile1D, INPUT_H*INPUT_W*sizeof(float));

		void* deviceMem = safeCudaMalloc(memSize);
		CHECK(cudaMemcpy(deviceMem, fileData, memSize, cudaMemcpyHostToDevice));

		buffers[bindingIdxInput] =  deviceMem;

		auto t_start = std::chrono::high_resolution_clock::now();
		context->execute(batchSize, &buffers[0]);
		auto t_end = std::chrono::high_resolution_clock::now();
		ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
		total += ms;

		for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
		{
			if (engine.bindingIsInput(bindingIdx))
				continue;

			auto bufferSizesOutput = buffersSizes[bindingIdx];
			int64_t eltCount = bufferSizesOutput.first;
			DataType dtype = bufferSizesOutput.second;
			float output_full[INPUT_H*INPUT_W];
			loadOutput(eltCount, dtype, buffers[bindingIdx], output_full);

			float output_ch1[INPUT_H][INPUT_W];
			//float output_ch2[INPUT_H][INPUT_W];
			memcpy(output_ch1, output_full, 28*28 * sizeof(output_full[0]));
			//printOutput(eltCount, dtype, buffers[bindingIdx]);
			//printTile2D(output_ch1);
			// TODO: This should be added to the global final count and then divided
			//saveCoeffs2D(output_ch1, 28, "../temp.txt");
			for (int i = 0; i < INPUT_H; i++)
			{
				for (int j = 0; j < INPUT_W; j++)
				{
					imgHits[x_offs + i][y_offs + j]++;
					imgTotal[x_offs + i][y_offs + j] = imgTotal[x_offs + i][y_offs + j] + output_ch1[i][j];
				}
			}
		}
		CHECK(cudaFree(buffers[bindingIdxInput]));

		x_offs_next = x_offs + separation;
		if (x_offs_next + separation > IMG_H)
		{
			y_offs_next = y_offs + separation;
			//std::cout << "Reached end of line at " << x_offs_next << std::endl;
			if (y_offs_next + separation > IMG_W)
			{
				//std::cout << "Reached end of image at " << x_offs_next << ":" << y_offs_next << std::endl;
				break;
			}
			x_offs_next = 0;
		}
		totalRuns++;
	}

	float imgFinal[IMG_H][IMG_W];
	for (int i = 0; i < IMG_H; i++) {
		for (int j = 0; j < IMG_W; j++) {
			imgFinal[i][j] = imgTotal[i][j] / imgHits[i][j];
		}
	}
	saveImageFull(imgFinal, "../full_image.txt");

	//printImage(imgHits);
	float average = total / totalRuns;
	std::cout << "Average over " << totalRuns << " runs is " << average << " ms. Total " << total << std::endl;

	for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
	{
		if (!engine.bindingIsInput(bindingIdx))
		{
			CHECK(cudaFree(buffers[bindingIdx]));
		}
	}
	context->destroy();
}


int main(int argc, char** argv)
{
    auto fileName = locateFile("uff_no_reshape.uff");

    int maxBatchSize = 1;
    auto parser = createUffParser();

    /* Register tensorflow input */
    parser->registerInput("Reshape", DimsCHW(1, 28, 28));
    parser->registerOutput("output_score/output_relu");

    ICudaEngine* engine = loadModelAndCreateEngine(fileName.c_str(), maxBatchSize, parser);
    std::cout << "TensorRT engine successfully created\n";
    if (!engine)
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");

    /* we need to keep the memory created by the parser */
    parser->destroy();

    execute(*engine);
    engine->destroy();
    shutdownProtobufLibrary();
    return EXIT_SUCCESS;
}
