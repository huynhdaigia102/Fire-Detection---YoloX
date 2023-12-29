#include <vector>
#include <iostream>
#include <fstream>

#include "engine.h"

using namespace std;

int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		cerr << "Usage: " << argv[0] << " core_model.onnx engine.plan" << endl;
	}

	ifstream onnxFile;
	onnxFile.open(argv[1], ios::in | ios::binary);
	cout << "Load model from " << argv[1] << endl;

	if (!onnxFile.good())
	{
		cerr << "\nERROR: Unable to read specified ONNX model " << argv[1] << endl;
		return -1;
	}

	onnxFile.seekg(0, onnxFile.end);
	size_t size = onnxFile.tellg();
	onnxFile.seekg(0, onnxFile.beg);

	auto *buffer = new char[size];
	onnxFile.read(buffer, size);
	onnxFile.close();

	bool verbose = true;
	size_t workspace_size = (1ULL << 30);
	const vector<int> dynamic_batch_opts{1, 8, 16};

	// decode params
	int top_n = 150;
	vector<vector<float>> anchors;
	anchors = {{10, 13, 16, 30, 33, 23},
						 {30, 61, 62, 45, 59, 119},
						 {116, 90, 156, 198, 373, 326}};
	vector<float> strides;
	strides = {8, 16, 32};

	// nms params
	// param from yolox
	float nms_thresh = 0.45;
	float score_thresh = 0.1f;
	int detections_per_im = 100;

	cout << "Building engine..." << endl;
	auto engine = rapid::Engine(buffer, size, dynamic_batch_opts,
															score_thresh, top_n, strides,
															nms_thresh, detections_per_im,
															verbose, workspace_size);
	engine.save(string(argv[2]));

	delete[] buffer;
	return 0;
}