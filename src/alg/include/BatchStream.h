#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include <vector>
#include <assert.h>
#include <algorithm>
#include "NvInfer.h"
#include "NvCaffeParser.h"


class BatchStream
{
public:
	BatchStream(int n, int c, int h, int w, int max_batches) : mBatchSize(n), mMaxBatches(max_batches)
	{
		char* batch_dump_dir = getenv("TENSORRT_INT8_BATCH_DIRECTORY");
		if (batch_dump_dir != NULL)
		{
			filePath = std::string(batch_dump_dir) + "/";
		}
		else
		{
			filePath = "./"; //校验文件路径
		}
		
		mDims = nvinfer1::DimsNCHW{ n, c, h, w};
		
		mImageSize = mDims.c() * mDims.h() * mDims.w();
		mBatch.resize(mBatchSize*mImageSize, 0);
		mLabels.resize(mBatchSize, 0);
		mFileBatch.resize(mDims.n()*mImageSize, 0);
		mFileLabels.resize(mDims.n(), 0);
		reset(0);
	}

	void reset(int firstBatch)
	{
		mBatchCount = 0;
		mFileCount = 0;
		mFileBatchPos = mDims.n();
		skip(firstBatch);
	}

	bool next()
	{
		if (mBatchCount == mMaxBatches)
			return false;

		for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
		{
			assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.n());
			if (mFileBatchPos == mDims.n() && !update())
				return false;

			// copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
			csize = std::min(mBatchSize - batchPos, mDims.n() - mFileBatchPos);
			std::copy_n(getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
			std::copy_n(getFileLabels() + mFileBatchPos, csize, getLabels() + batchPos);
		}
		mBatchCount++;
		return true;
	}

	void skip(int skipCount)
	{
		if (mBatchSize >= mDims.n() && mBatchSize%mDims.n() == 0 && mFileBatchPos == mDims.n())
		{
			mFileCount += skipCount * mBatchSize / mDims.n();
			return;
		}

		int x = mBatchCount;
		for (int i = 0; i < skipCount; i++)
			next();
		mBatchCount = x;
	}

	float *getBatch() { return &mBatch[0]; }
	float *getLabels() { return &mLabels[0]; }
	int getBatchesRead() const { return mBatchCount; }
	int getBatchSize() const { return mBatchSize; }
	nvinfer1::DimsNCHW getDims() const { return mDims; }
private:
	float* getFileBatch() { return &mFileBatch[0]; }
	float* getFileLabels() { return &mFileLabels[0]; }
	bool update()
	{
		std::string inputFileName = filePath + std::string("batch") + std::to_string(mFileCount++);
		std::cout << "update inputFileName " << inputFileName << std::endl;
		FILE * file = fopen(inputFileName.c_str(), "rb");
		if (!file)
		{
			std::cout << "ERROR: Can not find BatchFile " << inputFileName.c_str() << std::endl;
			assert(file != nullptr);
		}

		int d[4];
		fread(d, sizeof(int), 4, file);
		assert(mDims.n() == d[0] && mDims.c() == d[1] && mDims.h() == d[2] && mDims.w() == d[3]);
		
		size_t readInputCount = fread(getFileBatch(), sizeof(float), mDims.n()*mImageSize, file);
		//std::cout << "readInputCount" << readInputCount << " " << mDims.n() << " " << mImageSize<< std::endl;
		assert(readInputCount == size_t(mDims.n()*mImageSize));

		fclose(file);
		mFileBatchPos = 0;
		return true;
	}

	int mBatchSize{ 0 };
	int mMaxBatches{ 0 };
	int mBatchCount{ 0 };

	int mFileCount{ 0 }, mFileBatchPos{ 0 };
	int mImageSize{ 0 };

	nvinfer1::DimsNCHW mDims;
	std::vector<float> mBatch;
	std::vector<float> mLabels;
	std::vector<float> mFileBatch;
	std::vector<float> mFileLabels;
	std::string filePath;
};


#endif

