//修改人：周渝杰
//修改时间：2020/10/17
//运行说明：

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

// #define CIRCLE
// #define BOY
#define CALIFORMIA

//对Circuit_noise.jpg图像进行去噪处理,使用中值滤波去掉椒盐噪声
#ifdef CIRCLE 
int main(void)
{   
    Mat dstImage, srcImage;
    srcImage = imread("Circuit_noise.jpg");
    medianBlur(srcImage, dstImage, 3);
    imshow("原图", srcImage);
    imshow("处理后的图像", dstImage);
    waitKey(0);
    return 0;
}
#endif

#ifdef BOY
int main(void)
{
    Mat dstImage, srcImage, gif;
    VideoCapture capture("boy_noisy.gif"); 
    capture >> srcImage;

    Mat mImage = srcImage.clone();
    if (mImage.data == 0)
	{
		cerr << "Image reading error" << endl;
		system("pause");
		return -1;
	}
	namedWindow("The original image", WINDOW_NORMAL);
	imshow("The original image", mImage);
    
	//Extending image
	int m = getOptimalDFTSize(mImage.rows);
	int n = getOptimalDFTSize(mImage.cols);
	copyMakeBorder(mImage, mImage, 0, m - mImage.rows, 0, n - mImage.cols, BORDER_CONSTANT, Scalar(0));
    
	//Fourier transform
	Mat mFourier(mImage.rows + m, mImage.cols + n, CV_32FC2, Scalar(0, 0));
    Mat a = Mat_<float>(mImage);
    waitKey(0);
	Mat mForFourier[] = { Mat_<float>(mImage), Mat::zeros(mImage.size(), CV_32F) };
	Mat mSrc;
    
    
	merge(mForFourier, 2, mSrc);
	dft(mSrc, mFourier);
    
	//channels[0] is the real part of Fourier transform,channels[1] is the imaginary part of Fourier transform 
	vector<Mat> channels;
	split(mFourier, channels);
	Mat mRe = channels[0];
	Mat mIm = channels[1];

	//Calculate the amplitude
	Mat mAmplitude;
	magnitude(mRe, mIm, mAmplitude);

	//Logarithmic scale
	mAmplitude += Scalar(1);
	log(mAmplitude, mAmplitude);

	//The normalized
	normalize(mAmplitude, mAmplitude, 0, 255, NORM_MINMAX);

	Mat mResult(mImage.rows, mImage.cols, CV_8UC1, Scalar(0));
	for (int i = 0; i < mImage.rows; i++)
	{
		uchar* pResult = mResult.ptr<uchar>(i);
		float* pAmplitude = mAmplitude.ptr<float>(i);
		for (int j = 0; j < mImage.cols; j++)
		{
			pResult[j] = (uchar)pAmplitude[j];
		}
	}

	Mat mQuadrant1 = mResult(Rect(mResult.cols / 2, 0, mResult.cols / 2, mResult.rows / 2));
	Mat mQuadrant2 = mResult(Rect(0, 0, mResult.cols / 2, mResult.rows / 2));
	Mat mQuadrant3 = mResult(Rect(0, mResult.rows / 2, mResult.cols / 2, mResult.rows / 2));
	Mat mQuadrant4 = mResult(Rect(mResult.cols / 2, mResult.rows / 2, mResult.cols / 2, mResult.rows / 2));

	Mat mChange1 = mQuadrant1.clone();
	//mQuadrant1 = mQuadrant3.clone();
	//mQuadrant3 = mChange1.clone();
	mQuadrant3.copyTo(mQuadrant1);
	mChange1.copyTo(mQuadrant3);

	Mat mChange2 = mQuadrant2.clone();
	//mQuadrant2 = mQuadrant4.clone();
	//mQuadrant4 = mChange2.clone();
	mQuadrant4.copyTo(mQuadrant2);
	mChange2.copyTo(mQuadrant4);

	namedWindow("The Fourier transform", WINDOW_NORMAL);
	imshow("The Fourier transform", mResult);
	waitKey();
	destroyAllWindows();


    // medianBlur(srcImage, dstImage, 3);
    imshow("原图", srcImage);
    // imshow("处理后的图像", dstImage);
    waitKey(0);
    return 0;
}
#endif


#ifdef CALIFORMIA
Mat WienerFilter(const Mat &src, const Mat &ref, int stddev);
Mat GetSpectrum(const Mat &src);

int main(void)
{   
    Mat dstImage, srcImage, gaissianImage;
    srcImage = imread("blur3.JPG");
	cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
    GaussianBlur(srcImage, gaissianImage, Size(3, 3), 3, 3);
	dstImage = WienerFilter(srcImage, gaissianImage, 30);
    imshow("原图", srcImage);
	imshow("gaissianImage", gaissianImage);
    imshow("处理后的图像", dstImage);
    waitKey(0);
    return 0;
}

Mat WienerFilter(const Mat &src, const Mat &ref, int stddev)
{
    //这些图片是过程中会用到的，pad是原图像0填充后的图像，cpx是双通道频域图，mag是频域幅值图，dst是滤波后的图像
    Mat pad, cpx, dst;
	
	// imshow("src", src);
	// imshow("ref", ref);
    //获取傅里叶变化最佳图片尺寸，为2的指数
    int m = getOptimalDFTSize(src.rows);
    int n = getOptimalDFTSize(src.cols);

    //对原始图片用0进行填充获得最佳尺寸图片
    copyMakeBorder(src, pad, 0, m-src.rows, 0, n-src.cols, BORDER_CONSTANT, Scalar::all(0));
	
    //获得参考图片频谱
    Mat tmpR(pad.rows, pad.cols, CV_8U);
    resize(ref, tmpR, tmpR.size());
    Mat refSpectrum = GetSpectrum(tmpR);
	// waitKey(0);
    //获得噪声频谱
    Mat tmpN(pad.rows, pad.cols, CV_32F);
    randn(tmpN, Scalar::all(0), Scalar::all(stddev));
    Mat noiseSpectrum = GetSpectrum(tmpN);

    //对src进行傅里叶变换
    Mat planes[] = {Mat_<float>(pad), Mat::zeros(pad.size(), CV_32F)};
    merge(planes, 2, cpx);
    dft(cpx, cpx);
    split(cpx, planes);

    //维纳滤波因子
    Mat factor = refSpectrum / (refSpectrum + noiseSpectrum);
    multiply(planes[0], factor, planes[0]);
    multiply(planes[1], factor, planes[1]);

    //重新合并实部planes[0]和虚部planes[1]
    merge(planes, 2, cpx);

    //进行反傅里叶变换
    idft(cpx, dst, DFT_SCALE | DFT_REAL_OUTPUT);

    dst.convertTo(dst, CV_8UC1);
    return dst;
}


Mat GetSpectrum(const Mat &src)
{
    Mat dst, cpx;
    Mat planes[] = {Mat_<float>(src), Mat::zeros(src.size(), CV_32F)};
    merge(planes, 2, cpx);
    dft(cpx, cpx);
    split(cpx, planes);
    magnitude(planes[0], planes[1], dst);
    //频谱就是频域幅度图的平方
    multiply(dst, dst, dst);
    return dst;
}

#endif