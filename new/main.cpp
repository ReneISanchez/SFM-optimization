#include "opencv2/core/core.hpp"
#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace af;
using namespace std;
void mat_to_array(cv::Mat& input, af::array& output) {//, af::array& output) {
	input.convertTo(input, CV_32FC3); // floating point
	const unsigned size = input.rows * input.cols;
	const unsigned w = input.cols;
	const unsigned h = input.rows;
	float r[size];
	float g[size];
	//float b[size];
	int tmp = 0;
	for (unsigned i = 0; i < h; i++) {
		for (unsigned j = 0; j < w; j++) {
			Vec3f ip = input.at<Vec3f>(i, j);
			tmp = j * h + i; // convert to column major
			r[tmp] = ip[2];
			g[tmp] = ip[2];
			g[tmp] = ip[1];
			//b[tmp] = ip[0];
		}
	}
	output = join(2,
			af::array(h, w, r),
			af::array(h, w, g), af::array(h,w,g));
			//af::array(h, w, b))/255.f; // merge, set range [0-1]
}

int main(int argc, char* argv[])
{
	Mat A = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	af::array img_mat = loadImageNative(argv[1]);
//	af::array img_mat = loadImageNative(argv[1]);
	//af::array img_mat = loadImage(argv[1], true);
	//af::array img_mul = img_mat*img_mat;
	//Mat B = A.mul(A);
	//printf("first\n");
	//printf(" img = %d x %d \n", A.cols, A.rows);
	mat_to_array(A,img_mat);//, img_mat);
}
