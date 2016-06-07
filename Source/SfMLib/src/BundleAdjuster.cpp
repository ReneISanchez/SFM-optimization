/*
*  BundleAdjuster.cpp
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 5/1812.
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2013 Roy Shilkrot
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 */

#include "BundleAdjuster.h"
#include "Common.h"
#include <pthread.h>
#include <mutex>
#include <unistd.h>
#include <time.h>
#include <fstream>
#include <chrono>
#include <arrayfire.h>

using namespace af;
using namespace cv;
using namespace std;

#ifndef HAVE_SSBA
#include <opencv2/contrib/contrib.hpp>
#endif

std::mutex mtx;
std::mutex mtx3;
std::mutex mtx2;
//int local_cam_count = 0;
vector< vector<Point2d> > imagePoints;
vector< vector<int> > visibility;
vector<CloudPoint> pointcloud2;
vector<int> global_cam_id_to_local_id;
vector<int> local_cam_id_to_global_id;
map<int, cv::Matx34d> Pmats2;
vector< vector<cv::KeyPoint> > imgpts2;
cv::Mat_<double> cam_matrix2;

#ifdef HAVE_SSBA
#define V3DLIB_ENABLE_SUITESPARSE

#include "../3rdparty/SSBA-3.0/Math/v3d_linear.h"
#include "../3rdparty/SSBA-3.0/Base/v3d_vrmlio.h"
#include "../3rdparty/SSBA-3.0/Geometry/v3d_metricbundle.h"

using namespace V3D;
//std::mutex mtx;
//pthread_mutex_t mtx;
//int local_cam_count = 0;

/* HOW TO USE mtx:
 *Inside of a multi-thread function:
  mtx.lock() 
  [Critical section]
  mtx.unlock()
*/
namespace
{AAAAAA

	inline void
	showErrorStatistics(double const f0,
			StdDistortionFunction const& distortion,
			vector<CameraMatrix> const& cams,
			vector<Vector3d> const& Xs,
			vector<Vector2d> const& measurements,
			vector<int> const& correspondingView,
			vector<int> const& correspondingPoint)
	{
		int const K = measurements.size();

		double meanReprojectionError = 0.0;
		for (int k = 0; k < K; ++k)
		{
			int const i = correspondingView[k];
			int const j = correspondingPoint[k];
			Vector2d p = cams[i].projectPoint(distortion, Xs[j]);

			double reprojectionError = norm_L2(f0 * (p - measurements[k]));
			meanReprojectionError += reprojectionError;

//			cout << "i=" << i << " j=" << j << " k=" << k << "\n";
//			 displayVector(Xs[j]);
//			 displayVector(f0*p);
//			 displayVector(f0*measurements[k]);
//			 displayMatrix(cams[i].getRotation());
//			 displayVector(cams[i].getTranslation());
//			 cout << "##################### error = " << reprojectionError << "\n";
//			 if (reprojectionError > 2)
//             cout << "!\n";
		}
		cout << "mean reprojection error (in pixels): " << meanReprojectionError/K << endl;
	}
} // end namespace <>

#endif

/********************************************************************/
/**********************  ALL THREAD STUFF GOES HERE *****************/
/********************************************************************/

//Struct that we pass in to threadFunc1 and threadFunc2. Contains copies of the parameters that are specific to each 
//unrolled for loop of the bundle adjustment initialization process.
typedef struct 
{
	int thread_num;

	int pt3d_img;  //inner loop
	int pt3d;      //outer loop
} threadTuple;

//Used to replace 1st inner for-loop of what used to be the Bundle Adjustment initialization
//This functions is executed in parallel on the CPU.
void* threadFunc1(void* thread) 
{
	int local_cam_count = 0;
	threadTuple *t = (threadTuple*) thread;  //get the struct with the specific parameters for this loop


	if ((pointcloud2[t->pt3d]).imgpt_for_img[t->pt3d_img] >= 0)
	{
	    if (global_cam_id_to_local_id[t->pt3d_img] < 0)
		{
		 	mtx.lock();		//mutual exclusion lock. Only allow 1 thread to execute the following code
			local_cam_id_to_global_id[local_cam_count] = t->pt3d_img;
			global_cam_id_to_local_id[t->pt3d_img] = local_cam_count++;
			mtx.unlock();	//mutual exclusion unlock
		}

		int local_cam_id = global_cam_id_to_local_id[t->pt3d_img];

		//2d point
		Point2d pt2d_for_pt3d_in_img = 
			imgpts2[t->pt3d_img][pointcloud2[t->pt3d].imgpt_for_img[t->pt3d_img]].pt;

		imagePoints[local_cam_id][t->pt3d] = pt2d_for_pt3d_in_img;
		visibility[local_cam_id][t->pt3d] = 1;

	}
}

//Used to to replace 2nd inner for-loop of what used to be the Bundle Adjustment initialization.
//This function is executed in parallel on the CPU, and it will also spawn GPU threads for matrix
//multiplication through the use of Arrayfire.
void* threadFunc2(void* thread)
{

	threadTuple *t = (threadTuple*) thread;

	if ((pointcloud2[t->pt3d]).imgpt_for_img[t->pt3d_img] < 0)
	{
		vector<int>::iterator local_it = find(local_cam_id_to_global_id.begin(),
				local_cam_id_to_global_id.end(), t->pt3d_img);

		if(local_it != local_cam_id_to_global_id.end())
		{
			int local_id = local_it - local_cam_id_to_global_id.begin();

			if (local_id >= 0)
			{

				//Create a float array of the pointcloud so that we can create an Arrayfire array so we can feed it to Arrayfire as a matmul parameter
				float X_ptr[] = {(float)pointcloud2[t->pt3d].pt.x, (float)pointcloud2[t->pt3d].pt.y, (float) pointcloud2[t->pt3d].pt.z,1.0};
				af::array X(4,1, X_ptr);

				//Create a float array of the pmat so that we can create an Arrayfire array so we can feed it to Arrayfire as a matmul parameter	
				float P_ptr[] = {Pmats2[t->pt3d_img](0,0), Pmats2[t->pt3d_img](0,1), Pmats2[t->pt3d_img](0,2), 
					Pmats2[t->pt3d_img](0,3), Pmats2[t->pt3d_img](0,4), Pmats2[t->pt3d_img](1,0), Pmats2[t->pt3d_img](1,1),
					Pmats2[t->pt3d_img](1,2), Pmats2[t->pt3d_img](1,3), Pmats2[t->pt3d_img](1,4), Pmats2[t->pt3d_img](2,0),
					Pmats2[t->pt3d_img](2,1), Pmats2[t->pt3d_img](2,2), Pmats2[t->pt3d_img](2,3), Pmats2[t->pt3d_img](2,4),
					Pmats2[t->pt3d_img](3,0), Pmats2[t->pt3d_img](3,1), Pmats2[t->pt3d_img](3,2), Pmats2[t->pt3d_img](3,3),
					Pmats2[t->pt3d_img](3,4)};
				af::array P(3,4,P_ptr);

				//Same as above ^^
				float cam_ptr[] = {(float) cam_matrix2(0,0),(float) cam_matrix2(0,1),(float) cam_matrix2(0,2),
									(float) cam_matrix2(1,0),(float) cam_matrix2(1,1),(float) cam_matrix2(1,2)};			
				af::array cam(1, 2, cam_ptr);
	
				//Arrayfire matrix multiplications
				af::array KP = matmul(cam, P);
				af::array xPt_img = matmul(KP,X);

				//Now we need to extract the data from the GPU (GPU -> CPU)
				//Create a float array
				float *host_a = new float[4];

				//Have the resulting matrix have its data sent over
				xPt_img.host(host_a);

				Point2d xPt_img_(host_a[0] / host_a[2], 
						host_a[1] / host_a[2]);

				imagePoints[local_id][t->pt3d] = xPt_img_;

				visibility[local_id][t->pt3d] = 0;
    			cout << "Set visibility" << endl;
				delete[] host_a; //free up
			}
		}
	}

}

//Global bool used for knowing when all threads have finished executing
int t_finished = 0;

//Function that executes the two threadFuncs sequentially.
void* matMulThread(void* thread){

	threadFunc1(thread);
	threadFunc2(thread);

	t_finished++;
}

//
int create_matmul_threads(int numThreads, int num_global_cams, int point_cloud_size)
{
	int i,j,k;

	//Create the thread identifiers
	pthread_t threads[numThreads];

	//Create the thread tuples that will be passed into the thread function
	threadTuple* t = new threadTuple[numThreads];

	//Initialize threads tuples  
	int t_num = 1;
	int t_copy = 0;
	int prev = 0;
	int old_num = -1;
	
	//Initialize global variables
	for(i = 0; i < point_cloud_size; i++){
		for(j = 0; j < num_global_cams; j++){
			t_num = i*num_global_cams + j;
			old_num = t_num;
			t[t_num].thread_num = t_num;
			t[t_num].pt3d = i;			//first loop
			t[t_num].pt3d_img = j;		//second loop

			t_copy++;	

			//Execute the threads in batches of 1000, in order to not get killed by the OS
			if(t_copy >  999){
				for(k = prev; k <= t_num; k++){
					//Create a thread. Thread will execute matMulThread with t[k] as parameter
					pthread_create(&threads[k], NULL, matMulThread, (void*) &t[k]);
				}

				while(t_finished < t_num)
				{
				//sleep(1);
				//cout << "In while loop: " << t_finished << "/" << t_num << endl;
				} 
		
				prev = t_num;
				t_copy = 0;
//				cout << "Finished thread batch" << endl;
			}
		
		}
	}
	

	//Run any remaining threads
	if(t_copy > 0){
//		cout << "Last thread creation" << endl;
		for( k = prev; k < t_num; k++){
			pthread_create(&threads[k], NULL, matMulThread, (void*) &t[k]);
		}
	
	}

	//Wait until all threads are finished
	while(t_finished < point_cloud_size*num_global_cams){
		sleep(1);
	}

	return 0;
}


/******************************************************************/
/*********************  END OF THREAD STUFF  **********************/
/******************************************************************/



//count number of 2D measurements
int BundleAdjuster::Count2DMeasurements(const vector<CloudPoint>& pointcloud) 
{
	int K = 0;
	for (unsigned int i = 0; i < pointcloud.size(); i++) 
	{
		for (unsigned int ii = 0; ii < pointcloud[i].imgpt_for_img.size(); ii++) 
		{
			if (pointcloud[i].imgpt_for_img[ii] >= 0) 
			{
				K++;
			}
		}
	}
	return K;
}

void BundleAdjuster::adjustBundle(vector<CloudPoint>& pointcloud,
		Mat& cam_matrix, vector<vector<cv::KeyPoint> >& imgpts,
		map<int, cv::Matx34d>& Pmats) 
{
	int N = Pmats.size(), M = pointcloud.size(), K = Count2DMeasurements(pointcloud);

	cout << "N (cams) = " << N << " M (points) = " << M
			<< " K (measurements) = " << K << endl;

#ifdef HAVE_SSBA
	/********************************************************************************/
	/*	Use SSBA-3.0 for sparse bundle adjustment									*/
	/********************************************************************************/

	StdDistortionFunction distortion;

	//conver camera intrinsics to BA datastructs
	Matrix3x3d KMat;
	makeIdentityMatrix(KMat);
	//cout << "+1 simple mat declaration" << endl;
	KMat[0][0] = cam_matrix.at<double>(0,0);//fx
	KMat[1][1] = cam_matrix.at<double>(1,1);//fy
	KMat[0][1] = cam_matrix.at<double>(0,1);//skew
	KMat[0][2] = cam_matrix.at<double>(0,2);//ppx
	KMat[1][2] = cam_matrix.at<double>(1,2);//ppy

	double const f0 = KMat[0][0];
	//cout << "intrinsic before bundle = "; displayMatrix(KMat);
	Matrix3x3d Knorm = KMat;
	// Normalize the intrinsic to have unit focal length.
	scaleMatrixIP(1.0/f0, Knorm);
	Knorm[2][2] = 1.0;

	vector<int> pointIdFwdMap(M);
	map<int, int> pointIdBwdMap;

	//conver 3D point cloud to BA datastructs
	vector<Vector3d > Xs(M);
	for (int j = 0; j < M; ++j)
	{
		int pointId = j;
		Xs[j][0] = pointcloud[j].pt.x;
		Xs[j][1] = pointcloud[j].pt.y;
		Xs[j][2] = pointcloud[j].pt.z;
		pointIdFwdMap[j] = pointId;
		pointIdBwdMap.insert(make_pair(pointId, j));
	}
	//cout << "Read the 3D points." << endl;

	vector<int> camIdFwdMap(N,-1);
	map<int, int> camIdBwdMap;

	//convert cameras to BA datastructs
	vector<CameraMatrix> cams(N);
	for (int i = 0; i < N; ++i)
	{
		int camId = i;
		Matrix3x3d R;
		Vector3d T;

		Matx34d& P = Pmats[i];

		R[0][0] = P(0,0); R[0][1] = P(0,1); R[0][2] = P(0,2); T[0] = P(0,3);
		R[1][0] = P(1,0); R[1][1] = P(1,1); R[1][2] = P(1,2); T[1] = P(1,3);
		R[2][0] = P(2,0); R[2][1] = P(2,1); R[2][2] = P(2,2); T[2] = P(2,3);

		camIdFwdMap[i] = camId;
		camIdBwdMap.insert(make_pair(camId, i));

		cams[i].setIntrinsic(Knorm);
		cams[i].setRotation(R);
		cams[i].setTranslation(T);
	}
	//cout << "Read the cameras." << endl;

	vector<Vector2d > measurements;
	vector<int> correspondingView;
	vector<int> correspondingPoint;

	measurements.reserve(K);
	correspondingView.reserve(K);
	correspondingPoint.reserve(K);

	//convert 2D measurements to BA datastructs
	for (unsigned int k = 0; k < pointcloud.size(); ++k)
	{
		for (unsigned int i = 0; i < pointcloud[k].imgpt_for_img.size(); i++)
		{
			if (pointcloud[k].imgpt_for_img[i] >= 0)
			{
				int view = i, point = k;
				Vector3d p, np;

				Point cvp = imgpts[i][pointcloud[k].imgpt_for_img[i]].pt;
				p[0] = cvp.x;
				p[1] = cvp.y;
				p[2] = 1.0;

				if (camIdBwdMap.find(view) != camIdBwdMap.end() &&
						pointIdBwdMap.find(point) != pointIdBwdMap.end())
				{
					// Normalize the measurements to match the unit focal length.
					scaleVectorIP(1.0/f0, p);
					measurements.push_back(Vector2d(p[0], p[1]));
					correspondingView.push_back(camIdBwdMap[view]);
					correspondingPoint.push_back(pointIdBwdMap[point]);
				}
			}
		}
	} // end for (k)

	K = measurements.size();

//	cout << "Read " << K << " valid 2D measurements." << endl;

	showErrorStatistics(f0, distortion, cams, Xs, measurements, correspondingView, correspondingPoint);

//	V3D::optimizerVerbosenessLevel = 1;
	double const inlierThreshold = 2.0 / fabs(f0);

	Matrix3x3d K0 = cams[0].getIntrinsic();
	//cout << "K0 = "; displayMatrix(K0);

	bool good_adjustment = false;
	{
		ScopedBundleExtrinsicNormalizer extNorm(cams, Xs);
		ScopedBundleIntrinsicNormalizer intNorm(cams,measurements,correspondingView);
		CommonInternalsMetricBundleOptimizer opt(V3D::FULL_BUNDLE_FOCAL_LENGTH_PP, inlierThreshold, K0, distortion, cams, Xs,
				measurements, correspondingView, correspondingPoint);
//		StdMetricBundleOptimizer opt(inlierThreshold,cams,Xs,measurements,correspondingView,correspondingPoint);

		opt.tau = 1e-3;
		opt.maxIterations = 50;
		opt.minimize();

		//cout << "optimizer status = " << opt.status << endl;

		good_adjustment = (opt.status != 2);
	}

	//cout << "refined K = "; displayMatrix(K0);

	for (int i = 0; i < N; ++i)
	{
		cams[i].setIntrinsic(K0);
	}

	Matrix3x3d Knew = K0;
	scaleMatrixIP(f0, Knew);
	Knew[2][2] = 1.0;
	//cout << "Knew = "; displayMatrix(Knew);

	showErrorStatistics(f0, distortion, cams, Xs, measurements, correspondingView, correspondingPoint);

	if (good_adjustment)
	{ //good adjustment?

		//Vector3d mean(0.0, 0.0, 0.0);
		//for (unsigned int j = 0; j < Xs.size(); ++j) addVectorsIP(Xs[j], mean);
		//scaleVectorIP(1.0/Xs.size(), mean);
		//
		//vector<float> norms(Xs.size());
		//for (unsigned int j = 0; j < Xs.size(); ++j)
		//	norms[j] = distance_L2(Xs[j], mean);
		//
		//sort(norms.begin(), norms.end());
		//float distThr = norms[int(norms.size() * 0.9f)];
		//cout << "90% quantile distance: " << distThr << endl;

		//extract 3D points
		for (unsigned int j = 0; j < Xs.size(); ++j)
		{
			//if (distance_L2(Xs[j], mean) > 3*distThr) makeZeroVector(Xs[j]);

			pointcloud[j].pt.x = Xs[j][0];
			pointcloud[j].pt.y = Xs[j][1];
			pointcloud[j].pt.z = Xs[j][2];
		}

		//extract adjusted cameras
		for (int i = 0; i < N; ++i)
		{
			Matrix3x3d R = cams[i].getRotation();
			Vector3d T = cams[i].getTranslation();

			Matx34d P;
			P(0,0) = R[0][0]; P(0,1) = R[0][1]; P(0,2) = R[0][2]; P(0,3) = T[0];
			P(1,0) = R[1][0]; P(1,1) = R[1][1]; P(1,2) = R[1][2]; P(1,3) = T[1];
			P(2,0) = R[2][0]; P(2,1) = R[2][1]; P(2,2) = R[2][2]; P(2,3) = T[2];

			Pmats[i] = P;
		}

		//TODO: extract camera intrinsics
		cam_matrix.at<double>(0,0) = Knew[0][0];
		cam_matrix.at<double>(0,1) = Knew[0][1];
		cam_matrix.at<double>(0,2) = Knew[0][2];
		cam_matrix.at<double>(1,1) = Knew[1][1];
		cam_matrix.at<double>(1,2) = Knew[1][2];
	}
#else
	/********************************************************************************/
	/*	Use OpenCV contrib module for sparse bundle adjustment						*/
	/********************************************************************************/


	vector < Point3d > points(M);// positions of points in global coordinate system (input and output)
   
	// projections of 3d points for every camera
	imagePoints = vector<vector<Point2d>>(N, vector<Point2d>(M));

	// visibility of 3d points for every camera
	visibility = vector< vector<int> >(N, vector<int>(M));

	//We use a copy so that we don't modify the original
	pointcloud2 = pointcloud;

	vector < Mat > cameraMatrix(N);// intrinsic matrices of all cameras (input and output)
	vector < Mat > R(N);// rotation matrices of all cameras (input and output)
	vector < Mat > T(N);// translation vector of all cameras (input and output)
	vector < Mat > distCoeffs(0);// distortion coefficients of all cameras (input and output)

	int num_global_cams = pointcloud[0].imgpt_for_img.size();

	global_cam_id_to_local_id = vector<int>(num_global_cams,-1);
	local_cam_id_to_global_id = vector<int>(N,-1);

	//Use copies so that we don't modify the originals
	Pmats2 = Pmats;
	imgpts2 = imgpts;
	cam_matrix2 = cam_matrix;

	int total_t = num_global_cams*pointcloud.size();


	create_matmul_threads(total_t, num_global_cams, pointcloud.size());

	//The big double for-loop that we replaced with this^ function 
/*
	for (unsigned int pt3d = 0; pt3d < pointcloud.size(); pt3d++) 
	{
		points[pt3d] = pointcloud[pt3d].pt;

		for (int pt3d_img = 0; pt3d_img < num_global_cams; pt3d_img++) 
		{
			if (pointcloud[pt3d].imgpt_for_img[pt3d_img] >= 0) 
			{
				if (global_cam_id_to_local_id[pt3d_img] < 0) 
				{
					local_cam_id_to_global_id[local_cam_count] = pt3d_img;
					global_cam_id_to_local_id[pt3d_img] = local_cam_count++;
				}

				int local_cam_id = global_cam_id_to_local_id[pt3d_img];

				//2d point
				Point2d pt2d_for_pt3d_in_img =
						imgpts[pt3d_img][pointcloud[pt3d].imgpt_for_img[pt3d_img]].pt;
				cout << "---------------------------------" << endl;
				cout << "pt3d_img = " << pt3d_img << endl;
				cout << "local_cam_id = " << local_cam_id << endl;
				cout << "pt3d = " << pt3d << endl;
				imagePoints[local_cam_id][pt3d] = pt2d_for_pt3d_in_img;

				//visibility in this camera
				visibility[local_cam_id][pt3d] = 1;
			}
		}

		//2nd pass to mark not-founds
		for (int pt3d_img = 0; pt3d_img < num_global_cams; pt3d_img++) 
		{
			if (pointcloud[pt3d].imgpt_for_img[pt3d_img] < 0) 
			{
				//see if this global camera is being used locally in the BA
				vector<int>::iterator local_it = find(
						local_cam_id_to_global_id.begin(),
						local_cam_id_to_global_id.end(), pt3d_img);
				if (local_it != local_cam_id_to_global_id.end()) 
				{
					//this camera is used, and its local id is:
					int local_id = local_it - local_cam_id_to_global_id.begin();

					if (local_id >= 0) 
					{
						//reproject
						Mat_<double> X =
								(Mat_<double>(4, 1) << pointcloud[pt3d].pt.x, pointcloud[pt3d].pt.y, pointcloud[pt3d].pt.z, 1);
						Mat_<double> P(3, 4, Pmats[pt3d_img].val);
						Mat_<double> KP = cam_matrix * P;
						Mat_<double> xPt_img = KP * X;
						//cout << " +2 mul " << endl;
						/*cout << "Matrix P rows: " << P.rows << endl;
						cout << "Matrix P cols: " << P.cols << endl;

						cout << "Matrix cam_matrix rows: " << cam_matrix.rows << endl;
						cout << "Matrix cam_matrix cols: " << cam_matrix.cols << endl;

						cout << "Matrix X rows: " << X.rows << endl;
						cout << "Matrix X cols: " << X.cols << endl; */
/*					
						Point2d xPt_img_(xPt_img(0) / xPt_img(2),
								xPt_img(1) / xPt_img(2));

						imagePoints[local_id][pt3d] = xPt_img_; //TODO reproject point on this camera
						visibility[local_id][pt3d] = 0;
					}
				}
			}
		}
	}
*/
	//cout << "+" << 2*num_global_cams*2 << " 3x3 matrix muls" << endl;

	for (int i = 0; i < N; i++) 
	{
		//printf("i/N: %d/%d",i,N);
		cameraMatrix[i] = cam_matrix2;
		//cout << "cameraMatrix[i] = " << cam_matrix2 << endl;

		Matx34d& P = Pmats2[local_cam_id_to_global_id[i]];
		//cout << "local_cam_id_to_global_id[i] = " << local_cam_id_to_global_id[i] << endl;
		//cout << "Pmats = " << Pmats2[local_cam_id_to_global_id[i]] << endl;

		Mat_<double> camR(3, 3), camT(3, 1);
		camR(0, 0) = P(0, 0);
		camR(0, 1) = P(0, 1);
		camR(0, 2) = P(0, 2);
		camT(0) = P(0, 3);
		camR(1, 0) = P(1, 0);
		camR(1, 1) = P(1, 1);
		camR(1, 2) = P(1, 2);
		camT(1) = P(1, 3);
		camR(2, 0) = P(2, 0);
		camR(2, 1) = P(2, 1);
		camR(2, 2) = P(2, 2);
		camT(2) = P(2, 3);
		R[i] = camR;
		//cout << "R[i] = " << R[i] << endl;
		T[i] = camT;
		//cout << "T[i] = " << T[i] << endl;

	}

	//cout << "Adjust bundle... \n";
	cv::LevMarqSparse::bundleAdjust(points, imagePoints, visibility,
			cameraMatrix, R, T, distCoeffs);
	//cout << "DONE\n";

	//get the BAed points
	for (unsigned int pt3d = 0; pt3d < pointcloud2.size(); pt3d++) 
	{
		pointcloud2[pt3d].pt = points[pt3d];
	}

	//get the BAed cameras
	for (int i = 0; i < N; ++i) 
	{
		Matx34d P;
		P(0, 0) = R[i].at<double>(0, 0);
		P(0, 1) = R[i].at<double>(0, 1);
		P(0, 2) = R[i].at<double>(0, 2);
		P(0, 3) = T[i].at<double>(0);

		P(1, 0) = R[i].at<double>(1, 0);
		P(1, 1) = R[i].at<double>(1, 1);
		P(1, 2) = R[i].at<double>(1, 2);
		P(1, 3) = T[i].at<double>(1);

		P(2, 0) = R[i].at<double>(2, 0);
		P(2, 1) = R[i].at<double>(2, 1);
		P(2, 2) = R[i].at<double>(2, 2);
		P(2, 3) = T[i].at<double>(2);

		Pmats2[local_cam_id_to_global_id[i]] = P;
	}
	Pmats = Pmats2;
	cam_matrix = cam_matrix2;
	pointcloud = pointcloud2;

#endif
}
