#include <iostream>

#include "MultiCameraPnP.h"
#include "CloudViewer.h"


#include <fstream>
#include <string.h>
#include <vector>
#include <sched.h>
#include <unistd.h>
#include <sys/types.h>
#include <numeric>

#ifdef HAVE_OPENCV_GPU
#include <opencv2/gpu/gpu.hpp>
#endif

#include <pcl/point_cloud.h>
#include <Eigen/Core>



#define CPU_N 0 // affinity process

using namespace std;
using namespace cv;
using namespace sfm_gui;
using namespace pcl;




vector<cv::Mat> images;
vector<string> images_names;



int main(int argc, char** argv) 
{
//	{
//		CloudViewer viewer;
//		viewer.launchViewer("sfm_pcl.pcd");
//		viewer.waitUntilStopped();
//		return 0;
//	}



    if (argc < 2) 
    {
        cerr << "USAGE: " << argv[0]
            << " <path_to_images> [use rich features (RICH/OF) = RICH] [use GPU (GPU/CPU) = GPU] [downscale factor = 1.0]"
            << endl;
        return 0;
    }

    double downscale_factor = 1.0;
    if (argc >= 5) 
    {
        downscale_factor = atof(argv[4]);
    }

    open_imgs_dir(argv[1], images, images_names, downscale_factor);
    if (images.size() == 0) 
    {
        cerr << "[Error] can't get image files" << endl;
        return 1;
    }

    cv::Ptr < MultiCameraPnP > distance =
		new MultiCameraPnP(images, images_names, string(argv[1]));

    if (argc < 3) 
    {
        distance->use_rich_features = true;
    } 
    else 
    {
        distance->use_rich_features = (strcmp(argv[2], "RICH") == 0);
    }

#ifdef HAVE_OPENCV_GPU

    if (argc < 4)
    {
        distance->use_gpu = (cv::gpu::getCudaEnabledDeviceCount() > 0);
    }
    else
    {
        distance->use_gpu = (strcmp(argv[3], "GPU") == 0);
    }

#else

    distance->use_gpu = false;

#endif

    // Do SfM
    distance->RecoverDepthFromImages();

    // Get result
    vector< cv::Point3d > cv_pc     = distance->getPointCloud();
    vector< cv::Vec3b >   cv_pc_rgb = distance->getPointCloudRGB();

    // TODO why not the same size?
    if(cv_pc.size() != cv_pc_rgb.size())
    {
    	cerr << "Coordinates and colors don't have the same size! ("
    			<< cv_pc.size() << " and " << cv_pc_rgb.size() << ")" << endl;
    	//return EXIT_FAILURE;
    }


    cv::Point3d centroid(0.0, 0.0, 0.0);
//    for (size_t i = 0; i < cv_pc.size(); ++i)
//    {
//    	centroid.x += cv_pc[i].x;
//    	centroid.y += cv_pc[i].y;
//    	centroid.z += cv_pc[i].z;
//    }
//    centroid.x /= cv_pc.size();
//    centroid.y /= cv_pc.size();
//    centroid.z /= cv_pc.size();









    //vector<double> distances(cv_pc.size());
    //double mean = 0.0;
	//unsigned int numPoints = 0;
    //for (size_t i = 0; i < cv_pc.size(); ++i)
    //{
	//	const double& x = cv_pc[i].x;
	//	const double& y = cv_pc[i].y;
	//	const double& z = cv_pc[i].z;

	//	if(std::isnan(x) || std::isnan(y) || std::isnan(z)
	//			|| std::isinf(x) || std::isinf(y) || std::isinf(z))
	//	{
	//		cout << x << "  " << y << "  " << z << endl;
	//		continue;
	//	}

    //	distances[i] = sqrt(
    //			pow(x, 2) +
	//			pow(y, 2) +
	//			pow(z, 2)
	//			);

    //	mean += distances[i];

    //	centroid.x += x;
    //	centroid.y += y;
    //	centroid.z += z;

	//	numPoints++;
    //}

    //mean /= numPoints;
    //double sq_sum = std::inner_product(distances.begin(), distances.end(), distances.begin(), 0.0);
    //double stdev = std::sqrt(sq_sum / distances.size() - mean * mean);

    //centroid.x /= numPoints;
    //centroid.y /= numPoints;
    //centroid.z /= numPoints;

    //cout << "Mean dist: " << mean << "  Stdev: " << stdev << endl;
    //cout << "Centroid: " << centroid.x << " " << centroid.y << " " << centroid.z << endl;





    // Convert result to PCL format

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_pc(new PointCloud<PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_debug(new PointCloud<PointXYZ>);

    for (size_t i = 0; i < cv_pc.size(); ++i)
    {
		const double& x = cv_pc[i].x;
		const double& y = cv_pc[i].y;
		const double& z = cv_pc[i].z;

		if(std::isnan(x) || std::isnan(y) || std::isnan(z)
				|| std::isinf(x) || std::isinf(y) || std::isinf(z)
				|| max(max(x,y),z) > 10000 || min(min(x,y),z) < -10000)
		{
			continue;
		}


    	PointXYZRGB p(cv_pc_rgb[i][0], cv_pc_rgb[i][1], cv_pc_rgb[i][2]);
    	p.x = x - centroid.x;
    	p.y = y - centroid.y;
    	p.z = z - centroid.z;

    	//if(distances[i] < 3 * stdev)
    	{
    		pcl_pc->push_back(p);
    	}

		PointXYZ p_debug;
		p_debug.x = x;
		p_debug.y = y;
		p_debug.z = z;
		pcl_debug->push_back(p_debug);
    }

//    Eigen::Matrix< Scalar, 4, 1 > centroid;
//    pcl::compute3DCentroid(*pcl_pc, centroid);



    // Save result to file
    pcl::io::savePCDFileASCII("sfm_debug.pcd", *pcl_debug);
    pcl::io::savePCDFileASCII("sfm_pcl.pcd", *pcl_pc);
    cout << "Saved " << pcl_pc->size() << " data points to pcd file." << endl;


    // Visualize results
	CloudViewer viewer;
	viewer.launchViewer(pcl_pc);
	viewer.waitUntilStopped();



    return EXIT_SUCCESS;
}





 





