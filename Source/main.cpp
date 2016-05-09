/*
 *  main.cpp
 *  SfMToyUI
 *
 *  Created by Roy Shilkrot on 4/27/12.
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

#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>

#include "MultiCameraPnP.h"

#include <sched.h>
#include <unistd.h>
#include <sys/types.h>
#include <arrayfire.h>

#define CPU_N 0 // affinity process

using namespace std;
using namespace cv;

#ifdef HAVE_OPENCV_GPU
#include <opencv2/gpu/gpu.hpp>
#endif

using namespace af;

std::vector<cv::Mat> images;
std::vector<std::string> images_names;

void fix_affinity()
{
    int error=0;
    cpu_set_t mask;
    /*  mask init  */
    CPU_ZERO(&mask);
    /* add CPU_N to  the mask */
    CPU_SET(CPU_N,&mask);

    /**
      test root access
     **/

    if(getuid()==0)
    {
        /*change affinity of process */
        error=sched_setaffinity(0,sizeof(cpu_set_t),&mask);
    }
    else
    {
        printf("must be root to change affinity\n");
    }
    if(error<0)
    {
        printf("sched_setaffinity() failed \n");
    }

}
//---------------------------- Using command-line ----------------------------

int main(int argc, char** argv) 
{
	//Mat A;
	//A = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	//array zeros = constant(0,3);	
	//array img = loadImageNative(argv[1]);
	open_imgs_dir(argv[1], images, images_names, 1.0);
	//exit(1);


    fix_affinity();
    if (argc < 2) 
    {
        cerr << "USAGE: " << argv[0]
            << " <path_to_images> [use rich features (RICH/OF) = RICH] [use GPU (GPU/CPU) = GPU] [downscale factor = 1.0]"
            << endl;
        return 0;
    }


   // double downscale_factor = 1.0;
/*    if (argc >= 5) 
    {
        downscale_factor = atof(argv[4]);
    }*/

    //open_imgs_dir(argv[1], images, images_names, downscale_factor);
    if (images.size() == 0) 
    {
        cerr << "[Error] can't get image files" << endl;
        return 1;
    }

    cv::Ptr < MultiCameraPnP > distance = new MultiCameraPnP(images,
            images_names, string(argv[1]));
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

    distance->RecoverDepthFromImages();

    vector < cv::Point3d > cv_pc = distance->getPointCloud();
    vector < cv::Vec3b > cv_pc_rgb = distance->getPointCloudRGB();

    // Super hacky PCD save
    string file_name = "sfm_pcl.pcd";
    ofstream fs;
    fs.open(file_name.c_str());

    fs << "# .PCD v0.7 - Point Cloud Data file format" << endl;
    fs << "VERSION 0.7" << endl;
    fs << "FIELDS x y z rgba" << endl;
    fs << "SIZE 4 4 4 4" << endl;
    fs << "TYPE F F F U" << endl;
    fs << "COUNT 1 1 1 1" << endl;
    fs << "WIDTH " << cv_pc.size() << endl;
    fs << "HEIGHT 1" << endl;
    fs << "VIEWPOINT 0 0 0 1 0 0 0" << endl;
    fs << "POINTS " << cv_pc.size() << endl;
    fs << "DATA ascii" << endl;

    for (size_t i = 0; i < cv_pc.size(); ++i) 
    {
        uint32_t rgba = ((uint32_t) cv_pc_rgb[i][0] << 16)
            | ((uint32_t) cv_pc_rgb[i][1] << 8)
            | ((uint32_t) cv_pc_rgb[i][2]);
        fs << cv_pc[i].x << " " << cv_pc[i].y << " " << cv_pc[i].z << " "
            << rgba << endl;
    }

    cerr << "Saved " << cv_pc.size() << " data points to .pcd." << endl;
}

