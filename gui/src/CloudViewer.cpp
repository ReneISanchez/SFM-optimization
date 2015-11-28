/*
 * CloudViewer.cpp
 *
 *  Created on: Nov 20, 2015
 *      Author: qkgautier
 */

#include "CloudViewer.h"



using namespace std;



namespace sfm_gui {

//---------------------------------------------------
CloudViewer::CloudViewer()
//---------------------------------------------------
{
	// TODO Auto-generated constructor stub

}

//---------------------------------------------------
CloudViewer::~CloudViewer()
//---------------------------------------------------
{
	// TODO Auto-generated destructor stub
}


//---------------------------------------------------
void CloudViewer::launchViewer(
		const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud
		)
//---------------------------------------------------
{
    viewer_ = boost::make_shared<pcl::visualization::CloudViewer>("Cloud Viewer");

    //blocks until the cloud is actually rendered
    viewer_->showCloud(cloud);

    viewer_->runOnVisualizationThreadOnce(viewerOneOff);
    viewer_->runOnVisualizationThread (viewerPsycho);
}

//---------------------------------------------------
void CloudViewer::launchViewer(const std::string& filename)
//---------------------------------------------------
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZRGB>() );

    int status = pcl::io::loadPCDFile (filename, *cloud);

    if(status == -1)
    {
    	cerr << "Unable to read the file \"" << filename << "\"" << endl;
    }

    launchViewer(cloud);
}


//---------------------------------------------------
void CloudViewer::viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
//---------------------------------------------------
{
    viewer.setBackgroundColor (0.0, 0.0, 0.0);
}

//---------------------------------------------------
void CloudViewer::viewerPsycho (pcl::visualization::PCLVisualizer& viewer)
//---------------------------------------------------
{
    viewer.setCameraClipDistances(0.0001, 100000);
}









} /* namespace sfm_gui */
