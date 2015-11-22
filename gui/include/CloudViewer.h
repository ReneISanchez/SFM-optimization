/*
 * CloudViewer.h
 *
 *  Created on: Nov 20, 2015
 *      Author: qkgautier
 */

#ifndef GUI_INCLUDE_CLOUDVIEWER_H_
#define GUI_INCLUDE_CLOUDVIEWER_H_

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include <boost/make_shared.hpp>


namespace sfm_gui {

//---------------------------------------------------
class CloudViewer
//---------------------------------------------------
{


	//**************************************
	// Public
	//**************************************
public:

	//---------------------------------------------------
	/// Default constructor
	//---------------------------------------------------
	CloudViewer();

	//---------------------------------------------------
	/// Destructor
	//---------------------------------------------------
	virtual ~CloudViewer();


	//---------------------------------------------------
	/**
	 * Create a new instance of the visualizer and show the given cloud.
	 * The window runs on its own thread and this function returns after it is created.
	 *
	 * @note The visualizer only works with XYZRGB cloud.
	 */
	//---------------------------------------------------
	void launchViewer(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud);

	//---------------------------------------------------
	/**
	 * Load a XYZRGB cloud from the specified cloud and run the viewer.
	 *
	 * @see launchViewer(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
	 */
	//---------------------------------------------------
	void launchViewer(const std::string& filename);


	//---------------------------------------------------
	/// Check if the window was closed by the user.
	//---------------------------------------------------
	bool wasStopped(int millis_to_wait = 1){ return viewer_->wasStopped(millis_to_wait); }


	//---------------------------------------------------
	/**
	 * Spin wait until the window is closed.
	 * You can specify a duration to sleep between each spin.
	 */
	//---------------------------------------------------
	void waitUntilStopped(int64_t sleepMs = 1000)
	{
		while(!this->wasStopped())
		{ boost::this_thread::sleep(boost::posix_time::milliseconds(sleepMs)); }
	}


	//**************************************
	// Protected
	//**************************************
protected:

	static void viewerOneOff (pcl::visualization::PCLVisualizer& viewer);
	static void viewerPsycho (pcl::visualization::PCLVisualizer& viewer);


	//**************************************
	// Class members
	//**************************************
protected:

	boost::shared_ptr<pcl::visualization::CloudViewer> viewer_;

};

} /* namespace sfm_gui */

#endif /* GUI_INCLUDE_CLOUDVIEWER_H_ */
