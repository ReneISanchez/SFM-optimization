/*
*  MultiCameraPnP.h
*  SfMToyExample
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
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

#include "Triangulation.h"
#include "IFeatureMatcher.h"
#include "FindCameraMatrices.h"

#include "Common.h"
#include "SfMUpdateListener.h"

class MultiCameraPnP
{
protected:
	std::vector<std::vector<cv::KeyPoint> > imgpts;
	std::vector<std::vector<cv::KeyPoint> > fullpts;
	std::vector<std::vector<cv::KeyPoint> > imgpts_good;

	std::map<std::pair<int,int> ,std::vector<cv::DMatch> > matches_matrix;
	
	std::vector<cv::Mat_<cv::Vec3b> > imgs_orig;
	std::vector<cv::Mat> imgs;
	std::vector<std::string> imgs_names;
	
	std::map<int,cv::Matx34d> Pmats;

	cv::Mat K;
	cv::Mat_<double> Kinv;
	
	cv::Mat cam_matrix,distortion_coeff;
	cv::Mat distcoeff_32f; 
	cv::Mat K_32f;

	std::vector<CloudPoint> pcloud;
	std::vector<cv::Vec3b> pointCloudRGB;
	std::vector<cv::KeyPoint> correspImg1Pt; //TODO: remove
	
	cv::Ptr<IFeatureMatcher> feature_matcher;
	
	bool features_matched;
	
	std::vector<CloudPoint> pointcloud_beforeBA;
	std::vector<cv::Vec3b> pointCloudRGB_beforeBA;

public:
	MultiCameraPnP(
		const std::vector<cv::Mat>& imgs_, 
		const std::vector<std::string>& imgs_names_, 
		const std::string& imgs_path_):
			imgs_names(imgs_names_),
			features_matched(false),
			use_rich_features(true),
			use_gpu(false)
    {
		setImages(imgs_,imgs_names_,imgs_path_);
    }

	MultiCameraPnP():features_matched(false),use_rich_features(true),use_gpu(false)
	{
	}
	
	void RecoverDepthFromImages();

	void GetRGBForPointCloud(
		const std::vector<struct CloudPoint>& pcloud,
		std::vector<cv::Vec3b>& RGBforCloud
		);

	void setImages(const std::vector<cv::Mat>& imgs_,
			const std::vector<std::string>& imgs_names_,
			const std::string& imgs_path_);
	
	void init(const std::string& imgs_path_);

	void OnlyMatchFeatures();

	std::vector<cv::Point3d> getPointCloudBeforeBA()
	{
		return CloudPointsToPoints(pointcloud_beforeBA);
	}
	
	const std::vector<cv::Vec3b>& getPointCloudRGBBeforeBA()
	{
		return pointCloudRGB_beforeBA;
	}

	std::vector<cv::Point3d> getPointCloud()
	{
		return CloudPointsToPoints(pcloud);
	}
	
	const cv::Mat& get_im_orig(int frame_num)
	{
		return imgs_orig[frame_num];
	}
	
	const std::vector<cv::KeyPoint>& getcorrespImg1Pt()
	{
		return correspImg1Pt;
	}
	
	const std::vector<cv::Vec3b>& getPointCloudRGB()
	{
		if (pointCloudRGB.size() == 0)
		{
			GetRGBForPointCloud(pcloud, pointCloudRGB);
		}
		return pointCloudRGB;
	}
	
	std::vector<cv::Matx34d> getCameras()
	{
		std::vector<cv::Matx34d> v; 
		for (std::map<int ,cv::Matx34d>::const_iterator it = Pmats.begin(); it != Pmats.end(); ++it)
		{
			v.push_back(it->second);
		}
		return v;
    }

    // Options
	bool use_rich_features;
	bool use_gpu;
	bool bInitialized;

private:
	void PruneMatchesBasedOnF();
	
	void AdjustCurrentBundle();
	
	void GetBaseLineTriangulation();
	
	void Find2D3DCorrespondences(int working_view, 
		std::vector<cv::Point3f>& ppcloud, 
		std::vector<cv::Point2f>& imgPoints);
	
	bool FindPoseEstimation(
		int working_view,
		cv::Mat_<double>& rvec,
		cv::Mat_<double>& t,
		cv::Mat_<double>& R,
		std::vector<cv::Point3f> ppcloud,
		std::vector<cv::Point2f> imgPoints);
	
	bool TriangulatePointsBetweenViews(
		unsigned int working_view, 
		unsigned int second_view,
		std::vector<struct CloudPoint>& new_triangulated,
		std::vector<int>& add_to_cloud
		);
	
	int FindHomographyInliers2Views(int vi, int vj);
	
	int m_first_view;
	int m_second_view; //baseline's second view other to 0
	std::set<int> done_views;
	std::set<int> good_views;
	
/********** Subject / Observer **********/
	std::vector < SfMUpdateListener * > listeners;
public:
	
    void attach(SfMUpdateListener *sul)
    {
        listeners.push_back(sul);
    }
	
private:
    void update()
    {
        for (unsigned int i = 0; i < listeners.size(); i++)
		{
			listeners[i]->update(getPointCloud(),
								 getPointCloudRGB(),
								 getPointCloudBeforeBA(),
								 getPointCloudRGBBeforeBA(),
								 getCameras());
		}
    }
};

