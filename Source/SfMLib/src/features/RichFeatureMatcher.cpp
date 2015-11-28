/*
 *  RichFeatureMatcher.cpp
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 5/17/12.
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

#include "RichFeatureMatcher.h"
#include "FindCameraMatrices.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/nonfree/nonfree.hpp>

#include <iostream>
#include <set>


using namespace std;
using namespace cv;

static const unsigned char popCountTable[] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

static void getDescriptorDist(const cv::Mat& descriptors_1, 
        const cv::Mat& descriptors_2,
        std::vector<std::pair<int,int> >& minDist){
    for (int x = 0; x < descriptors_1.rows; x++)
    {
        int min = INT_MAX;
        int yIdx = 0;
        for (int y = 0 ; y < descriptors_2.rows ; y++)
        {
            int sumHamDist = 0;
            for (int z = 0; z < descriptors_1.cols; z++){
                sumHamDist += popCountTable[(descriptors_1.at<unsigned char>(x,z)) ^ (descriptors_2.at<unsigned char>(y,z))];
            }
            if (sumHamDist <= min)
            {
                min = sumHamDist;
                yIdx = y;
            }
        }
        minDist.push_back(std::make_pair(yIdx, min));
    }
} //--endGetDescriptorDist
static void verifyCrossCheck(std::vector<std::pair<int,int> >&minDist1,
        std::vector<std::pair<int,int> >&minDist2,
        vector < vector<DMatch> >& nn_matches){
    for (unsigned int x = 0; x < minDist1.size() && x < minDist2.size(); x++)
        // size cutoff at smaller vector
    {
        std::vector<cv::DMatch> tempVect;
        unsigned int tempDis1 = minDist2[ minDist1[x].first ].second;  // distances from <-
        unsigned int tempDis2 = minDist1[ minDist2[x].first ].second; // distances from ->
        if ( tempDis2 < tempDis1 )
        {
            /*
               --- minDistance1[x].second is the HamDist from checking all of descriptors_2
               against the first of descriptors_1, so we want the indices of the verified distance coming
               from descriptors_2.
               To locate it you only need which row it's in bc they all start at col 0 --- 
             */
            tempVect.push_back( cv::DMatch (x , minDist1[x].first , (float)tempDis2));
        }
        else
        {
            //tempVect.push_back(DMatch( minDistance1[x].first, x, (float)tempDis1) ); //orig
            tempVect.push_back(cv::DMatch( x, minDist1[x].first, (float)tempDis1) );
        }
        nn_matches.push_back(tempVect);
        tempVect.clear();
    }
} //endVerifyCrossCheck
/*---------------------------------------------------------------------------*/
//c'tor
RichFeatureMatcher::RichFeatureMatcher(std::vector<cv::Mat>& imgs_,
        std::vector<std::vector<cv::KeyPoint> >& imgpts_) :
    imgs(imgs_), imgpts(imgpts_) 
{
    detector = FeatureDetector::create("PyramidFAST");
    extractor = DescriptorExtractor::create("ORB");

    std::cout
        << " -------------------- extract feature points for all images -------------------\n";

    detector->detect(imgs, imgpts);
    extractor->compute(imgs, imgpts, descriptors);
}

void RichFeatureMatcher::MatchFeatures(int idx_i, int idx_j, vector<DMatch>* matches)
{

#ifdef __SFM__DEBUG__
    const Mat& img_1 = imgs[idx_i];
    const Mat& img_2 = imgs[idx_j];
#endif
    const vector<KeyPoint>& imgpts1 = imgpts[idx_i];
    const vector<KeyPoint>& imgpts2 = imgpts[idx_j];
    const Mat& descriptors_1 = descriptors[idx_i];
    const Mat& descriptors_2 = descriptors[idx_j];

    std::vector<DMatch> good_matches_, very_good_matches_;
    std::vector<KeyPoint> keypoints_1, keypoints_2;

    cout << "imgpts1 has " << imgpts1.size() << " points (descriptors "
        << descriptors_1.rows << ")" << endl;
    cout << "imgpts2 has " << imgpts2.size() << " points (descriptors "
        << descriptors_2.rows << ")" << endl;

    keypoints_1 = imgpts1;
    keypoints_2 = imgpts2;

    if (descriptors_1.empty()) 
    {
        CV_Error(0, "descriptors_1 is empty");
    }
    if (descriptors_2.empty()) 
    {
        CV_Error(0, "descriptors_2 is empty");
    }

    //matching descriptor vectors using Brute Force matcher
    BFMatcher matcher(NORM_HAMMING, true); //allow cross-check
    std::vector < DMatch > matches_;
    if (matches == NULL) {
        matches = &matches_;
    }


	// ---------------------------
	// My code
	// ---------------------------

	matches->clear();

	vector < vector<DMatch> > nn_matches;
    matcher.knnMatch(descriptors_1, descriptors_2, nn_matches, 1);


	for(unsigned i = 0; i < nn_matches.size(); i++)
	{
		if(nn_matches[i].size() > 0)
//		if(nn_matches[i][0].distance < 0.75 * nn_matches[i][1].distance)
		{
			good_matches_.push_back(nn_matches[i][0]);
		}
	}

	cout << "Keep " << good_matches_.size() << " out of " << nn_matches.size()
        		<< " matches" << endl;

	*matches = good_matches_;

	return;

	// ---------------------------










    vector<double> dists;
    if (matches->size() == 0) 
    {
        vector < vector<DMatch> > nn_matches;
        matcher.knnMatch(descriptors_1, descriptors_2, nn_matches, 1);
//        /*---------------------------------------------------------------------------*/
//        std::vector<std::pair<int,int> > minDistance1; // holds minDist from first check
//        std::vector<std::pair<int,int> > minDistance2; // holds minDist from cross check
//        getDescriptorDist(descriptors_1, descriptors_2, minDistance1); // returns minDistance1
//        getDescriptorDist(descriptors_2, descriptors_1, minDistance2); // returns minDistance2
//        verifyCrossCheck(minDistance1, minDistance2, nn_matches);
//        /*---------------------------------------------------------------------------*/
        matches->clear();

        for (unsigned int i = 0; i < nn_matches.size(); i++) 
        {
            if (nn_matches[i].size() > 0) 
            {
                matches->push_back(nn_matches[i][0]);
                double dist = matches->back().distance;
                if (fabs(dist) > 10000)
                {
                    dist = 1.0;
                }
                matches->back().distance = dist;
                dists.push_back(dist);
            }
        }
    }

    double max_dist = 0;
    double min_dist = 0.0;
    cv::minMaxIdx(dists, &min_dist, &max_dist);

#ifdef __SFM__DEBUG__
    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );
#endif

    vector<KeyPoint> imgpts1_good, imgpts2_good;

    if (min_dist < 10.0) 
    {
        min_dist = 10.0;
    }

    // Eliminate any re-matching of training points (multiple queries to one training)
    double cutoff = 4.0 * min_dist;
    std::set<int> existing_trainIdx;
    for (unsigned int i = 0; i < matches->size(); i++) 
    {
        //"normalize" matching: somtimes imgIdx is the one holding the trainIdx
        if ((*matches)[i].trainIdx <= 0) 
        {
        	cerr << "Train idx is " <<  (*matches)[i].trainIdx << ", replacing by " << (*matches)[i].imgIdx << endl;
            (*matches)[i].trainIdx = (*matches)[i].imgIdx;
        }

        int tidx = (*matches)[i].trainIdx;
        if ((*matches)[i].distance > 0.0 && (*matches)[i].distance < cutoff) 
        {
            if (existing_trainIdx.find(tidx) == existing_trainIdx.end()
                    && tidx >= 0 && tidx < (int) (keypoints_2.size())) 
            {
                good_matches_.push_back((*matches)[i]);
                //imgpts1_good.push_back(keypoints_1[(*matches)[i].queryIdx]);
                //imgpts2_good.push_back(keypoints_2[tidx]);
                existing_trainIdx.insert(tidx);
            }
        }
    }

    cout << "Keep " << good_matches_.size() << " out of " << matches->size()
        << " matches" << endl;

    *matches = good_matches_;

    return;

#if 0
#ifdef __SFM__DEBUG__
    cout << "keypoints_1.size() " << keypoints_1.size() << " imgpts1_good.size() " << imgpts1_good.size() << endl;
    cout << "keypoints_2.size() " << keypoints_2.size() << " imgpts2_good.size() " << imgpts2_good.size() << endl;

    {
        //-- Draw only "good" matches
        Mat img_matches;
        drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                good_matches_, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //-- Show detected matches
        imshow( "Feature Matches", img_matches );
        waitKey(100);
        destroyWindow("Feature Matches");
    }
#endif

    vector<uchar> status;
    vector<KeyPoint> imgpts2_very_good,imgpts1_very_good;

    //Select features that make epipolar sense
    GetFundamentalMat(imgpts1_good,imgpts2_good,imgpts1_very_good,imgpts2_very_good,good_matches_);

    //Draw matches

#ifdef __SFM__DEBUG__
    {
        //-- Draw only "good" matches
        Mat img_matches;
        drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                good_matches_, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //-- Show detected matches
        imshow( "Good Matches", img_matches );
        waitKey(100);
        destroyWindow("Good Matches");
    }
#endif
#endif
}
