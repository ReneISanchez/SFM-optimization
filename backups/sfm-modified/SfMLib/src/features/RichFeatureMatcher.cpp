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

#include "mmintrin.h" // Intel intrinsics - (mmx) pxor
#include "nmmintrin.h" // Intel intrinsics - popcnt
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

using namespace std;
using namespace cv;

// Get Hamming distances
static void getDescriptorDist(const cv::Mat& descriptors_1, 
        const cv::Mat& descriptors_2, std::vector<std::pair<int,int> >& minDist) {

    for (int x = 0; x < descriptors_1.rows; x++)
    {
        int min = INT_MAX;
        int yIdx = 0;

        __m128i mDes1_first, mDes1_sec;

        mDes1_first = _mm_set_epi8(
                descriptors_1.at<unsigned char>(x,0), 
                descriptors_1.at<unsigned char>(x,1), 
                descriptors_1.at<unsigned char>(x,2), 
                descriptors_1.at<unsigned char>(x,3), 
                descriptors_1.at<unsigned char>(x,4), 
                descriptors_1.at<unsigned char>(x,5), 
                descriptors_1.at<unsigned char>(x,6), 
                descriptors_1.at<unsigned char>(x,7), 
                descriptors_1.at<unsigned char>(x,8),
                descriptors_1.at<unsigned char>(x,9), 
                descriptors_1.at<unsigned char>(x,10), 
                descriptors_1.at<unsigned char>(x,11), 
                descriptors_1.at<unsigned char>(x,12), 
                descriptors_1.at<unsigned char>(x,13), 
                descriptors_1.at<unsigned char>(x,14), 
                descriptors_1.at<unsigned char>(x,15)); 

        mDes1_sec = _mm_set_epi8(
                descriptors_1.at<unsigned char>(x,16), 
                descriptors_1.at<unsigned char>(x,17), 
                descriptors_1.at<unsigned char>(x,18), 
                descriptors_1.at<unsigned char>(x,19), 
                descriptors_1.at<unsigned char>(x,20), 
                descriptors_1.at<unsigned char>(x,21), 
                descriptors_1.at<unsigned char>(x,22), 
                descriptors_1.at<unsigned char>(x,23), 
                descriptors_1.at<unsigned char>(x,24), 
                descriptors_1.at<unsigned char>(x,25), 
                descriptors_1.at<unsigned char>(x,26), 
                descriptors_1.at<unsigned char>(x,27), 
                descriptors_1.at<unsigned char>(x,28), 
                descriptors_1.at<unsigned char>(x,29), 
                descriptors_1.at<unsigned char>(x,30), 
                descriptors_1.at<unsigned char>(x,31)); 

        for (int y = 0 ; y < descriptors_2.rows ; y++)
        {
            int sumHamDist1 = 0;
            int sumHamDist2 = 0;
            int sumHamDist = 0;

            __m128i mDes2_first, mDes2_sec, c1, c2;

            mDes2_first = _mm_set_epi8(
                    descriptors_2.at<unsigned char>(y,0), 
                    descriptors_2.at<unsigned char>(y,1), 
                    descriptors_2.at<unsigned char>(y,2), 
                    descriptors_2.at<unsigned char>(y,3),
                    descriptors_2.at<unsigned char>(y,4), 
                    descriptors_2.at<unsigned char>(y,5), 
                    descriptors_2.at<unsigned char>(y,6), 
                    descriptors_2.at<unsigned char>(y,7), 
                    descriptors_2.at<unsigned char>(y,8),
                    descriptors_2.at<unsigned char>(y,9), 
                    descriptors_2.at<unsigned char>(y,10), 
                    descriptors_2.at<unsigned char>(y,11), 
                    descriptors_2.at<unsigned char>(y,12), 
                    descriptors_2.at<unsigned char>(y,13), 
                    descriptors_2.at<unsigned char>(y,14), 
                    descriptors_2.at<unsigned char>(y,15)); 

            mDes2_sec = _mm_set_epi8(
                    descriptors_2.at<unsigned char>(y,16), 
                    descriptors_2.at<unsigned char>(y,17), 
                    descriptors_2.at<unsigned char>(y,18), 
                    descriptors_2.at<unsigned char>(y,19), 
                    descriptors_2.at<unsigned char>(y,20), 
                    descriptors_2.at<unsigned char>(y,21), 
                    descriptors_2.at<unsigned char>(y,22), 
                    descriptors_2.at<unsigned char>(y,23), 
                    descriptors_2.at<unsigned char>(y,24), 
                    descriptors_2.at<unsigned char>(y,25), 
                    descriptors_2.at<unsigned char>(y,26), 
                    descriptors_2.at<unsigned char>(y,27), 
                    descriptors_2.at<unsigned char>(y,28), 
                    descriptors_2.at<unsigned char>(y,29), 
                    descriptors_2.at<unsigned char>(y,30), 
                    descriptors_2.at<unsigned char>(y,31)); 

            c1 = _mm_xor_si128(mDes1_first, mDes2_first);
            c2 = _mm_xor_si128(mDes1_sec, mDes2_sec);

            sumHamDist1 = _mm_popcnt_u64(_mm_extract_epi64(c1,0)) + 
                _mm_popcnt_u64(_mm_extract_epi64(c1,1)); //1 shifts first half, 0 doesn't

            sumHamDist2 = _mm_popcnt_u64(_mm_extract_epi64(c2,0)) + 
                _mm_popcnt_u64(_mm_extract_epi64(c2,1)); //1 shifts first half, 0 doesn't

            // sum the Hamming distances of the 2 halves of xored string
            sumHamDist = sumHamDist1 + sumHamDist2;

            if (sumHamDist <= min)
            {
                min = sumHamDist;
                yIdx = y;
            }
        }
        minDist.push_back(make_pair(yIdx, min));
    }
} //--endGetDescriptorDist

static void verifyCrossCheck(vector<pair<int,int> >& minDist1, 
        vector<pair<int,int> >& minDist2, vector < vector<DMatch> >& nn_matches) {

    // size cutoff at smaller vector
    for (unsigned int x = 0; x < minDist1.size() && x < minDist2.size(); x++)
    {
        vector<DMatch> tempVect(1);
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
            tempVect[0] = DMatch( x, minDist1[x].first, static_cast<float>(tempDis2));
        }
        else
        {
            //tempVect.push_back(DMatch( minDistance1[x].first, x, (float)tempDis1) ); //orig
            tempVect[0] = DMatch( x, minDist1[x].first, static_cast<float>(tempDis1));
        }

        nn_matches.push_back(tempVect);
    }
} //endVerifyCrossCheck
/*---------------------------------------------------------------------------*/

//c'tor
RichFeatureMatcher::RichFeatureMatcher(std::vector<cv::Mat>& imgs_, std::vector<std::vector<cv::KeyPoint> >& imgpts_) :
    imgs(imgs_), imgpts(imgpts_) 
{
    detector = FeatureDetector::create("PyramidFAST");
    extractor = DescriptorExtractor::create("ORB");
    std::cout << "-------------------- extract feature points for all images -------------------\n";

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

    vector < vector<DMatch> > nn_matches;
    vector<DMatch> good_matches_, very_good_matches_;
    vector<KeyPoint> keypoints_1, keypoints_2;

    cout << "imgpts1 has " << imgpts1.size() << " points (descriptors " << descriptors_1.rows << ")" << endl;
    cout << "imgpts2 has " << imgpts2.size() << " points (descriptors " << descriptors_2.rows << ")" << endl;

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

    vector < DMatch > matches_;

    if (matches == NULL) {
        matches = &matches_;
    }

    vector<double> dists;
    if (matches->size() == 0) 
    {
        //matcher.knnMatch(descriptors_1, descriptors_2, nn_matches, 1);
        //nn_matches.clear();
        /*---------------------------------------------------------------------------*/

        std::vector<std::pair<int,int> > minDistance1; // holds minDist from first check
        std::vector<std::pair<int,int> > minDistance2; // holds minDist from cross check
        getDescriptorDist(descriptors_1, descriptors_2, minDistance1); // returns minDistance1
        getDescriptorDist(descriptors_2, descriptors_1, minDistance2); // returns minDistance2
        verifyCrossCheck(minDistance1, minDistance2, nn_matches);

        /*---------------------------------------------------------------------------*/
        matches->clear();

        for (unsigned int i = 0; i < nn_matches.size(); i++) 
        {
            if (nn_matches[i].size() > 0) 
            {
                matches->push_back(nn_matches[i][0]);
                double dist = matches->back().distance;
                if (abs(dist) > 10000)
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
            (*matches)[i].trainIdx = (*matches)[i].imgIdx;
        }

        int tidx = (*matches)[i].trainIdx;
        if ((*matches)[i].distance > 0.0 && (*matches)[i].distance < cutoff) 
        {
            if (existing_trainIdx.find(tidx) == existing_trainIdx.end()
                    && tidx >= 0 && tidx < (int) (keypoints_2.size())) 
            {
                good_matches_.push_back((*matches)[i]);
                existing_trainIdx.insert(tidx);
            }
        }
    }

    cout << "Keep " << good_matches_.size() << " out of " << matches->size() << " matches" << endl;

    *matches = good_matches_;

    return;

