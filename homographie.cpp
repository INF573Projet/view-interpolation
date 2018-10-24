#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>

#include "image.h"


using namespace std;
using namespace cv;

int main()
{
	Image<uchar> I1 = Image<uchar>(imread("../face00.tif", CV_LOAD_IMAGE_GRAYSCALE));
	Image<uchar> I2 = Image<uchar>(imread("../face01.tif", CV_LOAD_IMAGE_GRAYSCALE));
	cout << I1.rows << I1.cols << endl;
	cout << I2.rows << I2.cols << endl;
	
	namedWindow("I1", 1);
	namedWindow("I2", 1);
	imshow("I1", I1);
	imshow("I2", I2);
    waitKey(0);

	Ptr<AKAZE> D = AKAZE::create();
	// ...
	vector<KeyPoint> m1, m2;

	Mat desc1, desc2;

    D->detectAndCompute(I1, noArray(), m1, desc1);
    D->detectAndCompute(I2, noArray(), m2, desc2);
	// ...
	
	Mat J1;
	drawKeypoints(I1, m1, J1);
	imshow("I1", J1);
	Mat J2;
	drawKeypoints(I2, m2, J2);
	imshow("I2", J2);
	waitKey(0);

	// Official doc:Brute-force descriptor matcher.
    //
    //For each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one.
	BFMatcher matcher(NORM_HAMMING);
    vector<DMatch>  matches;
    matcher.match(desc1, desc2, matches);

    // drawMatches ...
    Mat res;
    drawMatches(I1, m1, I2, m2, matches, res);
    imshow("match", res);
    waitKey(0);

    // Mat H = findHomography(...
    vector< Point2f> keypts1, keypts2;
    for (int i =0; i<matches.size(); i++){
        keypts1.push_back(m1[matches[i].queryIdx].pt);
        keypts2.push_back(m2[matches[i].trainIdx].pt);
    }

    Mat mask;
    //Mat H = findHomography(keypts1, keypts2, CV_RANSAC, 3, mask);
    Mat F = findFundamentalMat(keypts1, keypts2, CV_RANSAC, 3., 0.99, mask);
    cout << F << endl;
    //cout << "Homography matrix" << H << endl;

    vector< Point2f> correct_matches1, correct_matches2;
    vector< DMatch> correct_matches;
    for(int i = 0; i < mask.rows; i++){
        if( mask.at<uchar>(i, 0) > 0){
            correct_matches1.push_back(keypts1[i]);
            correct_matches2.push_back(keypts2[i]);
            correct_matches.push_back(matches[i]);
        }
    }
    drawMatches(I1, m1, I2, m2, correct_matches, res);
    imshow("correct match", res);
    waitKey(0);

    Mat H1, H2;
    stereoRectifyUncalibrated(correct_matches1, correct_matches2, F, I1.size(), H1, H2);
    Mat R1(I1.cols, I1.rows, CV_8U);
    Mat R2(I2.cols, I2.rows, CV_8U);
    warpPerspective(I1, R1, H1, R1.size());
    warpPerspective(I2, R2, H2, R2.size());
    imshow("R1", R1);
    imshow("R2", R2);
    imwrite("../rectified1.png", R1);
    imwrite("../rectified2.png", R2);
	
//	// merge two images
//	Mat K(2 * I1.cols, I1.rows, CV_8U);
//    Mat idmatrix = Mat::eye(3,3,CV_32F);
//	warpPerspective(I1, K, idmatrix, Size(2*I1.cols, I1.rows));
//	warpPerspective(I2, K, H, Size(2*I1.cols, I1.rows), CV_INTER_LINEAR+CV_WARP_INVERSE_MAP, BORDER_TRANSPARENT);
//
//	imshow("merge I1 and I2", K);

	waitKey(0);
	return 0;
}
