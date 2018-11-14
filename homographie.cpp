#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

#include "image.h"


using namespace std;
using namespace cv;

int main()
{
//	Image<uchar> I1 = Image<uchar>(imread("../image1.jpg", CV_LOAD_IMAGE_GRAYSCALE));
//	Image<uchar> I2 = Image<uchar>(imread("../image2.jpg", CV_LOAD_IMAGE_GRAYSCALE));
//	cout << I1.rows << I1.cols << endl;
//	cout << I2.rows << I2.cols << endl;
//
//	namedWindow("I1", 1);
//	namedWindow("I2", 1);
//	imshow("I1", I1);
//	imshow("I2", I2);
//    waitKey(0);
//
//	Ptr<AKAZE> D = AKAZE::create();
//	// ...
//	vector<KeyPoint> m1, m2;
//
//	Mat desc1, desc2;
//
//    D->detectAndCompute(I1, noArray(), m1, desc1);
//    D->detectAndCompute(I2, noArray(), m2, desc2);
//	// ...
//
//	Mat J1;
//	drawKeypoints(I1, m1, J1);
//	imshow("I1", J1);
//	Mat J2;
//	drawKeypoints(I2, m2, J2);
//	imshow("I2", J2);
//	waitKey(0);
//
//	// Official doc:Brute-force descriptor matcher.
//    //
//    //For each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one.
//	BFMatcher matcher(NORM_HAMMING);
//    vector<DMatch>  matches;
//    matcher.match(desc1, desc2, matches);
//
//    // drawMatches ...
//    Mat res;
//    drawMatches(I1, m1, I2, m2, matches, res);
//    imshow("match", res);
//    waitKey(0);
//
//    double max_dist = 0; double min_dist = 100;
//
//    //-- Quick calculation of max and min distances between keypoints
//    for( int i = 0; i < matches.size(); i++ )
//    { double dist = matches[i].distance;
//        if( dist < min_dist ) min_dist = dist;
//        if( dist > max_dist ) max_dist = dist;
//    }
//
//    printf("-- Max dist : %f \n", max_dist );
//    printf("-- Min dist : %f \n", min_dist );
//
//    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
//    vector< DMatch > good_matches;
//
//    for( int i = 0; i < matches.size(); i++ )
//    { if( matches[i].distance < 3*min_dist )
//        { good_matches.push_back( matches[i]); }
//    }
//
//
//    //-- Localize the object
//    std::vector<Point2f> obj;
//    std::vector<Point2f> scene;
//
//    for( int i = 0; i < good_matches.size(); i++ )
//    {
//        //-- Get the keypoints from the good matches
//        obj.push_back( m1[ good_matches[i].queryIdx ].pt );
//        scene.push_back( m2[ good_matches[i].trainIdx ].pt );
//    }
//
//
//    // Mat H = findHomography(...
//    vector< Point2f> keypts1, keypts2;
//    for (int i =0; i<matches.size(); i++){
//        keypts1.push_back(m1[matches[i].queryIdx].pt);
//        keypts2.push_back(m2[matches[i].trainIdx].pt);
//    }
//
//    Mat mask;
//    //Mat H = findHomography(keypts1, keypts2, CV_RANSAC, 3, mask);
//    Mat F = findFundamentalMat(obj, scene, CV_FM_RANSAC, 3., 0.99, mask);
//    cout << F << endl;
//    //cout << "Homography matrix" << H << endl;
//
//    vector< Point2f> correct_matches1, correct_matches2;
//    vector< DMatch> correct_matches;
//    for(int i = 0; i < mask.rows; i++){
//        if( mask.at<uchar>(i, 0) > 0){
//            correct_matches1.push_back(obj[i]);
//            correct_matches2.push_back(scene[i]);
//            correct_matches.push_back(matches[i]);
//        }
//    }
//    drawMatches(I1, m1, I2, m2, correct_matches, res);
//    imshow("correct match", res);
//    waitKey(0);
//
//    Mat H1, H2;
//    stereoRectifyUncalibrated(correct_matches1, correct_matches2, F, I1.size(), H1, H2, 1);
//    Mat R1(2*I1.cols, 2*I1.rows, CV_8U);
//    Mat R2(2*I2.cols, 2*I2.rows, CV_8U);
//    warpPerspective(I1, R1, H1, R1.size());
//    warpPerspective(I2, R2, H2, R2.size());
//    imshow("R1", R1);
//    imshow("R2", R2);
//    imwrite("../rectified1.png", R1);
//    imwrite("../rectified2.png", R2);
//
//	// merge two images
//	Mat K(2 * I1.cols, I1.rows, CV_8U);
//    Mat idmatrix = Mat::eye(3,3,CV_32F);
//	warpPerspective(I1, K, idmatrix, Size(2*I1.cols, I1.rows));
//	warpPerspective(I2, K, H, Size(2*I1.cols, I1.rows), CV_INTER_LINEAR+CV_WARP_INVERSE_MAP, BORDER_TRANSPARENT);
//
//	imshow("merge I1 and I2", K);
//
//	waitKey(0);
    Image<uchar> I1 = Image<uchar>(imread("../l.jpg", CV_LOAD_IMAGE_GRAYSCALE));
    Image<uchar> I2 = Image<uchar>(imread("../r.jpg", CV_LOAD_IMAGE_GRAYSCALE));


    imshow("Left", I1);
    imshow("Right", I2);
    waitKey(0);

    //Ptr<StereoBM> SBM = StereoBM::create();
    //SBM->compute(R1, R2, disparity);
    Ptr<StereoSGBM> sgbm_ = StereoSGBM::create();
    sgbm_->setBlockSize(5);
    sgbm_->setDisp12MaxDiff(-1);
    sgbm_->setP1(0);
    sgbm_->setP2(0);
    sgbm_->setMinDisparity(-46);
    sgbm_->setNumDisparities(128);
    sgbm_->setUniquenessRatio(15);
    sgbm_->setPreFilterCap(30);
    sgbm_->setSpeckleRange(0);
    sgbm_->setSpeckleWindowSize(0);

    Mat disparity_16S;
    sgbm_->compute(I1, I2, disparity_16S);

    int a = 170;
    int b = 180;
    for(int i=a-50; i<a+50; i++){
        for(int j=b-50; j<b+50; j++){
            cout << disparity_16S.at<short>(i,j)/16.0f << endl;
        }
    }
    cout << "rows" << I1.rows << endl;
    cout << "cols" << I1.cols << endl;
    //Finding corresponding points

    Point t1(a,b);
    circle(I1, t1, 2, Scalar(0,0,0), 2);
    imshow("I1", I1);
    waitKey(0);

    cout << "disparity = " << disparity_16S.at<short>(b, a) << endl;
    Point t2(a-(disparity_16S.at<short>(b, a)/16), b);
    circle(I2, t2, 2, Scalar(0,0,0), 2);
    imshow("I2", I2);
    waitKey(0);

    Mat disparity8U;
    disparity_16S.convertTo(disparity8U, CV_8U);
    imshow("disparity", Image<short>(disparity_16S).greyImage());
    waitKey(0);



	return 0;
}
