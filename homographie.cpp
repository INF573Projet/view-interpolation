#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>

#include "image.h"


using namespace std;
using namespace cv;

int main()
{
//	Image<uchar> I1 = Image<uchar>(imread("../face00.tif", CV_LOAD_IMAGE_GRAYSCALE));
//	Image<uchar> I2 = Image<uchar>(imread("../face01.tif", CV_LOAD_IMAGE_GRAYSCALE));

    Image<uchar> I1 = Image<uchar>(imread("../perra_7.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	Image<uchar> I2 = Image<uchar>(imread("../perra_8.jpg", CV_LOAD_IMAGE_GRAYSCALE));

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
	BFMatcher matcher(NORM_L2);
    vector<DMatch>  matches;
    matcher.match(desc1, desc2, matches);

    // drawMatches ...
    Mat res;
    drawMatches(I1, m1, I2, m2, matches, res);
    imshow("match", res);
    waitKey(0);

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < matches.size(); i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    vector< DMatch > good_matches;

    for( int i = 0; i < matches.size(); i++ )
    { if( matches[i].distance < 3*min_dist )
        { good_matches.push_back( matches[i]); }
    }


    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( m1[ good_matches[i].queryIdx ].pt );
        scene.push_back( m2[ good_matches[i].trainIdx ].pt );
    }


    // Mat H = findHomography(...
//    vector< Point2f> keypts1, keypts2;
//    for (int i =0; i<matches.size(); i++){
//        keypts1.push_back(m1[matches[i].queryIdx].pt);
//        keypts2.push_back(m2[matches[i].trainIdx].pt);
//    }

    Mat mask;
    Mat H = findHomography(obj, scene, CV_RANSAC, 3);
    cout << H << endl;
    Mat F = findFundamentalMat(obj, scene, CV_FM_RANSAC, 3., 0.99, mask);
    cout << F << endl;
    //cout << "Homography matrix" << H << endl;

    vector< Point2f> correct_matches1, correct_matches2;
    vector< DMatch> correct_matches;
    for(int i = 0; i < mask.rows; i++){
        if( mask.at<uchar>(i, 0) > 0){
            correct_matches1.push_back(obj[i]);
            correct_matches2.push_back(scene[i]);
            correct_matches.push_back(matches[i]);
        }
    }
    drawMatches(I1, m1, I2, m2, correct_matches, res);
    imshow("correct match", res);
    waitKey(0);

    Mat H1, H2;
    stereoRectifyUncalibrated(correct_matches1, correct_matches2, F, I1.size(), H1, H2, 1);
    Mat R1(I1.cols, I1.rows, CV_8U);
    Mat R2(I2.cols, I2.rows, CV_8U);
    warpPerspective(I1, R1, H1, R1.size());
    warpPerspective(I2, R2, H2, R2.size());
    imshow("R1", R1);
    imshow("R2", R2);

    imwrite("../rectified1.png", R1);
    imwrite("../rectified2.png", R2);

	// merge two images
//	Mat K(2 * I1.cols, I1.rows, CV_8U);
//    Mat idmatrix = Mat::eye(3,3,CV_32F);
//	warpPerspective(I1, K, idmatrix, Size(2*I1.cols, I1.rows));
//	warpPerspective(I2, K, H, Size(2*I1.cols, I1.rows), CV_INTER_LINEAR+CV_WARP_INVERSE_MAP, BORDER_TRANSPARENT);
//
//	imshow("merge I1 and I2", K);



//    Interpolation for rectified images
//    Mat IR(2*I1.cols, 2*I1.rows, CV_8U);
//    double i = 2;
//    for(int y=0; y<R1.rows; y++){
//        for(int x1=0; x1<R1.cols; x1++){
//            Vec3d p1(x1,y,1);
//            Mat p2 = A*Mat(p1);
//            int x2 = int(p2.at<double>(0,0) / p2.at<double>(2,0));
//            int x_i = int((2-i)*x1 + (i-1)*x2);
//            IR.at<uchar>(y,x_i) = (2-i)*R1.at<uchar>(y,x1) + (i-1)*R2.at<uchar>(y,x2);
//        }
//    }
//    imshow("IR", IR);
//    waitKey(0);

    //Finding disparities
//    Mat R1 = imread("../chess_1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//	Mat R2 = imread("../chess_2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    Mat disparity;

    Ptr<StereoBM> SBM = StereoBM::create();
    SBM->compute(R1, R2, disparity);

    int a = 200;
    int b = 150;
    for(int i=a-50; i<a+50; i++){
        for(int j=b-50; j<b+50; j++){
            cout << disparity.at<short>(j,i) << endl;
        }
    }

    //Finding corresponding points

	Point t1(a,b);
	circle(R1, t1, 2, Scalar(0,0,0), 2);
	imshow("R1", R1);
	waitKey(0);

	cout << "disparity = " << disparity.at<short>(b, a) << endl;
	Point t2(a-disparity.at<short>(b, a), b);
	circle(R2, t2, 2, Scalar(0,0,0), 2);
	imshow("R2", R2);
	waitKey(0);

	return 0;
}
