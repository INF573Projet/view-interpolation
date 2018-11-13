#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>

#include "image.h"


using namespace std;
using namespace cv;

bool distance_for_matches(DMatch d_i, DMatch d_j) {
    return d_i.distance < d_j.distance;
}

struct Data {
	Mat R1, R2;
	Mat disparity;
};

void onMouse(int event, int x, int y, int foo, void* p)
{
	if (event != CV_EVENT_LBUTTONDOWN)
		return;
	Point m1(x, y);
	Data* D = (Data*)p;
	circle(D->R1, m1, 2, Scalar(0, 255, 0), 2);
	imshow("R1", D->R1);

	short d = D->disparity.at<short>(y, x);
	Point m2(x + d, y);
    circle(D->R2,m2,2,Scalar(0,255,0),2);
	imshow("R2", D->R2);
}

int main()
{
//	Image<uchar> I1 = Image<uchar>(imread("../face00.tif", CV_LOAD_IMAGE_GRAYSCALE));
//	Image<uchar> I2 = Image<uchar>(imread("../face01.tif", CV_LOAD_IMAGE_GRAYSCALE));

    Image<uchar> I1 = Image<uchar>(imread("../images/perra_7.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	Image<uchar> I2 = Image<uchar>(imread("../images/perra_8.jpg", CV_LOAD_IMAGE_GRAYSCALE));

	cout << "Size of img1: " << I1.rows << "*" << I1.cols << endl;
	cout << "Size of img2: " << I2.rows << "*" << I2.cols << endl;

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
	BFMatcher matcher(NORM_HAMMING, true);
    vector<DMatch>  matches;
    matcher.match(desc1, desc2, matches);
    cout << matches.size() << " matches for BF Matching" << endl;

    // drawMatches ...
    Mat res;
    drawMatches(I1, m1, I2, m2, matches, res);
    imshow("match", res);
    waitKey(0);

    sort(matches.begin(), matches.end(), distance_for_matches);

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    int max_good_matches = 300;
    vector<DMatch>  good_matches;
    for( int i = 0; i < max_good_matches; i++ )
    {
        good_matches.push_back( matches[i]);
    }
    cout << "The size of the good_matches is: " << good_matches.size() << endl;

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( m1[ good_matches[i].queryIdx ].pt );
        scene.push_back( m2[ good_matches[i].trainIdx ].pt );
    }

    Mat mask;
    Mat F = findFundamentalMat(obj, scene, CV_FM_RANSAC, 3., 0.99, mask);
    cout << "Fundamental matrix: " << F << endl;

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

    // Rectify the images
    Mat H1, H2;
    stereoRectifyUncalibrated(correct_matches1, correct_matches2, F, I1.size(), H1, H2, 1);

    Mat R1, R2;
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

    // Compute disparity mapping
    Mat disparity;
    Ptr<StereoSGBM> SGBM = StereoSGBM::create();
    SGBM->compute(R1, R2, disparity);

    Mat disparity8u;
    disparity.convertTo(disparity8u, CV_8U);
    imshow("disparity", disparity8u);

    Data data;
    data.R1 = R1;
    data.R2 = R2;
    data.disparity = disparity;
	setMouseCallback("R1", onMouse, &data);
	waitKey(0);

	return 0;
}
