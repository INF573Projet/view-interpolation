#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "Eigen/Core"
#include "Eigen/SVD"
#include "Eigen/Dense"
#include <math.h>
#include <iostream>

#include "image.h"

using namespace std;
using namespace Eigen;
using namespace cv;

bool distance_for_matches(DMatch d_i, DMatch d_j) {
    return d_i.distance < d_j.distance;
}

//struct Data {
//	Mat R1, R2;
//	Mat disparity;
//};
//
//void onMouse(int event, int x, int y, int foo, void* p)
//{
//	if (event != CV_EVENT_LBUTTONDOWN)
//		return;
//	Point m1(x, y);
//	Data* D = (Data*)p;
//	circle(D->R1, m1, 2, Scalar(0, 255, 0), 2);
//	imshow("R1", D->R1);
////	circle(D->disparity, m1, 2, Scalar(0, 255, 0), 2);
////	imshow("disparity", D->disparity);
//
//	short d = D->disparity.at<short>(y, x);
//    cout<<"Disparity at point (" << x << "; " << y << "): " << d << endl;
//	Point m2(x - d, y);
//    circle(D->R2,m2,2,Scalar(0,255,0),2);
//	imshow("R2", D->R2);
//}

void matchingKeypoints(const Image<uchar>&I1, const Image<uchar>& I2, int max_good_matches, vector<Point2f>& kptsL, vector<Point2f>& kptsR, bool display=false) {
    //Finding keypoints
    Ptr<AKAZE> D = AKAZE::create();
    vector<KeyPoint> m1, m2;
    Mat desc1, desc2;
    D->detectAndCompute(I1, noArray(), m1, desc1);
    D->detectAndCompute(I2, noArray(), m2, desc2);
    if(display){
        //Displaying keypoints
        Mat J1;
        drawKeypoints(I1, m1, J1);
        imshow("I1", J1);
        Mat J2;
        drawKeypoints(I2, m2, J2);
        imshow("I2", J2);
        waitKey(0);
    }

    //For each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one.
    BFMatcher matcher(NORM_HAMMING, true);
    vector<DMatch>  matches;
    matcher.match(desc1, desc2, matches);
    cout << matches.size() << " matches for BF Matching" << endl;
    if(display){
        //Displaying matches
        Mat res;
        drawMatches(I1, m1, I2, m2, matches, res);
        imshow("match", res);
        waitKey(0);
    }

    sort(matches.begin(), matches.end(), distance_for_matches);
    //-- Draw only "good" matches (first max_good_matches)
    vector<DMatch>  good_matches;
    for( int i = 0; i < max_good_matches; i++ )
    {
        good_matches.push_back( matches[i]);
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

    Mat mask;
    Mat F = findFundamentalMat(obj, scene, CV_FM_RANSAC, 3., 0.99, mask);
    cout << "Fundamental matrix: " << F << endl;
    vector<DMatch> correct_matches;
    for(int i = 0; i < mask.rows; i++){
        if( mask.at<uchar>(i, 0) > 0){
            kptsL.push_back(obj[i]);
            kptsR.push_back(scene[i]);
            correct_matches.push_back(matches[i]);
        }
    }
    cout << "Final number of keypoints matches: " << correct_matches.size() << endl;

    if(display){
        Mat res;
        drawMatches(I1, m1, I2, m2, correct_matches, res);
        imshow("correct match", res);
        waitKey(0);
    }

}

void rectify(const Image<uchar>&I1, const Image<uchar>& I2, const vector<Point2f>& kptsL, const vector<Point2f>& kptsR, Image<uchar>& R1, Image<uchar>& R2) {

    // compute mean of x_value and y_value of key points in imgL
    double x_meanL = 0.;
    double y_meanL = 0.;
    for (auto &i : kptsL) {
        x_meanL += i.x;
        y_meanL += i.y;
    }
    // compute mean of x_value and y_value of key points in imgR
    double x_meanR = 0.;
    double y_meanR = 0.;
    for (auto &i : kptsR) {
        x_meanR += i.x;
        y_meanR += i.y;
    }

    // measurement matrix
    MatrixXd M(4, kptsL.size());
    for(int i=0; i<kptsL.size(); i++){
        M(0, i) = kptsL[i].x - x_meanL;
        M(1, i) = kptsL[i].y - y_meanL;
        M(2, i) = kptsR[i].x - x_meanR;
        M(3, i) = kptsR[i].y - y_meanR;
    }

    // Singular value decomposition of M
    JacobiSVD<MatrixXd> svd( M, ComputeFullV | ComputeFullU );
    cout << svd.computeU() << endl;
    MatrixXd U = svd.matrixU();

    //    U_ = U[:, :3], U1 = U_[:2, :], U2 = U_[2:, :]
    MatrixXd U_ = U.leftCols(3);
    MatrixXd U1 = U_.topRows(2);
    MatrixXd U2 = U_.bottomRows(2);

    // A1 = U1[:2, :2], d1 = U1[:, 2], A2 = U2[:2, :2], d2 = U2[:, 2]
    MatrixXd A1 = U1.block(0,0,2,2);
    MatrixXd d1 = U1.col(2);
    MatrixXd A2 = U2.block(0,0,2,2);
    MatrixXd d2 = U2.col(2);

    //    define B_i, U_1' and U_2'
    Matrix3d B1;
    //    B1[-1, -1] = 1, B1[:2, :2] = np.linalg.inv(A1), B1[:2, 2] = -np.dot(np.linalg.inv(A1), d1)
    B1(2,2) = 1;
    B1(2,0) = 0;
    B1(2,1) = 0;
    B1.block(0,0,2,2) = A1.inverse();
    B1.block(0,2,2,1) = A1.inverse()*d1;

    Matrix3d B2;
    B2(2,2) = 1;
    B2(2,0) = 0;
    B2(2,1) = 0;
    B2.block(0,0,2,2) = A2.inverse();
    B2.block(0,2,2,1) = A2.inverse()*d2;

    // U1_prime = np.dot(U1, B2), U2_prime = np.dot(U2, B1)
    MatrixXd U1_prime = U1*B2;
    MatrixXd U2_prime = U2*B1;

    //calculate theta1, theta2, x1 = U1_prime[0, -1], y1 = U1_prime[1, -1]
    //theta1 = np.arctan(y1 / x1)
    double x1 = U1_prime(0, 2); double y1 = U1_prime(1, 2);
    double theta1 = atan(y1 / x1);

    double x2 = U1_prime(0, 2); double y2 = U1_prime(1, 2);
    double theta2 = atan(y2 / x2);

    //rotation matrix, R1 = np.array([[cos(theta1), sin(theta1)],
    //                              [-sin(theta1), cos(theta1)]])
    Matrix2d rot1, rot2;
    rot1(0,0) = cos(theta1); rot1(0,1) = sin(theta1);
    rot1(1,0) = -sin(theta1);rot1(1,1) = cos(theta1);

    rot2(0,0) = cos(theta1); rot2(0,1) = sin(theta1);
    rot2(1,0) = -sin(theta1);rot2(1,1) = cos(theta1);

    //calculate B and B_inv, B[:2, :] = np.dot(R1, U1_prime), B[2, :] = np.dot(R2, U2_prime)[0, :]
    //    try:
    //        B_inv = np.linalg.inv(B)
    //    except LinAlgError:
    //        B[2, :] = np.array([0, 0, 1])
    //        B_inv = np.linalg.inv(B)
    Matrix3d B, B_inv;
    B.block(0,0,2,3) = rot1*U1_prime;
    B.bottomRows(1) = (rot2*U2_prime).topRows(1);
    if(B.determinant()!=0){
        B_inv = B.inverse();
    }else{
        B(2,0) = 0; B(2,1) = 0; B(2,2) = 1;
        B_inv = B.inverse();
    }
    // calculate s and H_s, tmp = np.dot(R2, np.dot(U2_prime, B_inv)), s = tmp[1, 1]
    //    H_s = np.array([[1, 0],
    //                    [0, 1. / s]])
    MatrixXd tmp;
    tmp = rot2*(U2_prime*B_inv);
    double s = tmp(1,1);
    MatrixXd H_s(2,2);
    H_s(0,0) = 1;H_s(0,1) = 0;
    H_s(1,0) = 0;H_s(1,1) = 1. / s;


    // TODO: rectify two images based on above geometry matrix;


}

void rectififyImages(const Image<uchar>&I1, const Image<uchar>& I2, Mat& R1, Mat& R2){
    cout << "Size of img1: " << I1.rows << "*" << I1.cols << endl;
	cout << "Size of img2: " << I2.rows << "*" << I2.cols << endl;

	namedWindow("I1", 1);
	namedWindow("I2", 1);
	imshow("I1", I1);
	imshow("I2", I2);
    waitKey(0);

	Ptr<AKAZE> D = AKAZE::create();
	vector<KeyPoint> m1, m2;

	Mat desc1, desc2;

    D->detectAndCompute(I1, noArray(), m1, desc1);
    D->detectAndCompute(I2, noArray(), m2, desc2);

	Mat J1;
	drawKeypoints(I1, m1, J1);
	imshow("I1", J1);
	Mat J2;
	drawKeypoints(I2, m2, J2);
	imshow("I2", J2);
	waitKey(0);

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

    //-- Draw only "good" matches (first max_good_matches)
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
    warpPerspective(I1, R1, H1, R1.size());
    warpPerspective(I2, R2, H2, R2.size());

}

void disparityMapping(const Image<uchar>& R1, const Image<uchar>& R2, Image<short>& disparity){
    Ptr<StereoSGBM> sgbm_ = StereoSGBM::create();
    sgbm_->setBlockSize(5);
    sgbm_->setDisp12MaxDiff(-1);
    sgbm_->setP1(600);
    sgbm_->setP2(2400);
    sgbm_->setMinDisparity(-64);
    sgbm_->setNumDisparities(192);
    sgbm_->setUniquenessRatio(1);
    sgbm_->setPreFilterCap(4);
    sgbm_->setSpeckleRange(2);
    sgbm_->setSpeckleWindowSize(150);

    sgbm_->compute(R1, R2, disparity);

    for(int x=0; x<disparity.width(); x++){
        for(int y=0; y<disparity.height(); y++){
            disparity(x,y) = short(disparity(x,y) / 16);
        }
    }
}

void interpolate(float i, const Image<uchar>& R1, const Image<uchar>& R2, const Image<short>& disparity, Image<uchar>& IR){
    for(int y=0; y<R1.height(); y++){
        for(int x1=0; x1<R1.width(); x1++){
            int x2 = x1 - disparity(x1,y);
            int x_i = int((2-i)*x1 + (i-1)*x2);
            if(x_i >=0 && x_i < IR.width()){
                IR(x_i, y) = uchar((2-i)*R1(x1,y) + (i-1)*R2(x2,y));
            } else {
                IR(x_i, y) = 0;
            }
        }
    }
}

void derectify(const Image<uchar>& IR, Image<uchar>& I){


}

int main()
{
    /* Parameters */
    int max_good_matches = 300;
    float i = 1.5;


    cout << "Reading left and right images..." << endl;
    Image<uchar> I1 = Image<uchar>(imread("../images/perra_7.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	Image<uchar> I2 = Image<uchar>(imread("../images/perra_8.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	imshow("left", I1);
    imshow("right", I2);
    waitKey(0);

	cout << "Finding keypoints and matching them..." << endl;
    vector<Point2f> kptsL, kptsR;
    matchingKeypoints(I1, I2, max_good_matches, kptsL, kptsR, false);

    cout << "Rectifying images..." << endl;
    Image<uchar> R1, R2;
    rectify(I1, I2, kptsL, kptsR, R1, R2);
    imshow("left_rectified", R1);
    imshow("right_rectified", R2);
    waitKey(0);g
//
//    cout << "Computing disparity..." << endl;
//    Image<short> disparity;
//    disparityMapping(R1, R2, disparity);
//    imshow("disparity", Image<short>(disparity).greyImage());
//    waitKey(0);
//
//    cout << "Interpolating rectified intermediate view..." << endl;
//    Image<uchar> IR;
//    interpolate(i, R1, R2, disparity, IR);
//    imshow("left_rectified + right_rectified", IR);
//    waitKey(0);
//
//    cout << "Derectifying interpolated view..." << endl;
//    Image<uchar> I;
//    derectify(IR, I);
//    imshow("left + rrectified", I);
//    waitKey(0);

	return 0;
}
