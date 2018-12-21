//
// Created by leon on 12/21/18.
//

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

