//
// Created by leon on 12/21/18.
//

#ifndef VIEW_INTERPOLATION_INTERPOLATION_H
#define VIEW_INTERPOLATION_INTERPOLATION_H


#include "image.h"
#include "Eigen/src/Core/Matrix.h"


struct RectParam{
    //TODO : check the right type for T1 and T2
    double theta1;
    double theta2;
    Eigen::Vector2d T1;
    Eigen::Vector2d T2;
    double s;
    double i;

};

class Interpolation{
public:
    static void matchingKeypoints(const Image<uchar>&I1, const Image<uchar>& I2, int max_good_matches, vector<Point2f>& kptsL, vector<Point2f>& kptsR, bool display=false);
    static void rectify(const Image<uchar>&I1, const Image<uchar>& I2, const vector<Point2f>& kptsL, const vector<Point2f>& kptsR, Image<uchar>& R1, Image<uchar>& R2, RectParam &D);
    static void disparityMapping(const Image<uchar>& R1, const Image<uchar>& R2, Image<short>& disparity);
    static void interpolate(double i, const Image<uchar>& R1, const Image<uchar>& R2, const Image<short>& disparity, Image<uchar>& IR, RectParam& D);
    static void derectify(const Image<uchar>& IR, const RectParam &D, Image<uchar>& I);

};

inline bool distance_for_matches(DMatch d_i, DMatch d_j) {
    return d_i.distance < d_j.distance;
}


#endif //VIEW_INTERPOLATION_INTERPOLATION_H
